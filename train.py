import logging
import os
import typing
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, BinaryAccuracy, \
    BinaryPrecision, BinaryRecall, BinaryConfusionMatrix, MulticlassConfusionMatrix
import albumentations as A
import pandas as pd
import picsellia
import torch
import torchvision
from albumentations import ToTensorV2
from joblib import delayed, Parallel
from picsellia import Client, Experiment, Asset, DatasetVersion
from picsellia.exceptions import ResourceNotFoundError
from picsellia.types.enums import InferenceType, ExperimentStatus, LogType
from pytorch_warmup import ExponentialWarmup
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ClassificationDataset
from logger import PicselliaLogger
from utils import download_datasets, get_class_mapping_from_picsellia

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def fill_picsellia_evaluation_tab(model: torch.nn.Module, validation_loader: DataLoader,
                                  test_dataset: picsellia.DatasetVersion, device: str,
                                  experiment: Experiment, labelmap: dict) -> None:
    """
    Fill Picsellia evaluation which allows comparing on a dedicated bench of images the prediction done by the freshly
    trained model with the ground-truth.
    :param model: Model which will be used to do the predictions
    :param data_loader: Dataloader which gathers the bench of images on which the evaluation will be done
    :param test_dataset_id: ID of the dataset version which comports the pictures on which the evaluation will be done
    """
    label_map = list(labelmap.values())

    if len(label_map) > 2:
        metric = MulticlassConfusionMatrix(num_classes=len(label_map), device=torch.device(device))
    else:
        metric = BinaryConfusionMatrix(device=torch.device(device))

    model.eval()

    with torch.no_grad():
        for inputs, labels, filenames in validation_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            if len(label_map) > 2:
                output_logits = torch.nn.Softmax(dim=1)(outputs)
                confidences = torch.max(output_logits, dim=1).values.cpu().numpy().tolist()
                predictions = torch.argmax(outputs, dim=1)

            else:
                output_logits = torch.nn.Sigmoid()(outputs)
                predictions = torch.round(output_logits).int()
                confidences = torch.tensor([i if i > 0.5 else 1-i for i in output_logits])


            metric.update(predictions, labels)

            predictions = predictions.cpu().numpy().tolist()

            for filename, pred, conf in zip(filenames, predictions, confidences):
                try:
                    asset = test_dataset.find_asset(filename=filename)
                except ResourceNotFoundError:
                    logging.error(f'Asset {asset.filename} not found in dataset version')
                    continue

                label = test_dataset.get_or_create_label(name=label_map[pred])
                experiment.add_evaluation(asset, classifications=[(label, round(conf, 3))])



    job = experiment.compute_evaluations_metrics(InferenceType.CLASSIFICATION)
    job.wait_for_done()

    confusion_matrix = metric.compute().cpu().numpy().astype(int)
    data_confusion = confusion = {
        'categories': label_map,
        'values': confusion_matrix.tolist()
    }
    experiment.log(name='Test confusion matrix', data=data_confusion, type=LogType.HEATMAP)


def get_filename_and_label(asset: Asset) -> tuple[str, str]:
    """
    Get the filename and label name of a given asset
    :param asset: given asset
    :return: filename and label name of a given asset
    """
    annotation = asset.list_annotations()[0]
    return asset.filename, annotation.list_classifications()[0].label.name


def create_label_file(datasets: list[DatasetVersion], labelmap: dict, dataset_root_folder: str,
                      csv_filename: str = 'labels.csv') -> None:
    if not os.path.isfile(os.path.join(dataset_root_folder, csv_filename)):
        # get inverse of labelmap -> change key-value by value-key
        value_key_labelmap = dict((v, k) for k, v in labelmap.items())

        data = {
            'filename': [],
            'label': []
        }

        for dataset_version in datasets:
            logging.info(f'Write labels on .csv for dataset version {dataset_version.version}')
            result = Parallel(n_jobs=os.cpu_count())(
                delayed(get_filename_and_label)(asset) for asset in tqdm(dataset_version.list_assets()))

            for filename, label in result:
                data['filename'].append(filename)
                data['label'].append(value_key_labelmap[label])

        df = pd.DataFrame(data)
        os.makedirs(dataset_root_folder, exist_ok=True)
        df.to_csv(os.path.join(dataset_root_folder, csv_filename))

    else:
        logging.info('.csv file had already been downloaded')


if __name__ == '__main__':
    # TODO
    """
    get labelmap
    construct .csv file from labelmap
    get context parameters
    create a warmup
    """
    random_seed: typing.Final[int] = 42

    torch.manual_seed(random_seed)

    # Define input/output folders
    dataset_root_folder: str = os.path.join(os.getcwd(), 'dataset')
    path_saved_models: str = os.path.join(os.getcwd(), 'saved_models')
    os.makedirs(path_saved_models, exist_ok=True)

    # Picsell.ia connection
    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]
    client = Client(api_token=api_token, organization_id=organization_id)

    # Get experiment
    experiment = client.get_experiment_by_id(id=os.environ["experiment_id"])

    # get params
    context = experiment.get_log(name='parameters').data
    image_side = context.get('image_size', 512)
    image_size: typing.Final[tuple[int, int]] = (image_side, image_side)
    learning_rate: typing.Final[float] = context.get('learning_rate', 1e-4)
    num_epochs: typing.Final[int] = context.get('num_epochs', 10)
    batch_size: typing.Final[int] = context.get('batch_size', 8)
    num_workers: typing.Final[int] = context.get('num_workers', os.cpu_count())
    pretrained_model: typing.Final[bool] = bool(context.get('pretrained_model', 1))
    warmup_period: typing.Final[int] = context.get('warmup_period', 2)
    warmup_last_step = context.get('warmup_last_step', 5)
    lr_scheduler_step_size = context.get('lr_scheduler_step_size', 10)
    lr_scheduler_gamma = context.get('lr_scheduler_gamma', 0.9)

    if 'nb_layers' not in list(context.keys()):
        logging.error('nb_layers not defined, the program will stop')
        experiment.update(status=ExperimentStatus.FAILED)
        raise SystemExit

    else:
        nb_layers: typing.Final[int] = context['nb_layers']

    logger = PicselliaLogger(client=client, experiment=experiment)

    # Get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f'Using device: {device.type.upper()}')

    # Download datasets
    datasets = experiment.list_attached_dataset_versions()

    # get labelmap
    labelmap = get_class_mapping_from_picsellia(dataset_versions=datasets)

    # Create label .csv file
    create_label_file(datasets=datasets, dataset_root_folder=dataset_root_folder, labelmap=labelmap)

    if not os.path.exists(os.path.join(dataset_root_folder, 'train')):
        # TODO create function to download dataset depending on constraints
        download_datasets(experiment=experiment, root_folder=dataset_root_folder)

    else:
        logging.warning(f'A dataset was previously imported before the training.')

    base_model = experiment.get_base_model_version()

    train_transform = A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(0.5),
            A.VerticalFlip(0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    validation_transform = A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_dataset = ClassificationDataset(csv_path=os.path.join(dataset_root_folder, 'labels.csv'),
                                          image_folder=os.path.join(dataset_root_folder, 'train'),
                                          transform=train_transform)
    val_dataset = ClassificationDataset(csv_path=os.path.join(dataset_root_folder, 'labels.csv'),
                                        image_folder=os.path.join(dataset_root_folder, 'val'),
                                        transform=validation_transform, is_test=True)

    test_dataset = ClassificationDataset(csv_path=os.path.join(dataset_root_folder, 'labels.csv'),
                                         image_folder=os.path.join(dataset_root_folder,
                                                                   'test' if len(datasets) == 3 else 'val'),
                                         transform=validation_transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    if hasattr(torchvision.models, f'resnet{nb_layers}'):
        model = getattr(torchvision.models, f'resnet{nb_layers}')(pretrained=pretrained_model)

    else:
        logging.error(f'The model ResNet with {nb_layers} was not found. The training process will close.')
        raise f'The model ResNet with {nb_layers} was not found. The training process will close.'

    # Modify the last layer of the model
    num_classes = 1 if len(labelmap) == 2 else len(labelmap)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Set the model to run on the device
    model.to(device)

    if len(labelmap) > 2:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    warmup_scheduler = ExponentialWarmup(optimizer,
                                         warmup_period=warmup_period,
                                         last_step=warmup_last_step)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=lr_scheduler_step_size,
                                                   gamma=lr_scheduler_gamma)

    logger.on_train_begin(class_mapping=labelmap)

    if len(labelmap) > 2:
        val_accuracy = MulticlassAccuracy(device=torch.device(device))
        val_precision = MulticlassPrecision(device=torch.device(device))
        val_recall = MulticlassRecall(device=torch.device(device))

    else:
        val_accuracy = BinaryAccuracy(device=torch.device(device))
        val_precision = BinaryPrecision(device=torch.device(device))
        val_recall = BinaryRecall(device=torch.device(device))

    for epoch in range(num_epochs):
        train_loss = 0.0
        validation_loss = 0.0

        with tqdm(train_loader, unit="batch") as t_epoch:
            for inputs, labels in t_epoch:
                t_epoch.set_description(f"Epoch {epoch}")

                # Move input and label tensors to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero out the optimizer
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                if len(labelmap) > 2:
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(torch.squeeze(outputs), labels.float())

                train_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()

        model.eval()
        total_correct = 0
        total_instances = 0

        with torch.no_grad():
            for inputs, labels, _ in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                if len(labelmap) > 2:
                    loss = criterion(outputs, labels)
                    classifications = torch.argmax(outputs, dim=1)
                else:
                    outputs = torch.squeeze(outputs)
                    loss = criterion(torch.squeeze(outputs), labels.float())
                    classifications = torch.round(outputs).int()

                validation_loss += loss.item()

                val_accuracy.update(classifications, labels)
                val_precision.update(classifications, labels)
                val_recall.update(classifications, labels)

        logger.on_epoch_end(train_loss=train_loss, val_loss=validation_loss,
                            val_accuracy=float(val_accuracy.compute()),
                            val_precision=float(val_precision.compute()),
                            val_recall=float(val_recall.compute()),
                            display_gpu_occupancy=True if torch.cuda.is_available() else False, epoch=epoch,
                            current_lr=round(optimizer.param_groups[0]['lr'], 6))

        # Reset metrics for the next epoch
        val_recall.reset()
        val_accuracy.reset()
        val_precision.reset()

        # update the learning rate
        with warmup_scheduler.dampening():
            lr_scheduler.step()

    logging.info("Saving model...")
    last_model_path = os.path.join(path_saved_models, "last_weights.pth")
    torch.save(model.state_dict(), last_model_path)
    logger.store_model(model_path=last_model_path, model_name="last_weights")
    logging.info("Model weights were successfully saved.")

    evaluation_dataset = experiment.get_dataset('val') if len(datasets) == 2 else experiment.get_dataset('test')
    fill_picsellia_evaluation_tab(model=model, validation_loader=test_loader,
                                  test_dataset=evaluation_dataset,
                                  device=device, experiment=experiment, labelmap=labelmap)

    logging.info("Training was successfully completed.")
    experiment.update(status=ExperimentStatus.SUCCESS)
