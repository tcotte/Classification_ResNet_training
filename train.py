import logging
import os
import typing

import albumentations as A
import pandas as pd
import picsellia
import torch
import torchvision
from albumentations import ToTensorV2
from joblib import delayed, Parallel
from picsellia import Client, Experiment, Asset, DatasetVersion
from picsellia.exceptions import ResourceNotFoundError
from picsellia.types.enums import InferenceType, ExperimentStatus
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BinaryDataset
from logger import PicselliaLogger
from utils import download_datasets, get_class_mapping_from_picsellia

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def fill_picsellia_evaluation_tab(model: torch.nn.Module, validation_loader: DataLoader,
                                  test_dataset: picsellia.DatasetVersion, device: str,
                                  experiment: Experiment) -> None:
    """
    Fill Picsellia evaluation which allows comparing on a dedicated bench of images the prediction done by the freshly
    trained model with the ground-truth.
    :param model: Model which will be used to do the predictions
    :param data_loader: Dataloader which gathers the bench of images on which the evaluation will be done
    :param test_dataset_id: ID of the dataset version which comports the pictures on which the evaluation will be done
    """
    label_map = ['raw', 'marked']

    model.eval()

    with torch.no_grad():
        for inputs, labels, filenames in validation_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            output_logits = torch.nn.Softmax(dim=1)(outputs)

            confidences = torch.max(output_logits, dim=1).values.cpu().numpy().tolist()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy().tolist()

            for filename, pred, conf in zip(filenames, predictions, confidences):
                try:
                    asset = test_dataset.find_asset(filename=filename)
                except ResourceNotFoundError:
                    logging.error(f'Asset {asset.filename} not found in dataset version')
                    continue

                label = test_dataset.get_or_create_label(name=label_map[pred])
                experiment.add_evaluation(asset, classifications=[(label, round(conf, 3))])
    #
    job = experiment.compute_evaluations_metrics(InferenceType.CLASSIFICATION)
    job.wait_for_done()


def get_filename_and_label(asset: Asset) -> tuple[str, str]:
    annotation = asset.list_annotations()[0]
    return asset.filename, annotation.list_classifications()[0].label.name


def create_label_file(datasets: list[DatasetVersion], labelmap: dict, csv_filename: str = 'labels.csv') -> None:
    # get inverse of labelmap -> change key-value by value-key
    value_key_labelmap = dict((v, k) for k, v in labelmap.items())

    data = {
        'filename': [],
        'label': []
    }

    for dataset_version in datasets:
        result = Parallel(n_jobs=os.cpu_count())(
            delayed(get_filename_and_label)(asset) for asset in tqdm(dataset_version.list_assets()))

        for filename, label in result:
            data['filename'].append(filename)
            data['label'].append(value_key_labelmap[label])

    df = pd.DataFrame(data)
    os.makedirs(dataset_root_folder, exist_ok=True)
    df.to_csv(os.path.join(dataset_root_folder, csv_filename))


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
    dataset_root_folder: str = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
    path_saved_models: str = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
    os.makedirs(path_saved_models, exist_ok=True)

    # Picsell.ia connection
    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]
    client = Client(api_token=api_token, organization_id=organization_id)

    # Get experiment
    experiment = client.get_experiment_by_id(id=os.environ["experiment_id"])

    # TODO get params
    context = experiment.get_log(name='parameters').data
    image_side = context.get('image_size', 512)
    image_size: typing.Final[tuple[int, int]] = (image_side, image_side)
    learning_rate: typing.Final[float] = context.get('learning_rate', 1e-4)
    num_epochs: typing.Final[int] = context.get('num_epochs', 10)
    batch_size: typing.Final[int] = context.get('batch_size', 8)
    num_workers: typing.Final[int] = context.get('num_workers', os.cpu_count())
    pretrained_model: typing.Final[bool] = bool(context.get('pretrained_model', 1))

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
    create_label_file(datasets=datasets, labelmap=labelmap)


    if not os.path.exists(dataset_root_folder):
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


    train_dataset = BinaryDataset(csv_path=os.path.join(dataset_root_folder, 'labels.csv'),
                                  image_folder=os.path.join(dataset_root_folder, 'train'),
                                  transform=train_transform)
    val_dataset = BinaryDataset(csv_path=os.path.join(dataset_root_folder, 'labels.csv'),
                                image_folder=os.path.join(dataset_root_folder, 'val'),
                                transform=validation_transform, is_test=False)

    test_dataset = BinaryDataset(csv_path=os.path.join(dataset_root_folder, 'labels.csv'),
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
    num_classes = 2
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Set the model to run on the device
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.on_train_begin(class_mapping=labelmap)

    for epoch in range(num_epochs):
        train_loss = 0.0
        validation_loss = 0.0
        validation_accuracy = 0.0

        for inputs, labels in train_loader:
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
                loss = criterion(outputs, labels)

                validation_loss += loss.item()

                classifications = torch.argmax(outputs, dim=1)
                correct_predictions = sum(classifications == labels).item()
                total_correct += correct_predictions
                total_instances += len(inputs)

        validation_accuracy = total_correct / total_instances
        logger.on_epoch_end(train_loss=train_loss, val_loss=validation_loss, val_accuracy=validation_accuracy,
                            display_gpu_occupancy=True if torch.cuda.is_available() else False, epoch=epoch)

    logging.info("Saving model...")
    last_model_path = os.path.join(path_saved_models, "last_weights.pth")
    torch.save(model.state_dict(), last_model_path)
    logger.store_model(model_path=last_model_path, model_name="last_weights")
    logging.info("Model weights were successfully saved.")

    evaluation_dataset = experiment.get_dataset('val') if len(datasets) == 2 else experiment.get_dataset('test')
    fill_picsellia_evaluation_tab(model=model, validation_loader=test_loader,
                                  test_dataset=evaluation_dataset,
                                  device=device, experiment=experiment)

    logging.info("Training was successfully completed.")
    experiment.update(status=ExperimentStatus.SUCCESS)
