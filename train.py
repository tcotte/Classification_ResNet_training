import logging
import os
import typing

import albumentations as A
import torch
import torchvision
from albumentations import ToTensorV2
from picsellia import Client
from picsellia.types.enums import InferenceType
from torch.utils.data import DataLoader

from datasets import BinaryDataset
from logger import PicselliaLogger
from utils import download_datasets

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def fill_picsellia_evaluation_tab(model: torch.nn.Module, validation_loader: DataLoader, test_dataset) -> None:
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

            print(outputs)

            # outputs = model.generate(batch["pixel_values"].to(device))
        # generated_text = processor.batch_decode(outputs, skip_special_tokens=True)

        for filename, pred, conf in zip(filenames.numpy().tolist(), predictions, confidences):
            asset = test_dataset.find_asset(filename=filename)
            #
            label = test_dataset.get_or_create_label(name=label_map[pred])
            experiment.add_evaluation(asset, classifications=[(label, round(conf, 3))])
    #
    job = experiment.compute_evaluations_metrics(InferenceType.CLASSIFICATION)
    job.wait_for_done()


if __name__ == '__main__':
    image_size: typing.Final[tuple[int, int]] = (512, 512)
    learning_rate: typing.Final[float] = 1e-4
    num_epochs: typing.Final[int] = 10
    batch_size: typing.Final[int] = 4
    num_workers: typing.Final[int] = 8
    random_seed: typing.Final[int] = 42
    pretrained_model: typing.Final[bool] = True
    nb_layers: typing.Final[int] = 18

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

    logger = PicselliaLogger(client=client, experiment=experiment)

    # Get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f'Using device: {device.type.upper()}')

    # Download datasets
    datasets = experiment.list_attached_dataset_versions()

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

    """dataset = CT_Dataset(csv_path='data.csv', image_folder='data', transform=train_transform)
    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)"""

    train_dataset = BinaryDataset(csv_path='./dataset/labels.csv', image_folder='dataset/train',
                                  transform=train_transform)
    test_dataset = BinaryDataset(csv_path='./dataset/labels.csv', image_folder='dataset/test',
                                 transform=validation_transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

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

    logger.on_train_begin()

    for epoch in range(num_epochs):
        train_loss = 0.0
        validation_loss = 0.0
        validation_accuracy = 0.0

        for inputs, labels in train_loader:
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        # logging.info(train_loss / len(train_loader))

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

    fill_picsellia_evaluation_tab(model=model, validation_loader=validation_loader,
                                  test_dataset=experiment.get_dataset('val'))
