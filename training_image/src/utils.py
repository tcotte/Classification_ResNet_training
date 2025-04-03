import logging
import os
import typing
import torch
from picsellia import Experiment, DatasetVersion


def get_class_mapping_from_picsellia(dataset_versions: typing.List[DatasetVersion]) -> typing.Dict[int, str]:
    """
    Get different labels names of different dataset versions sent into Picsell.ia's experiment.
    :param dataset_versions: dataset versions sent into Picsell.ia's experiment.
    :return: dictionary with number of label name as key and label name as value. For example:
                {
                    0: 'cat',
                    1: 'dog'
                }
    """
    labels = []
    for ds_version in dataset_versions:
        for label in ds_version.list_labels():
            if label.name not in labels:
                labels.append(label.name)

    return dict(zip(range(len(labels)), labels))


def download_datasets(experiment: Experiment, root_folder: str = 'dataset'):
    """
    Download dataset versions from Picsell.ia experiment.
    The folder names of each dataset version will be their alias.
    Number of dataset versions can be:
    - 3 if there are tagged as 'train', 'val', 'test'.
    - 2 if there are tagged as 'train', 'val'.
    If there are only two dataset version: file structure will be like this at the end of download:
    root
    ├── train/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── val/
        ├── img1.jpg
        ├── img2.jpg
        └── ...

    :param experiment: Picsell.ia experiment
    :param root_folder: path of the root folder of the dataset.
    """

    def download_dataset_version() -> None:
        """
        Download image from dataset version in specified folder.
        """
        # create specific directory
        images_folder_path = os.path.join(root, alias)
        os.makedirs(images_folder_path)
        # download dataset version's data in this specific directory
        assets = dataset_version.list_assets()
        assets.download(images_folder_path, max_workers=8)

    root = root_folder
    if len(experiment.list_attached_dataset_versions()) == 3:
        for alias in ['test', 'train', 'val']:
            dataset_version = experiment.get_dataset(alias)
            logging.info(f'{alias} alias for {dataset_version}')
            download_dataset_version()

    elif len(experiment.list_attached_dataset_versions()) == 2:
        for alias in ['train', 'val']:
            dataset_version = experiment.get_dataset(alias)
            logging.info(f'{alias} alias for {dataset_version}')
            download_dataset_version()


def get_GPU_occupancy(gpu_id: int = 0) -> float:
    """
    Get memory occupancy of the used GPU for training.
    :param gpu_id: id of the GPU used for training model.
    :return: Memory occupancy in percentage.
    """
    if torch.cuda.is_available():
        free_memory, total_memory = torch.cuda.mem_get_info(device=gpu_id)
        return 1 - free_memory / total_memory

    else:
        return 0.0


