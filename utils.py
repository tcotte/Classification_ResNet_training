import logging
import os
import typing

import pandas as pd
import torch
from matplotlib import pyplot as plt
from picsellia import Experiment, DatasetVersion


def get_class_mapping_from_picsellia(dataset_versions: typing.List[DatasetVersion]) -> typing.Dict[int, str]:
    labels = []
    for ds_version in dataset_versions:
        for label in ds_version.list_labels():
            if label.name not in labels:
                labels.append(label.name)

    return dict(zip(range(len(labels)), labels))


def download_datasets(experiment: Experiment, root_folder: str = 'dataset'):
    """
    .
    ├── train/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── val/
        ├── img1.jpg
        ├── img2.jpg
        └── ...
    """
    def download_dataset_version():
        images_folder_path = os.path.join(root, alias)

        os.makedirs(images_folder_path)

        assets = dataset_version.list_assets()

        # create csv annotation

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

def get_label_distribution(csv_filepath: str, y_column_name: str = 'y') -> dict:
    df = pd.read_csv(csv_filepath)
    return df[y_column_name].value_counts().to_dict()


def plot_label_distribution(csv_filepath: str, y_column_name: str):
    label_distribution = get_label_distribution(csv_filepath, y_column_name)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.35)
    cls_names = list(label_distribution.keys())
    counts = list(label_distribution.values())

    ax.bar(cls_names, counts)

    ax.set_ylabel('Samples number')
    ax.set_title('Dataset distribution')

    plt.xticks(rotation=45, ha='right')

    plt.show()


if __name__ == '__main__':
    # plot_label_distribution('data.csv', y_column_name='matrix')

    labels = get_label_distribution('utils_ct/data.csv', y_column_name='matrix')

    dict_part = {}
    test_size = 0.15
    for label, count in labels.items():
        dict_part[label] = round(count * test_size)

    print(dict_part)

    df = pd.read_csv('utils_ct/data.csv')

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    for label, count in labels.items():
        nb_test_elements = round(count * test_size)

        train_X.extend(df[df.matrix == label][nb_test_elements:].filename)
        train_X.extend(df[df.matrix == label][nb_test_elements:].matrix)
        test_X.extend(df[df.matrix == label][:nb_test_elements].filename)
        test_Y.extend(df[df.matrix == label][:nb_test_elements].matrix)




