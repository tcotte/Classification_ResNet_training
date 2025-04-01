import logging
import os
import uuid
from collections import Counter
from typing import Dict

import picsellia
from picsellia import Client, Experiment, DatasetVersion, Annotation
from picsellia.types.enums import LogType, ExperimentStatus

from utils import get_GPU_occupancy
from joblib import Parallel, delayed


class PicselliaLogger:
    def __init__(self, client: Client, experiment: Experiment):
        """
        Logger class which enables to log metrics and store files on Picsellia.
        :param client: picsellia client
        :param experiment: experiment in which the metrics will be logged and files will be stored
        """
        self._client: Client = client
        self._experiment: Experiment = experiment

    def log_labelmap(self, class_mapping: dict[int, str]):
        self._experiment.log(name='LabelMap', type=LogType.TABLE,
                             data={str(key): value for key, value in class_mapping.items()})

    def get_label_map(self, list_dataset_versions: list[DatasetVersion]) -> list:
        label_map: list = []
        for dataset_version in list_dataset_versions:
            label_map.extend([label.name for label in dataset_version.list_labels()])

        # remove dupli
        return list(set(label_map))

    def on_end_training(self):
        logging.info("Training was successfully completed.")
        self._experiment.update(status=ExperimentStatus.SUCCESS)

    def get_picsellia_experiment_link(self) -> str:
        """
        Get Picsellia experiment link
        :return: experiment link
        """
        client_id = self._client.id
        project_id = self.get_project_id_from_experiment()
        experiment_id = self._experiment.id

        link = f'https://app.picsellia.com/{str(client_id)}/project/{str(project_id)}/experiment/{experiment_id}'
        return link

    def get_project_id_from_experiment(self) -> uuid.UUID:
        """
        Retrieve project id from experiment id
        :return: project id
        """
        for project in self._client.list_projects():
            for experiment in project.list_experiments():
                if str(experiment.id) == os.environ["experiment_id"]:
                    return project.id

    def log_split_table(self, annotations_in_split: Dict, title: str) -> None:
        """
        Log table
        :param annotations_in_split:
        :param title:
        :return:
        """
        data = {'x': [], 'y': []}
        for key, value in annotations_in_split.items():
            data['x'].append(key)
            data['y'].append(value)

        self._experiment.log(name=title, type=LogType.BAR, data=data)

    def on_train_begin(self, class_mapping: dict[int, str]) -> None:
        """
        Do some actions when training begins:
        - Write experiment link in telemetry.
        - Plot labels of training/validation dataset version.
        :return:
        """
        logging.info(f"Successfully logged to Picsellia\n You can follow your experiment here: "
                     f"{self.get_picsellia_experiment_link()} ")

        self.log_labelmap(class_mapping=class_mapping)

        self._experiment.update(status=ExperimentStatus.RUNNING)

        self.plot_dataset_version_labels(dataset_version_names=['train', 'val'])

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, val_accuracy: float, val_recall: float,
                     val_precision: float, display_gpu_occupancy: bool, current_lr: float) -> None:
        """
        Log training loss and validation accuracy when the epoch finishes
        :param epoch: epoch number which finished
        :param train_loss: previous training loss
        :param val_accuracy: previous validation accuracy
        :param display_gpu_occupancy: boolean which indicates if the GPU occupancy is displayed on Picsellia
        """
        train_loss = round(train_loss, 2)
        val_loss = round(val_loss, 2)
        val_accuracy = round(val_accuracy, 2)
        val_precision = round(val_precision, 2)
        val_recall = round(val_recall, 2)

        self._experiment.log(name='Training loss', type=LogType.LINE, data=train_loss)
        self._experiment.log(name='Validation loss', type=LogType.LINE, data=val_loss)

        logging.info(f"Epoch {epoch + 1}: Training loss {train_loss} / Validation loss: {val_loss} "
                     f"/ Accuracy {val_accuracy} / Precision {val_precision} / Recall {val_recall}")

        self._experiment.log(name='Accuracy', type=LogType.LINE, data=val_accuracy)
        self._experiment.log(name='Precision', type=LogType.LINE, data=val_precision)
        self._experiment.log(name='Recall', type=LogType.LINE, data=val_recall)

        if display_gpu_occupancy:
            self._experiment.log(name='GPU occupancy (%)', type=LogType.LINE, data=round(get_GPU_occupancy(), 2))

        self._experiment.log(name='Learning rate', type=LogType.LINE, data=current_lr)

    def store_model(self, model_path: str, model_name: str) -> None:
        """
        Store model as zip of files in Picsellia
        :param model_path: path of the folder that will be zipped and store in Picsellia
        :param model_name: name of the model on Picsellia
        """
        self._experiment.store(model_name, model_path, do_zip=True)

    def plot_dataset_version_labels(self, dataset_version_names: list[str]) -> None:
        """
        Plot label distribution of several dataset versions.
        :param dataset_version_names: Names of dataset versions
        """
        for version_name in dataset_version_names:
            self.dataset_label_distribution(dataset_version_name=version_name)


    def dataset_label_distribution(self, dataset_version_name: str) -> None:
        """
        Plot label distribution of dataset version in Picsellia.
        :param dataset_version_name: Alias of dataset version
        """
        def get_classification_label(annotation: Annotation) -> str:
            return annotation.list_classifications()[0].label.name

        try:
            dataset_version: DatasetVersion = self._experiment.get_dataset(name=dataset_version_name)

            list_label_names = Parallel(n_jobs=os.cpu_count())(delayed(get_classification_label)(annotation) for
                                                               annotation in dataset_version.list_annotations())

            distribution_dict = Counter(list_label_names)
            data = {'x': list(distribution_dict.keys()), 'y': list(distribution_dict.values())}
            self._experiment.log(name=f'{dataset_version_name}_labels', type=LogType.BAR, data=data)

        except picsellia.exceptions.ResourceNotFoundError:
            logging.warning(f'Dataset version with name {dataset_version_name} was not found \n')

        except Exception as e:
            logging.warning(str(e))


if __name__ == '__main__':
    print(os.environ['api_token'])
    client = Client(api_token=os.environ['api_token'], organization_id=os.environ["organization_id"])
    experiment_id = '0195ada0-141a-793a-a176-b380b5bf2736'
    experiment = client.get_experiment_by_id(experiment_id)
    logger = PicselliaLogger(client=client, experiment=experiment)
    # ds_versions = experiment.list_attached_dataset_versions()
    # print(ds_versions)

    for ds_alias in ['train', 'val']:
        logger.dataset_label_distribution(dataset_version_name=ds_alias)

    # training_dataset_version = experiment.get_dataset(name='train')

    # training_dataset_version.list_labels()
