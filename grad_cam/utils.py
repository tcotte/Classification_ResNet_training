import logging
import os
import zipfile

from picsellia import Experiment

def get_label_map_from_experiment(experiment: Experiment) -> dict:
    return experiment.get_log('LabelMap').data

def download_model_weights(experiment: Experiment, model_filename: str, output_model_path: str):
    try:
        artifact = experiment.get_artifact(model_filename)
        # download model
        logging.info('Downloading the model...')

        artifact.download(output_model_path)
        logging.info('Model was successfully downloaded !')

        logging.info('Extracting model zip...')
        zip_filename = f'{model_filename}.zip'

        with zipfile.ZipFile(os.path.join(output_model_path, zip_filename), 'r') as zip_ref:
            zip_ref.extractall(output_model_path)
        os.remove(os.path.join(output_model_path, zip_filename))

        logging.info('Extraction was done !')



    except:
        logging.warning(f'Can not retrieve {model_filename} file')


def download_test_dataset_version(experiment: Experiment, dataset_alias:str = 'test', output_directory: str = '.'):
    dataset_version = experiment.get_dataset(name=dataset_alias)
    dataset_version.download(output_directory)

