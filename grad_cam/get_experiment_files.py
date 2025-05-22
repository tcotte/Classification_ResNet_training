import os

from picsellia import Client, Experiment

from utils import download_model_weights, download_test_dataset_version



if __name__ == '__main__':
    client = Client(api_token=os.environ['api_token'], organization_id=os.environ['organization_id'])
    experiment = client.get_experiment_by_id(os.environ['experiment_id'])

    download_model_weights(experiment=experiment, model_filename='last_weights', output_model_path='.')

    download_test_dataset_version(experiment=experiment, dataset_alias='test', output_directory='./dataset')