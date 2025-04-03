import os

from picsellia import Client, DatasetVersion
from sklearn.model_selection import train_test_split


def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name


if __name__ == '__main__':
    client = Client(os.environ['api_token'], organization_id=os.environ['organization_id'])
    datalake = client.get_datalake()

    # get dataset versions ids which are important
    datasets_label_ids = {
        'raw': ['01954714-a30b-7c87-a774-a9e9d3adb80d', '0195d697-46a8-73ed-9cf1-62102f1d0996'],
        'marked': ['0195d697-47ef-72be-a1ea-70b6b926cfad', '01954714-a3b9-7a0a-bab1-00fe3387b69e']
    }

    # get assets and labels iterating through dataset_label_ids values
    X = []
    y = []

    for dataset_label, list_ids in datasets_label_ids.items():
        for dataset_version_id in list_ids:
            dataset_version: DatasetVersion = client.get_dataset_version_by_id(id=dataset_version_id)
            for asset in dataset_version.list_assets():
                X.append(asset)
                y.append(dataset_label)

    # Randomly split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # create new dataset versions
    dataset = client.get_dataset_by_id('0195d714-888b-7857-8bfa-d47dd9d1dcc2')

    train_dataset_version = dataset.create_version('train')
    val_dataset_version = dataset.create_version('val')

    for label_name in list(datasets_label_ids.keys()):
        for ds_version in [train_dataset_version, val_dataset_version]:
            ds_version.create_label(name=label_name)

    # add data to new dataset versions
    train_dataset_version.add_data([datalake.find_data(filename=asset.filename) for asset in X_train])
    val_dataset_version.add_data([datalake.find_data(filename=asset.filename) for asset in X_test])

    # create classification annotation for each asset of training dataset version
    for asset, label in zip(X_train, y_train):
        asset = train_dataset_version.find_asset(filename=asset.filename)
        annotation = asset.create_annotation()
        annotation.create_classification(train_dataset_version.get_label(name=label))

    # create classification annotation for each asset of test dataset version
    for asset, label in zip(X_test, y_test):
        asset = val_dataset_version.find_asset(filename=asset.filename)
        annotation = asset.create_annotation()
        annotation.create_classification(val_dataset_version.get_label(name=label))