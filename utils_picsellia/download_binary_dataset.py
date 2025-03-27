import os

import pandas as pd
from sklearn.model_selection import train_test_split
from picsellia import Client, DatasetVersion
from joblib import parallel, delayed, Parallel


def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name


if __name__ == '__main__':
    client = Client(os.environ['api_token'], organization_id=os.environ['organization_id'])
    datalake = client.get_datalake()

    # get datasets ids which are important
    datasets_label_ids = {
        'not_annotated': ['01954714-a30b-7c87-a774-a9e9d3adb80d', '0195d697-46a8-73ed-9cf1-62102f1d0996'],
        'annotated': ['0195d697-47ef-72be-a1ea-70b6b926cfad', '01954714-a3b9-7a0a-bab1-00fe3387b69e']
    }

    # get assets and labels iterating through dataset_label_ids values
    X = []
    y = []

    for dataset_label, list_ids in datasets_label_ids.items():
        for dataset_version_id in list_ids:
            dataset_version: DatasetVersion = client.get_dataset_version_by_id(id=dataset_version_id)
            for asset in dataset_version.list_assets():
                X.append(asset)
                y.append(0 if dataset_label == 'not_annotated' else 1)

    # Randomly split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Store pictures in different folders depending on their set (training or test)
    root_dataset_dirname: str = 'dataset'
    for set_ in [X_train, X_test, y_train, y_test]:
        set_name = get_var_name(set_)
        if set_name.split('_')[0] == 'X':
            path_set_ = os.path.join(root_dataset_dirname, set_name.split('_')[1])
            # create relative folder
            os.makedirs(path_set_, exist_ok=True)
            _ = Parallel(n_jobs=os.cpu_count(), verbose=10)(delayed(asset.download)(path_set_) for asset in set_)

    # Store labels in .csv
    data_x = [asset.filename for asset in X_train] + [asset.filename for asset in X_test]
    data_y = y_train + y_test
    df = pd.DataFrame({'filename': data_x, 'label': data_y})
    df.to_csv(os.path.join(root_dataset_dirname, 'labels.csv'))


