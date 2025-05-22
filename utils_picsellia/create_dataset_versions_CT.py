import os
from enum import Enum

from picsellia import Client
from picsellia.types.enums import InferenceType
from tqdm import tqdm


class Matrices(Enum):
    """
    Enumeration of different matrix types
    """
    EscherichiaColi = 'Escherichia coli'
    CandidaAlbicans = 'Candida albicans'
    AspergillusBrasiliensis = 'Aspergillus brasiliensis'
    PseudomonasAeruginosa = 'Pseudomonas aeruginosa'
    StaphylococcusAureus = 'Staphylococcus aureus'
    BurkholderiaCepacia = 'Burkholderia cepacia'
    PluralibacterGergoviae = 'Pluralibacter gergoviae'
    # EffetMatrice = 'Effet Matrice'




if __name__ == '__main__':
    client = Client(os.environ['api_token'], organization_id=os.environ['organization_id'])
    datalake = client.get_datalake()

    dataset_id = '0195f1d2-368c-7629-a601-a7dabffc1408'
    dataset_classification = client.get_dataset_by_id(dataset_id)

    # get dataset versions ids which are important -> first is challenge_test and second one is challenge_test_v2
    dataset_version_ids = {
        # 'challenge_test': '01954714-a30b-7c87-a774-a9e9d3adb80d',
        # 'challenge_test_v2': '0195d697-46a8-73ed-9cf1-62102f1d0996',
        'challenge_test_v3': '0196b40e-bb9d-7047-aaa7-fa2f39c64db9'

    }

    for set_name, dataset_version_id in dataset_version_ids.items():
        classification_dataset_version = dataset_classification.create_version(version=set_name,
                                                                               type=InferenceType.CLASSIFICATION)

        for label_name in [m.value for m in Matrices]:
            classification_dataset_version.create_label(label_name)

        previous_dataset_version = client.get_dataset_version_by_id(dataset_version_id)
        assets = previous_dataset_version.list_assets()

        classification_dataset_version.add_data([asset.get_data() for asset in assets])

        for asset in tqdm(classification_dataset_version.list_assets()):
            tags = [tag.name for tag in asset.get_data_tags()]
            try:
                cls = list(set(tags).intersection([m.value for m in Matrices]))[0]
                annotation = asset.create_annotation()
                annotation.create_classification(label=classification_dataset_version.get_label(name=cls))

            except IndexError:
                raise f'{asset.filename} has no tags associated with challenge test matrices'
