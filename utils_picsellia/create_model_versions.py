import os

from picsellia.exceptions import ResourceConflictError
from picsellia import Client
from picsellia.types.enums import Framework, InferenceType

if __name__ == '__main__':

    client = Client(api_token=os.environ['api_token'], organization_id=os.environ['organization_id'])
    model = client.get_model(name='ResNet')

    base_parameters = {
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "num_workers": 8,
        "pretrained_model": 1,
        "nb_layers": 18
    }

    for nb_layers in [18, 50]:
        base_parameters['nb_layers'] = nb_layers
        try:
            model.create_version(base_parameters=base_parameters,
                                 docker_env_variables={'architecture': nb_layers},
                                 name='ResNet' + str(nb_layers),
                                 type=InferenceType.CLASSIFICATION,
                                 framework=Framework.PYTORCH)

        except ResourceConflictError:
            model.get_version(version='ResNet' + str(nb_layers))
            model.update(base_parameters=base_parameters,
                         docker_env_variables={'architecture': nb_layers},
                         name='ResNet' + str(nb_layers),
                         type=InferenceType.CLASSIFICATION,
                         framework=Framework.PYTORCH)


