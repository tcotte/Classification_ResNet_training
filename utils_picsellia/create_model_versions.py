"""
The aim of this script is to deploy several ResNet model versions to Picsell.ia: one for each existing architecture.
Each architecture (number of layers) is described in Docker environment variables.
"""
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
        # "nb_layers": 18,
        'warmup_period': 2,
        'warmup_last_step': 5,
        'lr_scheduler_step_size': 10,
        'lr_scheduler_gamma': 0.9,
        'image_size': 640,
        'optimizer': 'Adam',
        'weight_decay': 0.0
    }

    for nb_layers in [18, 34, 50, 101, 152]:
        base_parameters['nb_layers'] = nb_layers
        try:
            model.create_version(base_parameters=base_parameters,
                                 docker_env_variables={'architecture': nb_layers},
                                 name='ResNet' + str(nb_layers),
                                 type=InferenceType.CLASSIFICATION,
                                 framework=Framework.PYTORCH,
                                 docker_image_name='9d8xtfjr.c1.gra9.container-registry.ovh.net/picsellia/resnet_trainer',
                                 docker_tag='1.0',
                                 docker_flags=['--gpus all'])

        # if model version already exists: delete it and recreate it
        except ResourceConflictError:
            model_version = model.get_version(version='ResNet' + str(nb_layers))
            model_version.delete()

            model.create_version(base_parameters=base_parameters,
                                 docker_env_variables={'architecture': nb_layers},
                                 name='ResNet' + str(nb_layers),
                                 type=InferenceType.CLASSIFICATION,
                                 framework=Framework.PYTORCH,
                                 docker_image_name='9d8xtfjr.c1.gra9.container-registry.ovh.net/picsellia/resnet_trainer',
                                 docker_tag='1.0',
                                 docker_flags=['--gpus all'])


