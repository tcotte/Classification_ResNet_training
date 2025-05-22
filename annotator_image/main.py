import logging
import os.path
from distutils.util import strtobool

import torch.cuda
from picsellia import Client

from annotator import PreAnnotator

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def str2bool(str_value: str) -> bool:
    return str_value.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    use_picsellia_processing: bool = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info(f"Using device: {device}")

    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]
    job_id = os.environ["job_id"]

    client = Client(api_token=api_token, organization_id=organization_id)

    job = client.get_job_by_id(job_id)

    context = job.sync()["dataset_version_processing_job"]
    input_dataset_version_id = context["input_dataset_version_id"]
    output_dataset_version = context["output_dataset_version_id"]
    parameters = context["parameters"]
    model_version_id = context["model_version_id"]

    parameters = context["parameters"]
    confidence_threshold = parameters.get("confidence_threshold", 0.25)
    image_size = parameters.get("image_size", 1024)

    if 'single_class' in parameters:
        single_class = strtobool(val=parameters['single_class'])


    logging.info(f"Used parameters:")
    for k, v in parameters.items():
        logging.info(f"{k}: {v}")

    pre_annotator = PreAnnotator(client=client,
                                 dataset_version_id=input_dataset_version_id,
                                 model_version_id=model_version_id,
                                 parameters=parameters,
                                 img_size=image_size)

    pre_annotator.setup_pre_annotation_job()
    pre_annotator.pre_annotate(confidence_threshold)

    logging.info("Pre-annotation done!")