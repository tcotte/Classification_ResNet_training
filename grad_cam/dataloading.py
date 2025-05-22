import os

import albumentations as A
import numpy as np
from PIL import Image
from albumentations import ToTensorV2
from torch.utils.data import Dataset


def get_test_transform(image_size: tuple[int, int]) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

class ImageDataset(Dataset):
    def __init__(self, image_folder: str, transform = None):
        """
        Classification dataset
        :param csv_path: path of the .csv file which contains labels.
        :param image_folder: dataset root image path
        return the filename of the picture.
        :param transform: transformations to apply to each items.
        """
        self._image_folder = image_folder
        self.image_filenames = os.listdir(self._image_folder)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        filename = self.image_filenames[index]
        image = Image.open(os.path.join(self._image_folder, filename)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"] / 255.

        return image, filename

    def __len__(self):
        return len(os.listdir(self._image_folder))
