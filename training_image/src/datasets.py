import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, csv_path: str, image_folder: str, is_test: bool = False, transform = None):
        """
        Classification dataset
        :param csv_path: path of the .csv file which contains labels.
        :param image_folder: dataset root image path
        :param is_test: boolean indicating if the dataset is test or not. If it is a test one, *getitem* method will
        return the filename of the picture.
        :param transform: transformations to apply to each items.
        """
        self._is_test = is_test
        self.df = pd.read_csv(csv_path)
        self._image_folder = image_folder
        self.image_filenames = os.listdir(self._image_folder)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        filename = self.image_filenames[index]
        image = Image.open(os.path.join(self._image_folder, filename)).convert('RGB')

        label = int(self.df[self.df['filename'] == filename].label)

        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"] / 255.

        if not self._is_test:
            return image, label

        else:
            return image, label, filename

    def __len__(self):
        return len(os.listdir(self._image_folder))
