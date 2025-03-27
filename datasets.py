import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CT_Dataset(Dataset):
    def __init__(self, csv_path: str, image_folder: str, transform=None):
        df = pd.read_csv(csv_path)
        self._image_folder = image_folder
        self._labels = df.y.values()
        self.image_filenames = df.X.values()
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        image = Image.open(os.path.join(self._image_folder, self.image_filenames[index])).convert('RGB')

        label = self._labels[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

    def __len__(self):
        return len(self._labels)


class BinaryDataset(Dataset):
    def __init__(self, csv_path: str, image_folder: str, is_test: bool = False, transform=None):
        self._is_test = is_test
        self.df = pd.read_csv(csv_path)
        self._image_folder = image_folder
        self.image_filenames = os.listdir(self._image_folder)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        filename = self.image_filenames[index]
        image = Image.open(os.path.join(self._image_folder, filename)).convert('RGB')

        label = int(self.df[self.df['filename'] == filename].label)
        # label = self._labels[index]

        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"] / 255.

        if not self._is_test:
            return image, label

        else:
            return image, label, filename

    def __len__(self):
        return len(os.listdir(self._image_folder))


if __name__ == '__main__':
    train_dataset = BinaryDataset(csv_path='./dataset/labels.csv', image_folder='dataset/train')
    print(train_dataset[0])
