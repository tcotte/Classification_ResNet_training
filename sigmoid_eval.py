import os
import albumentations as A
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from datasets import ClassificationDataset

image_size = (512, 512)

num_classes = 2
batch_size = 2

if __name__ == '__main__':
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('saved_models/last_weights.pth', weights_only=True))

    validation_transform = A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dataset_root_folder = os.path.join(os.getcwd(), 'dataset')
    val_dataset = ClassificationDataset(csv_path=os.path.join(dataset_root_folder, 'labels.csv'),
                                        image_folder=os.path.join(dataset_root_folder, 'test'),
                                        transform=validation_transform, is_test=True)

    model.eval()

    validation_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    images, labels, filenames = next(iter(validation_loader))

    with torch.no_grad():
        outputs = model(images)
        print(outputs.size())