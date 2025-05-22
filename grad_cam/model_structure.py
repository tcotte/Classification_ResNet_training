import os

import cv2
import numpy as np
import torch
import torchvision
from picsellia import Client
from torch import nn
from torch.nn import AdaptiveAvgPool2d
from torch.utils import data
import matplotlib.pyplot as plt

from grad_cam.dataloading import ImageDataset, get_test_transform
from grad_cam.utils import download_model_weights, get_label_map_from_experiment, download_test_dataset_version


class GradCamResNet(nn.Module):
    def __init__(self, num_classes: int, nb_layers: int, trained_weights: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        All ResNet models have the same configurations with 4 residual blocks described as *layer x* in torchvision 
        models.
        """
        self.resnet = getattr(torchvision.models, f'resnet{nb_layers}')(
            num_classes=1 if num_classes == 2 else num_classes)
        self.resnet.load_state_dict(torch.load(trained_weights, weights_only=False))
        self.features_conv = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )

        self.avg_pool = AdaptiveAvgPool2d(output_size=(1, 1))

        # placeholder for the gradients
        self.gradients = None

        self.classifier = self.resnet.fc

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        """
        hook function can monitor the output of the layer during the forward or backward pass of the network
        -> https://medium.com/@deepeshdeepakdd2/cnn-visualization-techniques-feature-maps-gradient-ascent-aec4f4aaf5bd
        :param grad:
        :return:
        """
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.avg_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        """
        The gradients (used to compute α) are obtained during the backward pass after the first forward pass.
        Relu activation function used to eliminate gradients negative values -> https://www.google.com/search?client=firefox-b-d&sca_esv=3035b77ba2076880&sxsrf=AHTn8zoAfASOpZZMaj908tTedV62R_O8Bg:1747899824561&q=Grad-CAM+explained&sa=X&ved=2ahUKEwiYm5WhyraNAxWKE1kFHUrhL6YQ1QJ6BAhSEAE&biw=1869&bih=927&dpr=1#fpstate=ive&vld=cid:90071438,vid:_QiebC9WxOc,st:0
        :return:
        """
        return torch.nn.functional.relu(self.gradients)

    # method for the activation extraction
    def get_activations(self, x):
        return self.features_conv(x)

    def compute_predicted_class_and_confidence(self):
        if outputs.size()[1] > 1:
            output_logits = torch.nn.Softmax(dim=1)(outputs)
            confidences = torch.max(output_logits, dim=1).values.detach().cpu().numpy().tolist()
            predictions = torch.argmax(outputs, dim=1)

        else:
            output_logits = torch.nn.Sigmoid()(outputs)
            predictions = torch.round(output_logits).int()
            confidences = torch.tensor([i if i > 0.5 else 1 - i for i in output_logits])

        return int(predictions[0]), float(confidences[0])


if __name__ == '__main__':
    image_size: tuple[int, int] = (640, 640)

    client = Client(api_token=os.environ['api_token'], organization_id=os.environ['organization_id'])
    experiment = client.get_experiment_by_id(os.environ['experiment_id'])

    if not os.path.exists('saved_models'):
        download_model_weights(experiment=experiment, model_filename='last_weights', output_model_path='.')

    if not os.path.exists('dataset'):
        download_test_dataset_version(experiment=experiment, dataset_alias='test', output_directory='./dataset')

    label_map = get_label_map_from_experiment(experiment=experiment)

    grad_cam_resnet = GradCamResNet(num_classes=len(label_map),
                                    nb_layers=experiment.get_log('parameters').data['nb_layers'],
                                    trained_weights='saved_models/last_weights.pth')
    grad_cam_resnet.to('cuda')
    grad_cam_resnet.eval()

    # define a 1 image dataset
    dataset = ImageDataset(image_folder='dataset', transform=get_test_transform(image_size=image_size))

    # define the dataloader to load that single image
    dataloader = data.DataLoader(dataset=dataset, shuffle=True, batch_size=1)

    for i in range(20):
        # get the image from the dataloader
        img, filename = next(iter(dataloader))
        img = img.to('cuda')

        # get the most likely prediction of the model -> forward pass
        outputs = grad_cam_resnet(img)

        # compute confidence and predicted class
        predicted_class, confidence = grad_cam_resnet.compute_predicted_class_and_confidence()

        # backward pass
        if len(label_map) > 2:
            outputs[:, predicted_class].backward()
        else:
            outputs[:, 0].backward()

        # get gradients of the backward pass -> α
        gradients = grad_cam_resnet.get_activations_gradient()
        # Calculate the weighted activation map and then compute mean.
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Activation map is a 3D tensor of shape HxWxK
        # where H and W are the spatial dimensions of the feature map and K is the number of channels in the feature map
        activations = grad_cam_resnet.get_activations(img).detach()

        # multiply and average the convolutional filters and gradients to see which zones have the most importance
        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()

        heatmap = np.maximum(heatmap.cpu(), 0)

        heatmap /= torch.max(heatmap)

        heatmap = heatmap.numpy()
        img = cv2.imread(os.path.join('dataset', filename[0]))
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        alpha = 0.8
        superposed_img = cv2.addWeighted(img, 0.7, heatmap, 1 - alpha, 0, dtype=cv2.CV_32F).astype(int)

        plt.suptitle(f'Prediction: {list(label_map.values())[predicted_class]}')
        plt.title(f'With confidence: {confidence:.2f}', fontsize=10)
        plt.imshow(superposed_img[:, :, ::-1], cmap='gray')
        plt.show()
