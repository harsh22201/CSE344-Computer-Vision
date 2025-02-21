import torchvision
from torchvision import models
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes:int = 10):
        super(ConvNet, self).__init__()

        self.num_classes = num_classes  # Can be set dynamically before training

        # First Conv Layer: 3 -> 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Second Conv Layer: 32 -> 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third Conv Layer: 64 -> 128 feature maps
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input Size: 224x224, Flattened size: 14x14x128
        self.flatten = nn.Flatten()

        # Fully connected layers (classification head)
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes: int=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Use resnet18 from torchvision with Pretrained Weights.
        in_features = self.model.fc.in_features  # Get input features of the original resenet18 (image size : 224 x 224) fc layer
        self.model.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)