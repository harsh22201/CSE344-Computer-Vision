import torch
import numpy as np
torch.manual_seed(2022201)
np.random.seed(2022201)
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

# dataset directory
dataset_dir = "../data/russian-wildlife-dataset/Cropped_final"

# Define class labels mapping
class_mapping = {
    'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3,
    'brown_bear': 4, 'dog': 5, 'roe_deer': 6, 'sika_deer': 7,
    'wild_boar': 8, 'people': 9
}

# Define Number of classes
num_classes = len(class_mapping)

# Transformations Used by Resnet
resnet18_transform = transforms.Compose([
    transforms.Resize(256),  # Resize shorter side to 256 while maintaining aspect ratio
    transforms.CenterCrop(224),  # Crop to 224x224 at the center
    transforms.ToTensor(),  # Convert image to tensor (C, H, W) in range [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Custom Dataset class
class RussianWildlifeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, preload=True):
        self.image_paths = image_paths
        self.transform = transform
        self.preload = preload
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Preload data into RAM if preload=True
        if self.preload:
            self.preloaded_data = [self.process_image(img_path) for img_path in self.image_paths]

    def process_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.preloaded_data[idx] if self.preload else self.process_image(self.image_paths[idx])
        label = self.labels[idx]
        return image, label