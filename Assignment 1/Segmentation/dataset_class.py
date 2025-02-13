import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CamVidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_dict_path, transform=None, preload=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.preload = preload
        self.class_dict = pd.read_csv(class_dict_path)
        self.images = os.listdir(image_dir)
        
        # Load class mapping
        self.color_to_label = {
            tuple(self.class_dict.iloc[i, 1:4].astype(int)): i 
            for i in range(len(self.class_dict))
        }

        self.mask_transform = transforms.Compose([
            transforms.Resize((360, 480), interpolation=Image.NEAREST)
        ])

        # Preload data into RAM if preload=True
        if self.preload:
            self.preloaded_data = []
            for img_name in self.images:
                self.preloaded_data.append(self.process_image_mask(img_name))

    def __len__(self):
        return len(self.images)

    def encode_mask(self, mask):
        mask = np.array(mask, dtype=np.uint8)
        label_mask = np.zeros(mask.shape[:2], dtype=np.int64)
        for color, label in self.color_to_label.items():
            label_mask[(mask == color).all(axis=-1)] = label
        return torch.tensor(label_mask, dtype=torch.long)
    
    def process_image_mask(self, img_name): 
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.png', '_L.png'))
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        mask = self.mask_transform(mask)
        mask = self.encode_mask(mask)
        
        return image, mask

    def __getitem__(self, idx):
        return self.preloaded_data[idx] if self.preload else self.process_image_mask(self.images[idx])

    def visualize_sample(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.png', '_L.png'))
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        axes[0].imshow(image)  
        axes[0].set_title("Image")
        axes[0].axis("off")
        
        axes[1].imshow(mask) 
        axes[1].set_title("Mask")
        axes[1].axis("off")
        
        plt.show()

# Transform for the CamVid dataset
transform = transforms.Compose([
    transforms.Resize((360, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])