import os
from PIL import Image
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset


class PetNoseLoader(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.images_dir = image_dir
        self.transform = transform
        self.labels = []

        with open(labels_file, 'r') as file:
            for line in file.readlines():
                line = line.strip()
                img, label_x, label_y = line.split(",")
                label_x = label_x.strip('("')
                label_y = label_y.strip(')"')
                label_y = label_y.strip(' ')
                label_y = float(label_y)
                label_x = float(label_x)
                self.labels.append((img, (label_x, label_y)))
            
    def __len__(self):
        return len(self.labels)
    
    def get_padding(self, image, new_size):
        w, h = image.size
        scale_w = new_size[0] / w
        scale_h = new_size[1] / h
        scale = min(scale_w, scale_h)

        return scale

    def __getitem__(self, index):
        img_name, (label_x, label_y) = self.labels[index]
        img_pth = os.path.join(self.images_dir, img_name)
        image = Image.open(img_pth).convert('RGB')
        
        scale = self.get_padding(image, (224, 224))
        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)), Image.ANTIALIAS)

        image_np = np.array(image)
        old_size = image_np.shape[:2]
        new_size = max(old_size)
        delta_w = new_size - old_size[1]
        delta_h = new_size - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        image_np = cv2.copyMakeBorder(image_np, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        label_x = (label_x * scale) + left
        label_y = (label_y * scale) + top
        label = torch.tensor([label_x, label_y], dtype=torch.float)
        
        image = Image.fromarray(image_np)
                     
        if self.transform:
            image = self.transform(image)

        return image, label