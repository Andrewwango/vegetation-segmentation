import os
import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms

def convert_from_color_mode(img, color_mode):
    if color_mode == "rgb":
        a = img.convert("RGB")
    elif color_mode == "grayscale":
        a = img.convert("L")
    else:
        a = img
    return np.array(a)

class FreiburgDataset(VisionDataset):
    def __init__(self, root,
                 seed=None,ext_img='.jpg', ext_mask='.png',
                 image_color_mode="rgb",
                 mask_color_mode=""#"grayscale"
                ):

        super().__init__(root, transforms)
        
        self.imgs  = [p for p in sorted(os.listdir(os.path.join(root, "img")))  if p.endswith(ext_img)]
        self.masks = [p for p in sorted(os.listdir(os.path.join(root, "mask"))) if p.endswith(ext_mask)]

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode
        
        self.obj_ids = np.array([[170, 170, 170], 
                                 [0, 255, 0],
                                 [102, 102, 51],
                                 [0, 60, 0],
                                 [0, 120, 255],
                                 [0, 0, 0]])
        self.obj_names = ["Road", "Grass", "Vegetation", "Tree", "Sky", "Obstacle"]
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.CenterCrop((480, 880))])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.root, "img",  self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])

        img = Image.open(img_path)
        img = convert_from_color_mode(img, self.image_color_mode)
        mask = Image.open(mask_path)
        mask = convert_from_color_mode(mask, self.mask_color_mode)
        
        masks = np.all(mask == self.obj_ids[:, None, None], axis=3).astype(np.uint8) * 255
        masks = masks.transpose(1,2,0)

        #img = img.transpose(1,2,0)
        
        
        if self.transforms:
            img = self.transforms(img)
            masks = self.transforms(masks)
        
        return img, masks

import os
import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms

def convert_from_color_mode(img, color_mode):
    if color_mode == "rgb":
        a = img.convert("RGB")
    elif color_mode == "grayscale":
        a = img.convert("L")
    else:
        a = img
    return np.array(a)

class FreiburgTestDataset(VisionDataset):
    def __init__(self, root,
                 seed=None,ext_img='.jpg',
                 image_color_mode="rgb"
                ):

        super().__init__(root, transforms)
        
        self.imgs  = [p for p in sorted(os.listdir(os.path.join(root, "img")))  if p.endswith(ext_img)]

        self.image_color_mode = image_color_mode
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((480, 880))])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.root, "img",  self.imgs[idx])
        img = Image.open(img_path)
        img = convert_from_color_mode(img, self.image_color_mode)
        if self.transforms:
            img = self.transforms(img)        
        return img
