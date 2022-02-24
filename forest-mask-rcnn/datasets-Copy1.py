import os
import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset

class FreiburgDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, transforms=None, ext_img='.jpg', ext_mask='.png'):
        self.root = root
        self.transforms = transforms
        self.imgs  = [p for p in sorted(os.listdir(os.path.join(root, "img")))  if p.endswith(ext_img)]
        self.masks = [p for p in sorted(os.listdir(os.path.join(root, "mask"))) if p.endswith(ext_mask)]
        
        self.obj_ids = np.array([[170, 170, 170], 
                                 [0, 255, 0],
                                 [102, 102, 51],
                                 [0, 60, 0],
                                 [0, 120, 255],
                                 [0, 0, 0]])
        
    def __getitem__(self, idx):
        img_path  = os.path.join(self.root, "img",  self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        masks = np.all(mask == self.obj_ids[:, None, None], axis=3).astype(np.uint8)
        

        # get bounding box coordinates for each mask
        num_objs = len(self.obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

