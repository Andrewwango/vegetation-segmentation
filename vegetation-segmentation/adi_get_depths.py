import os, copy, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch, torchvision
from deeplabv3 import FreiburgDataset, FreiburgTestDataset, cf, dn, nut,c2e
from monodepth2 import estimate_depthmap
from skimage.transform import resize

batch_size=101
test_orig_img_shape = (480,640)#(768, 1024)

fd_test = FreiburgTestDataset("data/freiburg/test")
test_dataloader = torch.utils.data.DataLoader(fd_test, batch_size=batch_size, shuffle=False)
test_img = next(iter(test_dataloader))

fd_test.transforms = torchvision.transforms.Compose([Image.fromarray,
                                                     lambda x: x.resize((640, 192), Image.LANCZOS), 
                                                     torchvision.transforms.ToTensor()])
test_dataloader2 = torch.utils.data.DataLoader(fd_test, batch_size=batch_size, shuffle=False)
test_img2 = next(iter(test_dataloader2))
test_depth_pred = estimate_depthmap(test_img2)
test_img2 = dn(test_img2)
depth_resized = resize(test_depth_pred, (batch_size, 480, 640))
#fig,axs=plt.subplots(batch_size,2, figsize=(20,20))
for i in range(batch_size):
    #axs[i,0].imshow(c2e(test_img2[i]))
    plt.imshow(depth_resized[i], cmap='gray' )
    plt.savefig(f'data/freiburg/test/depths/{i}.jpg')