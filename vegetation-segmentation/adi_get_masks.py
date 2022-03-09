import os, copy, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch, torchvision
from deeplabv3 import FreiburgDataset, FreiburgTestDataset, cf, dn, nut


batch_size=51
test_orig_img_shape = (768, 1024)

fd_test = FreiburgTestDataset("data/freiburg/test")

test_dataloader = torch.utils.data.DataLoader(fd_test, batch_size=batch_size, shuffle=False)
test_img = next(iter(test_dataloader))

deeplabv3_model = torch.load('deeplabv3/results/deeplabv3_model.pt', map_location=torch.device('cpu')).to('cpu')
deeplabv3_model.eval()
_=1

with torch.no_grad():
    
    test_mask_pred = deeplabv3_model(test_img)['out']


for i in range(batch_size):
    test_img_current = dn(test_img)[i].transpose(1,2,0)
    test_mask_current = dn(test_mask_pred)[i]
    
    fig,axs=plt.subplots(2,2)

    axs[0,0].imshow(nut(test_mask_current[0]), cmap='gray',interpolation='nearest')
    axs[0,0].title.set_text('Path')
    axs[0,1].imshow(nut(test_mask_current[1]), cmap='Greens',interpolation='nearest')
    axs[0,1].title.set_text('Grass')
    axs[1,0].imshow(nut(test_mask_current[2]), cmap='Reds',interpolation='nearest')
    axs[1,0].title.set_text('Vegetation')
    axs[1,1].imshow(nut(test_mask_current[4]), cmap='Blues',interpolation='nearest')
    axs[1,1].title.set_text('Sky')
  
    plt.savefig(f'data/freiburg/test/preds/image{i}.png')
    #plt.show()