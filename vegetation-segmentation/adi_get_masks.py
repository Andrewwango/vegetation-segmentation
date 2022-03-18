import os, copy, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch, torchvision
from deeplabv3 import FreiburgDataset, FreiburgTestDataset, cf, dn, nut
from skimage.transform import resize
from monodepth2 import estimate_depthmap
import pickle


batch_size=101


fd_test = FreiburgTestDataset("data/freiburg/test")

test_dataloader = torch.utils.data.DataLoader(fd_test, batch_size=batch_size, shuffle=False)
test_img = next(iter(test_dataloader))

deeplabv3_model = torch.load('deeplabv3/results/deeplabv3_model.pt', map_location=torch.device('cpu')).to('cpu')
deeplabv3_model.eval()
_=1

with torch.no_grad():
    
    test_mask_pred = deeplabv3_model(test_img)['out']


"""for i in range(batch_size):
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
    #plt.show()"""

mask_resized  = resize(test_mask_pred,  (batch_size, 6, 480, 640))
for i in range(batch_size):
    test_img_current = dn(test_img)[i].transpose(1,2,0)
    test_mask_current = mask_resized[i]

    final_mat=0.3*nut(test_mask_current[0])+0.6*nut(test_mask_current[4])+0.8*nut(test_mask_current[2])+1.2*nut(test_mask_current[1])

    
    final_mat = np.where(final_mat == 1.5, 0.3, final_mat)
    final_mat=np.clip(final_mat,0,1.2)
    plt.imshow(final_mat, cmap='viridis',interpolation='nearest')
    plt.savefig(f'data/freiburg/test/stacked_preds/{i}.jpg')
    #plt.show()

print('Masks done, now for depth')

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

print('Depths done, now for combo')

fd_test.transforms = torchvision.transforms.Compose([Image.fromarray,
                                                     lambda x: x.resize((640, 480), Image.LANCZOS),
                                                     np.array])
test_dataloader3 = torch.utils.data.DataLoader(fd_test, batch_size=batch_size, shuffle=False)
test_img_orig = next(iter(test_dataloader3))

from vegetation_index import vegetation_index

index_dict = vegetation_index(mask_resized, depth_resized)

with open('index_dict.pkl', 'wb') as f:
    pickle.dump(index_dict, f)