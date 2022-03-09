import imageio
from os import listdir
from os.path import isfile, join
import glob

counter=0

"""
images = []

filenames_imgs= glob.glob(r"/Users/adityajain/Documents/Modules IIB/4M25 Advanced robotics/vegetation-segmentation/vegetation-segmentation/data/freiburg/test/img/*0.jpg")
filenames_imgs=sorted(filenames_imgs)



#print(filenames_imgs)
for filename in filenames_imgs:
    #print(counter)
    #counter+=1
    images.append(imageio.imread(filename))
imageio.mimsave('real.gif', images,fps=2)


filenames_preds= glob.glob(r"/Users/adityajain/Documents/Modules IIB/4M25 Advanced robotics/vegetation-segmentation/vegetation-segmentation/data/freiburg/test/preds/*.png")
filenames_preds=sorted(filenames_preds)

pred_images=[]

for filename in filenames_preds:
    #print(counter)
    #counter+=1
    pred_images.append(imageio.imread(filename))
imageio.mimsave('preds.gif', pred_images,fps=2)"""

import sys
from PIL import Image
import matplotlib.pyplot as plt
import re

filenames_imgs= glob.glob(r"/Users/adityajain/Documents/Modules IIB/4M25 Advanced robotics/vegetation-segmentation/vegetation-segmentation/data/freiburg/test/img/*0.jpg")
filenames_imgs=sorted(filenames_imgs)

filenames_preds= glob.glob(r"/Users/adityajain/Documents/Modules IIB/4M25 Advanced robotics/vegetation-segmentation/vegetation-segmentation/data/freiburg/test/stacked_preds/*.jpg")
#filenames_preds=filenames_preds.sort(key = lambda x: int(x.split('mage')[1].split('.jpg')[0]))
#filenames_preds=filenames_preds.sort(key = lambda x: int(re.split(r'[mage,.png]',x)[2]))

#print(filenames_preds)

for count in range(len(filenames_preds)):
    if counter<=13:
        images = [Image.open(x) for x in [f'img/0{50*counter+300}.jpg',f'stacked_preds/image{counter}.jpg']]
    else:
        images = [Image.open(x) for x in [f'img/{50*counter+300}.jpg',f'stacked_preds/image{counter}.jpg']]
    
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(f'preds_plus_real/{counter}.jpg')
    counter+=1