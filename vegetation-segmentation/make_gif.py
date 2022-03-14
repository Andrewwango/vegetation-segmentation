import imageio
from os import listdir
from os.path import isfile, join
import glob
counter=0
filenames_preds= glob.glob(r"/Users/adityajain/Documents/Modules IIB/4M25 Advanced robotics/vegetation-segmentation/vegetation-segmentation/data/freiburg/test/stacked_preds/*.jpg")
#filenames_preds=sorted(filenames_preds)
#filenames_preds=filenames_preds.sort(key = lambda x: int(x.split('.jpg')[0]))

pred_images=[]
print(len(filenames_preds))

#or GIF of real images, uncomment sectio below

"""for i in range(len(filenames_preds)):
    #print(counter)
    if counter<=27:
        pred_images.append(imageio.imread(f'vegetation-segmentation/data/freiburg/test/img/0{25*counter+300}.jpg'))
    else:
        pred_images.append(imageio.imread(f'vegetation-segmentation/data/freiburg/test/img/{25*counter+300}.jpg'))
    counter+=1
imageio.mimsave('vegetation-segmentation/data/freiburg/test/orig_images.gif', pred_images,fps=3)"""

#For GIF of depths, uncomment section below

"""for i in range(len(filenames_preds)):
    #print(counter)
    
    pred_images.append(imageio.imread(f'vegetation-segmentation/data/freiburg/test/depths/{counter}.jpg'))
    
    counter+=1
imageio.mimsave('vegetation-segmentation/data/freiburg/test/depth_images.gif', pred_images,fps=3)"""

#For GIF of predictions, uncomment section below

"""counter=0
for i in range(len(filenames_preds)):
    #print(counter)
    
    pred_images.append(imageio.imread(f'vegetation-segmentation/data/freiburg/test/stacked_preds/{counter}.jpg'))
    
    counter+=1
imageio.mimsave('vegetation-segmentation/data/freiburg/test/preds.gif', pred_images,fps=3)"""
