import imageio
from os import listdir
from os.path import isfile, join
import glob
counter=0
filenames_preds= glob.glob(r"/Users/adityajain/Documents/Modules IIB/4M25 Advanced robotics/vegetation-segmentation/vegetation-segmentation/data/freiburg/test/preds_plus_real/*.jpg")
#filenames_preds=sorted(filenames_preds)
#filenames_preds=filenames_preds.sort(key = lambda x: int(x.split('.jpg')[0]))

pred_images=[]

for i in range(len(filenames_preds)):
    #print(counter)
    
    pred_images.append(imageio.imread(f'preds_plus_real/{counter}.jpg'))
    counter+=1
imageio.mimsave('combi_preds.gif', pred_images,fps=2)