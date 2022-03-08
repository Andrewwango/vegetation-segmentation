import imageio
from os import listdir
from os.path import isfile, join
import glob

counter=0


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
imageio.mimsave('preds.gif', pred_images,fps=2)