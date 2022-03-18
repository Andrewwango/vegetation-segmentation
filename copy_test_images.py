import shutil, os
import numpy as np
files = []
numbers=np.linspace(300, 2800,101)
int_numbers=numbers.astype(int)
for num in int_numbers:
    if num<1000:
        files.append(f'freiburg_forest_raw/2016-02-26-15-20-48/0{num}.jpg')
    else:
        files.append(f'freiburg_forest_raw/2016-02-26-15-20-48/{num}.jpg')
for f in files:
    shutil.copy(f, 'vegetation-segmentation/data/freiburg/test/img')



"""path = r"/Users/adityajain/Documents/Modules IIB/4M25 Advanced robotics/vegetation-segmentation/vegetation-segmentation/data/freiburg/test/img"

files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.jpg'])))"""