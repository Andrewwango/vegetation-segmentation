#!/bin/bash

wget -c http://deepscene.cs.uni-freiburg.de/static/datasets/freiburg_forest_annotated.tar.gz.part-aa
wget -c http://deepscene.cs.uni-freiburg.de/static/datasets/freiburg_forest_annotated.tar.gz.part-ab
wget -c http://deepscene.cs.uni-freiburg.de/static/datasets/freiburg_forest_annotated.tar.gz.part-ac
cat freiburg_forest_annotated.tar.gz.part-a* > freiburg_forest_annotated.tar.gz

echo "unzipping..."
tar -xvzf community_images.tar.gz
rm -y freiburg_forest_annotated.tar.gz*

echo "moving"
mv "freiburg_forest_annotated/train/rgb/*.jpg" "forest-mask-rcnn/data/freiburg/train/img"
mv "freiburg_forest_annotated/train/GT_color/*.jpg" "forest-mask-rcnn/data/freiburg/train/mask"
mv "freiburg_forest_annotated/test/rgb/*.jpg" "forest-mask-rcnn/data/freiburg/test/img"
mv "freiburg_forest_annotated/test/GT_color/*.jpg" "forest-mask-rcnn/data/freiburg/test/mask"
