#!/bin/bash

wget -c http://deepscene.cs.uni-freiburg.de/static/datasets/freiburg_forest_raw.tar.gz.part-aa
wget -c http://deepscene.cs.uni-freiburg.de/static/datasets/freiburg_forest_raw.tar.gz.part-ab

cat freiburg_forest_raw.tar.gz.part-* > freiburg_forest_raw.tar.gz
