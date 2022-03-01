1. download a bit of the Freiburg forest raw using `bash download_freiburg_forest_raw_part.sh` and extract (probs will get some errors)
2. copy some files into data/test/img.
3. create model by running `sh model_merge.sh`
4. Run analysis and visualisation in analysis.ipynb (you can change batch_size to include more data in one batch)
5. create video from raw images (maybe downsample the data because the original frames are 20Hz)
6. have next to it a video of the predicted masks: maybe road, vegetation and trees?
7. per frame, aggregate (or don't) vegetation, trees and grass and sum the number of pixels/sum the pre-thresholded softmaxes to get a vegetation number per frame, and plot that for the whole video, or throughout the whole video.