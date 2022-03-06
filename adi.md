1. download a bit of the Freiburg forest raw using `bash download_freiburg_forest_raw_part.sh` and extract (probs will get some errors)
2. copy some files into vegetation-segmentation/data/test/img.
3. create model from model parts by running `cd vegetation-segmentation/deeplabv3/results`, `sh model_merge.sh`
4. Run analysis and visualisation in analysis.ipynb (you can change batch_size to include more data in one batch)
5. create video from raw image sequence (maybe downsample the data because the original frames are 20Hz), in vis_video.ipynb
6. have next to it a video of the predicted masks: maybe road, vegetation (and trees) coloured per mask?
7. per frame, estimate depth map, and calculate vegetation index using vegetation_index (see analysis.ipynb). Obtain time sequence of vegetation index over all frames. Maybe smooth it? Then plot time sequence of vegetation index throughout frame using:

    from matplotlib.animation import FuncAnimation
    def save_time_seq(seq, title, name): #univariate sequence
        x = np.arange(len(seq))
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set(xlim=(x.min(), x.max()), ylim=(seq.min(), seq.max()), title=title)
        line = ax.plot(x[0], seq[0], color='k', lw=2)[0]
        anim = FuncAnimation(fig, lambda i: line.set_data(x[:i], seq[:i]), interval=80, frames=len(seq)-1)
        anim.save(f'{name}')