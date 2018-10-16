import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, glob
from scipy.misc import imread
import numpy as np

datadir = '/Users/rattie/Data/Ben/SteinSDO'
outputdir = os.path.join(datadir, 'plots')

npts = 9
vx_rates = np.linspace(-0.2, 0.2, npts)

tf_drift0 = sorted(glob.glob(os.path.join(datadir, 'calibration/drift_0', 'data_forward*.png')))
tf_drift4 = sorted(glob.glob(os.path.join(datadir, 'calibration/drift_4', 'data_forward*.png'))) # rate 0
tf_drift8 = sorted(glob.glob(os.path.join(datadir, 'calibration/drift_8', 'data_forward*.png')))
tf_drift0_prep = sorted(glob.glob(os.path.join(datadir, 'calibration/drift_0', 'prep_data_forward_top*.png')))
tf_drift4_prep = sorted(glob.glob(os.path.join(datadir, 'calibration/drift_4', 'prep_data_forward_top*.png'))) # rate 0
tf_drift8_prep = sorted(glob.glob(os.path.join(datadir, 'calibration/drift_8', 'prep_data_forward_top*.png')))

pngs = ([tf_drift0, tf_drift4, tf_drift8],
        [tf_drift0_prep, tf_drift4_prep, tf_drift8_prep])

nframes = len(tf_drift0)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(11,11))
idx = 0
data00 = imread(pngs[0][0][idx])
data01 = imread(pngs[1][0][idx])
data10 = imread(pngs[0][1][idx])
data11 = imread(pngs[1][1][idx])
data20 = imread(pngs[0][2][idx])
data21 = imread(pngs[1][2][idx])

# img00 = axs[0, 0].imshow(data00, vmin=0, vmax=255, cmap='gray', interpolation='lanczos')
# img01 = axs[0, 1].imshow(data01, vmin=0, vmax=255, cmap='gray', interpolation='lanczos')
# img10 = axs[1, 0].imshow(data10, vmin=0, vmax=255, cmap='gray', interpolation='lanczos')
# img11 = axs[1, 1].imshow(data11, vmin=0, vmax=255, cmap='gray', interpolation='lanczos')
# img20 = axs[2, 0].imshow(data20, vmin=0, vmax=255, cmap='gray', interpolation='lanczos')
# img21 = axs[2, 1].imshow(data21, vmin=0, vmax=255, cmap='gray', interpolation='lanczos')

img00 = axs[0, 0].imshow(data00, vmin=0, vmax=255, cmap='gray', interpolation='lanczos')
img01 = axs[0, 1].imshow(data01, vmin=0, vmax=255, cmap='gray', interpolation='lanczos')
img20 = axs[1, 0].imshow(data20, vmin=0, vmax=255, cmap='gray', interpolation='lanczos')
img21 = axs[1, 1].imshow(data21, vmin=0, vmax=255, cmap='gray', interpolation='lanczos')
plt.tight_layout()

imgs = [img00, img01, img20, img21]
# axs[0,0].imshow(data, vmin=0, vmax=255, cmap='gray', interpolation='lanczos')
# plt.tight_layout()

def update_images(idx):

    imgs[0].set_data(imread(pngs[0][0][idx]))
    imgs[1].set_data(imread(pngs[1][0][idx]))
    # imgs[2].set_data(imread(pngs[0][1][idx]))
    # imgs[3].set_data(imread(pngs[1][1][idx]))
    # imgs[4].set_data(imread(pngs[0][2][idx]))
    # imgs[5].set_data(imread(pngs[1][2][idx]))
    imgs[2].set_data(imread(pngs[0][2][idx]))
    imgs[3].set_data(imread(pngs[1][2][idx]))

    return imgs


frames_idx = range(nframes)

ani = animation.FuncAnimation(fig, update_images, frames_idx, blit=True, interval=100)

ani.save(os.path.join(outputdir, 'movie_plots.mp4'), writer='ffmpeg', fps=24)

