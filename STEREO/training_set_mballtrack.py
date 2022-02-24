import sys, os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import fitstools
from pathlib import PurePath
import glob
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from astropy.io.fits import getheader
from balltracking import mballtrack as mblt
from skimage.feature import peak_local_max
from skimage.feature import blob_log, blob_dog
from matplotlib.patches import Ellipse

plt.rcParams.update({'font.size': 12})
dpi = 168
DTYPE = np.float32


def prep_data(image):
    image2 =np.abs(image)
    image3 = image2.max() - image2
    surface = (image3 - image3.mean())/image3.std()
    return surface.copy(order='C').astype(DTYPE)

def add_colorbar(axes, image_object):
    # position for the colorbar
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='1%', pad=0.3)
    # Adding the colorbar
    cbar = plt.colorbar(im, cax=cax)
    return cbar

# Data prepped by COR2_tracking_prep.ipynb
datadir = PurePath(os.environ['DATA'], 'STEREO/L7tum/prep_fits')
outputdir = PurePath(os.environ['DATA'], 'STEREO/L7tum/')
datafiles = sorted(glob.glob(str(PurePath(datadir, '*.fits'))))
nfiles = len(datafiles)
nsamples = 10

mbt_dict = {"nt": nsamples,
            "rs": 4,
            "am": 1,
            "dp": 0.3,
            "tdx": 1,
            "tdy": 100,
            "zdamping": 1,
            "ballspacing": 15,
            "intsteps": 20,
            "mag_thresh": 3.5,
            "noise_level": 2,
            "track_emergence": False,
            "prep_function": prep_data,
            "datafiles": datafiles,
            "do_plots": False,
            "astropy": True,
            "verbose": False,
            "outputdir": datadir,
            "fig_dir": PurePath(datadir, 'figures')}

mbt = mblt.MBT(polarity=1, **mbt_dict)

n = 0
# Load the image at the current time index
image = fitstools.fitsread(datafiles[n], cube=False)
# Search for local extrema
xpeak, ypeak = mblt.get_local_extrema(image, mbt.polarity, 1, mbt.noise_level)
# Create their labels. They can change at each new image
peak_labels = np.arange(0, len(xpeak))
peak_labels_str = [str(l) for l in peak_labels]

fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8, 7))
im = axs.imshow(mbt.surface, vmin=-2, vmax=0, origin='lower', cmap='Greys_r')
axs.plot(xpeak, ypeak, 'g.', ms=4)

cbar = add_colorbar(axs, im)

if n == 0:
    axs.plot(mbt.xstart, mbt.ystart, 'r+')
    labels = np.arange(0, mbt.nballs)
    labels_str = [str(l) for l in labels]
    for i, l in enumerate(labels_str):
        axs.text(mbt.xstart[i]+5, mbt.ystart[i], l, color='black', fontsize=10,
                 bbox=dict(facecolor='yellow', alpha=0.4, edgecolor='none', pad=1),
                 clip_on=True)

axs.set_xlim([0, 600])
axs.set_ylim([0, 659])
plt.tight_layout()
plt.savefig(PurePath(outputdir, 'figures/training_set', f'peaks_{n:03d}.png'))

blob_thresh = 1.0
overlap = 0.6

blobs_dog = blob_dog(-mbt.surface, overlap=overlap, threshold=blob_thresh, min_sigma=[6, 2], max_sigma=[20, 10])

fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(10, 9))
im = axs.imshow(-mbt.surface, vmin=0, vmax=3, origin='lower', cmap='Greys')
cbar = add_colorbar(axs, im)

for b, blob in enumerate(blobs_dog):
    blob_label = str(b)
    y, x, sigma_rows, sigma_cols = blob
    r1 = sigma_rows * np.sqrt(2) * 2
    r2 = sigma_cols * np.sqrt(2) * 2
    ell = Ellipse((x, y), r2, r1, linewidth=1, fill=False, color='green', linestyle='-')
    axs.add_artist(ell)
    axs.plot(x, y, 'g.')
    axs.text(x+1, y+1, blob_label, color='black', fontsize=8,
             bbox=dict(facecolor='yellow', alpha=0.4, edgecolor='black', pad=1), clip_on=True)

for b, blob in enumerate(blobs_log):
    blob_label = str(b)
    y, x, sigma_rows, sigma_cols = blob
    r1 = sigma_rows * np.sqrt(2) * 2
    r2 = sigma_cols * np.sqrt(2) * 2
    ell = Ellipse((x, y), r2, r1, linewidth=1, fill=False, color='red', linestyle='-')
    axs.add_artist(ell)
    # axs.text(x+1, y+1, blob_label, color='black', fontsize=8,
    #          bbox=dict(facecolor='yellow', alpha=0.4, edgecolor='black', pad=1), clip_on=True)

if n == 0:
    axs.plot(mbt.xstart, mbt.ystart, 'r+')


axs.set_xlim([0, 600])
axs.set_ylim([0, 659])

plt.tight_layout()
plt.savefig(PurePath(outputdir, 'figures/training_set', f'blobs_dog_log_overlap_{overlap:0.1}_threshold_{blob_thresh:0.1f}_frame_{n:03d}.png'))
plt.close()
