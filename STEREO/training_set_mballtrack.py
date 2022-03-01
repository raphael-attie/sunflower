import sys, os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import fitstools
from pathlib import PurePath
import glob
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from balltracking import mballtrack as mblt
from skimage.feature import blob_log, blob_dog, peak_local_max
from matplotlib.patches import Ellipse

plt.rcParams.update({'font.size': 12})
dpi = 168
DTYPE = np.float32


def prep_data(image):
    image2 = np.abs(image)
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


def make_ell_points(xc, yc, a, b, imshape):
    ellgridx = np.arange(xc - a, xc + a + 1, dtype=np.int32)
    ellgridy = np.arange(yc - b, yc + b + 1, dtype=np.int32)
    xx, yy = np.meshgrid(ellgridx, ellgridy)
    ell_x = []
    ell_y = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        if x > 0 and y > 0 and x < imshape[1] and y < imshape[0] and (x - xc) ** 2 / a ** 2 + (
                y - yc) ** 2 / b ** 2 <= 1:
            ell_x.append(x)
            ell_y.append(y)

    return ell_x, ell_y


# Data prepped by COR2_tracking_prep.ipynb
datadir = PurePath(os.environ['DATA'], 'STEREO/L7tum/prep_fits')
outputdir = PurePath(os.environ['DATA'], 'STEREO/L7tum/')
datafiles = sorted(glob.glob(str(PurePath(datadir, '*.fits'))))
nfiles = len(datafiles)
nsamples = 10

image = fitstools.fitsread(datafiles[0], cube=False)
blob_thresh = 1.0
overlap = 0.6

surface_inv = -prep_data(image)
blobs_dog = blob_dog(surface_inv, overlap=overlap, threshold=blob_thresh, min_sigma=[6, 2], max_sigma=[20, 10])

init_pos = np.fliplr(blobs_dog[:, 0:2]).T

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
            "init_pos": init_pos,
            "do_plots": False,
            "astropy": True,
            "verbose": False,
            "outputdir": datadir,
            "fig_dir": PurePath(datadir, 'figures')}

mbt = mblt.MBT(polarity=1, **mbt_dict)

n = 0
# Load the image at the current time index
image = fitstools.fitsread(datafiles[n], cube=False)

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

if n == 0:
    axs.plot(mbt.xstart, mbt.ystart, 'r+')


axs.set_xlim([0, 600])
axs.set_ylim([0, 659])

plt.tight_layout()
plt.savefig(PurePath(outputdir, 'figures/training_set', f'blobs_dog_{overlap:0.1}_threshold_{blob_thresh:0.1f}_frame_{n:03d}.png'))
plt.close()

# Create label mask and apply labelled peak detection
label_mask = np.zeros(image.shape, dtype=np.int64)
for i, blob in enumerate(blobs_dog):
    yc, xc, b, a = blob
    ell_x, ell_y = make_ell_points(xc, yc, a, b, image.shape)
    label_mask[ell_y, ell_x] = i+1

peaks = peak_local_max(surface_inv, labels=label_mask, num_peaks_per_label=1)


plt.figure(figsize=(10,9))
plt.imshow(label_mask, origin='lower', vmin=0, vmax=1, cmap='gray')
plt.plot(peaks[:,1], peaks[:,0], 'r+')
plt.xlim([0, 600])
plt.ylim([0, 659])

plt.tight_layout()