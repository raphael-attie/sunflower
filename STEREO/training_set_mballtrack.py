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


def add_colorbar(axes, image_object, title=''):
    # position for the colorbar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='1%', pad=0.3)
    # Adding the colorbar
    cbar = plt.colorbar(image_object, cax=cax)
    cbar.ax.set_ylabel('Data surface depth')
    return cbar


def make_ell_points(xc, yc, rx, ry, imshape):
    ellgridx = np.arange(xc - rx, xc + rx + 1, dtype=np.int32)
    ellgridy = np.arange(yc - ry, yc + ry + 1, dtype=np.int32)
    xx, yy = np.meshgrid(ellgridx, ellgridy)
    ell_x = []
    ell_y = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        if 0 < x < imshape[1] and 0 < y < imshape[0] and (x - xc) ** 2 / rx ** 2 + (y - yc) ** 2 / ry ** 2 <= 1:
            ell_x.append(x)
            ell_y.append(y)

    return ell_x, ell_y


def make_label_mask(ycen, xcen, sig_b, sig_a):
    rad_x = sig_a * np.sqrt(2)
    rad_y = sig_b * np.sqrt(2)
    ell = Ellipse((xcen, ycen), rad_x * 2, rad_y * 2, linewidth=1, fill=False, color='green', linestyle='-')
    axs.add_artist(ell)
    ell_x, ell_y = make_ell_points(xcen, ycen, rad_x, rad_y, image.shape)
    label_mask = np.zeros(image.shape, dtype=np.int64)
    label_mask[ell_y, ell_x] = 1
    peak = peak_local_max(surface_inv, labels=label_mask, num_peaks_per_label=1)
    return peak


# Data prepped by COR2_tracking_prep.ipynb
datadir = PurePath(os.environ['DATA'], 'STEREO/L7tum/prep_fits')
outputdir = PurePath(os.environ['DATA'], 'STEREO/L7tum/')
datafiles = sorted(glob.glob(str(PurePath(datadir, '*.fits'))))
nfiles = len(datafiles)
nsamples = 10

image = fitstools.fitsread(datafiles[0], cube=False)
surface_inv = -prep_data(image)
blob_thresh = 1.0
overlap = 0.6
# Toggle blob detection: if false, will load existing files
detect_blob = False
# Toggle 1st / 2nd selection pass
first_select = True


peaks_time_series = []
for n in range(0, 10):
    print('Frame n= ', n)
    # n = 0
    # Load the image at the current time index
    image = fitstools.fitsread(datafiles[n], cube=False)
    surface_inv = -prep_data(image)
    if detect_blob:
        blobs_d = blob_dog(surface_inv, overlap=overlap, threshold=blob_thresh, min_sigma=[5, 1], max_sigma=[20, 10])
        np.save(PurePath(outputdir, 'training_set', f'blobs_{n:02d}.npy'), blobs_d)
    else:
        blobs_d = np.load(PurePath(outputdir, 'training_set', f'blob_{n:02d}.npy'))

    plt.close('all')
    fig, axs = plt.subplots(figsize=(16, 14))
    im = axs.imshow(surface_inv, vmin=0, vmax=3, origin='lower', cmap='Greys')
    axs.set_xlim([0, 600])
    axs.set_ylim([0, 659])
    axs.set_xlabel('Azimuth [px] - 1px = 0.1 deg')
    axs.set_ylabel('Radial distance [px]')
    cbar = add_colorbar(axs, im)
    plt.tight_layout()

    if first_select:
        targets = []
        figure_title = PurePath(outputdir, 'training_set', f'training_frame_v1_{n:03d}.jpg')
    else:
        targets = np.load(PurePath(outputdir, 'training_set', 'targets'))
        figure_title = PurePath(outputdir, 'training_set', f'training_frame_v2_{n:03d}.jpg')

    peaks = []
    for i, blob in enumerate(blobs_d):
        if i not in targets:
            yc, xc, b, a = blob
            peak = make_label_mask(yc, xc, b, a)

            # Only consider if close enough from the geometric center of the ellipse
            if len(peak) > 0:
                peaks.append(peak[0])
                ypeak, xpeak = peak[0]
                axs.plot(xpeak, ypeak, 'r+')
                # print(i, xc, yc, a, b)
                ry = b * np.sqrt(2)
                rx = a * np.sqrt(2)
                xrel = abs(peak[0][1] - xc)/rx
                yrel = abs(peak[0][0] - yc)/ry
                if xrel <= 0.8 and yrel <= 0.5:
                    axs.text(xc+1, yc+1, str(i), color='black', fontsize=10,
                             bbox=dict(facecolor='yellow', alpha=0.4, edgecolor='black', pad=1), clip_on=True)
            else:
                peaks.append(np.array([np.NaN, np.NaN]))
        else:
            peaks.append(np.array([np.NaN, np.NaN]))

    plt.savefig(PurePath(outputdir, 'training_set/figures', figure_fname))
    plt.close()

    peaks = np.array(peaks)
    peaks_time_series.append(peaks)


targets = np.array([
    [21, 27, 32, 42, 59, 100, 185, 358, 643, 1155],
    [52, 70, 92, 97, 93, 75, 66, 56, 55, 53],
    [159, 141, 148, 157, 165, 164, 170, 181, 174, 167],
    [346, 350, 341, 331, 319, 296, 286, 262, 249, 246],
    [369, 403, 386, 376, 353, 289, 233, 177, 148, 125],
    [409, 385, 361, 336, 320, 292, 269, 247, 228, 224],
    [335, 275, 246, 215, 191, 219, 258, 274, 305, 374],
    [492, 539, 547, 545, 498, 467, 445, 467, 517, 528],
    [9, 10, 9, 10, 13, 23, 29, 38, 62, 108],
    [6, 7, 8, 8, 9, 10, 12, 14, 13, 12],
    [270, 308, 336, 382, 400, 416, 374, 325, 291, 273],
    [590, 566, 569, 601, 639, 639, 597, 586, 564, 508],
    [127, 115, 116, 112, 113, 105, 101, 115, 131, 155],
    [169, 170, 183, 201, 208, 198, 178, 173, 165, 160],
    [1354, 1350, 1297, 1251, 1201, 1090, 1003, 965, 953, 946],
    [1172, 1062, 1031, 997, 1019, 1030, 1073, 1163, 1309, 1536],
    [915, 1006, 1095, 1155, 1228, 1325, 1508, 1631, 1658, 1874],
    [277, 204, 155, 119, 95, 70, 54, 45, 38, 27],
    [774, 747, 743, 739, 702, 644, 601, 588, 596, 607],
    [739, 654, 618, 620, 617, 610, 604, 695, 776, 942]
])
np.save(PurePath(outputdir, 'training_set', 'targets'), targets)

