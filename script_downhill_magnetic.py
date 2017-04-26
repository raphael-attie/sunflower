"""
Script testing the dowhill-clumping algorithm.
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from image_processing import fragments


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * sig**2))


def make2DGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


gpos = make2DGaussian(100, 20) * 200
gneg = -make2DGaussian(50, 10) * 200


# # Load an HMI magnetogram
# data_dir = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/'
# # Get the files from the local data directory
# files   = glob.glob(os.path.join(data_dir, '*.fits'))
# # pick one
# f = files[3000]
# hdu = fits.open(f)
# hdu.verify('silentfix')
# header = hdu[1].header
# data = hdu[1].data
#
#
# data = np.zeros([200, 200])
# data[100:150, 100:150] = -100
# data[115:135, 115:135] = -150
#
# data[10:60, 10:60] = 100
# data[15:35, 15:35] = 150

data = np.zeros([300, 300])

data[10:110, 10:110] = gpos
data[120:170, 120:170] = gneg

# magnetic threshold for the downhill algorithm
threshold = 50

labels_pos, labels_neg = fragments.detect(data, threshold)

mask = -labels_neg.astype(np.int32) + labels_pos.astype(np.int32)

plt.figure(0)
plt.subplot(221)
plt.imshow(data, cmap='gray')
plt.subplot(222)
plt.imshow(mask, cmap='gray')
plt.subplot(223)
plt.imshow(labels_pos, cmap='gray')
plt.subplot(224)
plt.imshow(labels_neg, cmap='gray')



