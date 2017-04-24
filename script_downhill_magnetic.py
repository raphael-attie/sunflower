"""
Script testing the dowhill-clumping algorithm.
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from image_processing import fragments


# Load an HMI magnetogram
data_dir = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/'
# Get the files from the local data directory
files   = glob.glob(os.path.join(data_dir, '*.fits'))
# pick one
f = files[3000]
hdu = fits.open(f)
hdu.verify('silentfix')
header = hdu[1].header
data = hdu[1].data

# magnetic threshold for the downhill algorithm
threshold = 50

labels_pos, labels_neg = fragments.detect(data, threshold)

mask = -labels_neg.astype(np.int32) + labels_pos.astype(np.int32)

plt.figure(0)
plt.subplot(131)
plt.imshow(labels_neg, cmap='gray')
plt.subplot(132)
plt.imshow(labels_pos, cmap='gray')
plt.subplot(133)
plt.imshow(mask, cmap='gray')


