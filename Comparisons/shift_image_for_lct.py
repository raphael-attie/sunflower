import os, glob
import matplotlib
import numpy as np
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
import filters
import fitsio
from skimage.feature import register_translation


filepath = '/Users/rattie/Data/Ben/SteinSDO/SDO_int290040.fits'
outputdir = '/Users/rattie/Data/Ben/SteinSDO/calibration/sanity_check_shifted_images'

image = fitsio.read(filepath)

# Velocity offset. Will be applied to both x and y axis
drifts = np.arange(0.1, 2.1, 0.1)

for drift in drifts:
    shifted_image = filters.translate_by_phase_shift(image, -drift, -drift)
    shift, error, diffphase = register_translation(image, shifted_image, 100)
    print('[{:0.2f}, {:0.2f}]'.format(-shift[0], -shift[1]))
    filename = os.path.join(outputdir, 'shifted_{:0.1f}.fits'.format(drift))
    fitsio.write(filename, shifted_image)


# Shift pairs of slightly time-spaced images
files = sorted(glob.glob('/Users/rattie/Data/Ben/SteinSDO/SDO_int*.fits'))

dts = [1, 3, 6, 9]

for dt in dts:
    outputdir_dt = os.path.join(outputdir, 'dt_{:d}'.format(dt))
    if not os.path.exists(outputdir_dt):
        os.makedirs(outputdir_dt)
    image = fitsio.read(files[dt])
    for drift in drifts:
        shifted_image = filters.translate_by_phase_shift(image, -drift, -drift)
        filename = os.path.join(outputdir_dt, 'shifted_{:0.1f}.fits'.format(drift))
        fitsio.write(filename, shifted_image)

# vmin = np.percentile(image, 1)
# vmax = np.percentile(image, 99)
# vmin2 = np.percentile(shifted_image, 1)
# vmax2 = np.percentile(shifted_image, 99)
#
# fig, ax = plt.subplots(2,1, figsize=(8,10))
# ax[0].imshow(image, vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
# ax[0].set_title('Original image')
# ax[1].imshow(shifted_image, vmin=vmin2, vmax=vmax2, origin='lower', cmap='gray')
# ax[1].set_title('Shifted image')
# plt.tight_layout()
# plt.show()
