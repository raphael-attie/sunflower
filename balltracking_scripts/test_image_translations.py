from importlib import reload
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib import cm
import fitstools
import fitsio
import filters

import scipy.ndimage.interpolation as sci_interp

file = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/series_continuum/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum_00000.fits'
#file = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/series_continuum_calibration/calibration/drift0.20/drift_0001.fits'
# Get the header
h       = fitstools.fitsheader(file)
# Get the 1st image
image   = fitsio.read(file).astype(np.float32)
image1  = image.copy()
image2  = image.copy()

# Shifts amount
dx = 110.3
dx1 = 0.3
dy = 0
# Shift the image

for i in range(2):
    image1 = filters.translate_by_phase_shift(image1, dx, dy)
    image2 = sci_interp.shift(image2, (dy, dx), order=3, mode='wrap')
    dx *= -1
    dx1 *= -1

mad1 = np.abs(image - image1)[150:350, 200:400].mean()
mad2 = np.abs(image - image2)[150:350, 200:400].mean()

plt.figure(0, figsize = (18, 10))
plt.subplot(151)
plt.imshow(image, origin='lower', cmap='gray')
plt.axis([200, 400, 150, 350])
plt.title('Original [MAD = 0]')
plt.subplot(152)
plt.imshow(image1, origin='lower', cmap='gray')
plt.axis([200, 400, 150, 350])
plt.title('Phase shift 2x [MAD = %0.1f]'%mad1)
plt.subplot(153)
plt.imshow(image1, origin='lower', cmap='gray')
plt.axis([200, 400, 150, 350])
plt.title('Interpolation 2x [MAD = %0.1f]'%mad2)

for i in range(100):
    image1 = filters.translate_by_phase_shift(image1, dx, dy)
    image2 = sci_interp.shift(image2, (dy, dx), order=3, mode='wrap')
    dx *= -1
    dx1 *= -1

mad1 = np.abs(image - image1)[150:350, 200:400].mean()
mad2 = np.abs(image - image2)[150:350, 200:400].mean()

plt.subplot(154)
plt.imshow(image1, origin='lower', cmap='gray')
plt.axis([200, 400, 150, 350])
plt.title('Phase shift 100x [MAD = %0.1f]'%mad1)
plt.subplot(155)
plt.imshow(image1, origin='lower', cmap='gray')
plt.axis([200, 400, 150, 350])
plt.title('Interpolation 100x [MAD = %0.1f]'%mad2)

plt.tight_layout()



image3  = image.copy()

for i in range(2):
    image3 = filters.translate_by_phase_shift(image3, dx1, dy)
    dx1 *= -1

mad3 = np.abs(image - image3)[150:350, 200:400].mean()

plt.figure(1, figsize = (18, 10))
plt.subplot(131)
plt.imshow(image, origin='lower', cmap='gray')
plt.axis([200, 400, 150, 350])
plt.title('Original [MAD = 0]')

plt.subplot(132)
plt.imshow(image3, origin='lower', cmap='gray')
plt.axis([200, 400, 150, 350])
plt.title('Original [MAD = %0.1f]'%mad3)

for i in range(100):
    image3 = filters.translate_by_phase_shift(image3, dx1, dy)
    dx1 *= -1

mad3 = np.abs(image - image3)[150:350, 200:400].mean()

plt.subplot(133)
plt.imshow(image3, origin='lower', cmap='gray')
plt.axis([200, 400, 150, 350])
plt.title('Original [MAD = %0.1f]'%mad3)

