import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import fitstools
import skimage.morphology
from scipy.ndimage.morphology import binary_dilation, binary_opening, distance_transform_edt
from scipy.ndimage.measurements import center_of_mass
import cv2

fs = 9

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'

frame_number = 80
cont = fitstools.fitsread(datafile, tslice=frame_number).astype(np.float32)
ymin, xmin = np.unravel_index(np.argmin(cont, axis=None), cont.shape)
fov = [0, 200, 0, 200]
scont = cont[fov[2]:fov[3], fov[0]:fov[1]]
sfactor = 5
threshold = scont.mean() - sfactor*scont.std()
# Mask based on standard deviation
spot_mask = cont < threshold
# Dilation
se = skimage.morphology.disk(6)
spot_mask_dilated = binary_dilation(spot_mask, structure=se)
spot_mask_dilated2 = binary_dilation(spot_mask_dilated, structure=skimage.morphology.disk(25))
spot_dist = distance_transform_edt(spot_mask_dilated)
ymax2, xmax2 = np.unravel_index(np.argmax(spot_dist, axis=None), spot_dist.shape)
ycom, xcom = center_of_mass(spot_mask_dilated)
# Make a mask of same size as image, unmasking a disk-shaped area of given radius.
xm, ym = np.meshgrid(np.arange(cont.shape[1]), np.arange(cont.shape[0]))
r = np.sqrt((xm - xcom)**2 + (ym -ycom)**2)
rm1 = r < spot_dist.max()

#
pimages0_rgb[i, :, :, k] = cv2.linearPolar(imagesf[i, :, :, k], center, nx, cv2.INTER_LANCZOS4 + cv2.WARP_FILL_OUTLIERS)

# Continuum
figsize = (9, 9)
fig, axs = plt.subplots(2,2, figsize=figsize)
im1 = axs[0,0].imshow(cont, cmap='gray', origin='lower')
#axs[0].contour(cont, levels = [threshold], colors='orange')
#axs[0].plot(xmin, ymin, 'r.')
# axs[0].contour(spot_mask_dilated.astype(int), 1, colors='red')
# axs[0].contour(spot_mask_dilated2.astype(int), 1, colors='green')
axs[0,1].imshow(spot_mask, cmap='gray', origin='lower')
axs[1,0].imshow(spot_mask_dilated, cmap='gray', origin='lower')
axs[1,1].imshow(spot_dist, cmap='gray', origin='lower')
#axs[0].plot(xmax2, ymax2, 'r+')
axs[0,0].plot(xcom, ycom, 'g.')
axs[0,0].contour(rm1, 1, colors='red')

fig.tight_layout()
