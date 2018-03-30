from importlib import reload
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, thin
from skimage.exposure import rescale_intensity
import fitstools
from matplotlib.colors import LinearSegmentedColormap
import balltracking.mballtrack as mblt

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
data = fitstools.fitsread(datafile, tslice=0)
# Get a mask of where to look for local maxima.
threshold = 30
mask_maxi = data >= threshold
local_maxi = peak_local_max(data, indices=False, footprint=np.ones((5, 5)), labels=mask_maxi)
local_maxi_coords = peak_local_max(data, indices=True, footprint=np.ones((5, 5)), labels=mask_maxi)

#markers = mblt.label_from_pos(local_maxi_coords[:,1], local_maxi_coords[:,0], data.shape)
# #markers2 = dilation(markers, selem=np.ones((3,3)))
# labels_ws = watershed(-data, markers, mask=mask_maxi)

labels_ws, markers, borders = mblt.marker_watershed(data, local_maxi_coords[:,1], local_maxi_coords[:,0], threshold)

bordery, borderx = np.where(borders)
# colormap
colors = plt.cm.Set1_r(np.linspace(0, 1, 9))
#colors = plt.cm.Paired(np.linspace(0, 1, 12))
cmap = matplotlib.colors.ListedColormap(colors, name='mycmap', N=local_maxi_coords.shape[0])


plt.figure(0, figsize=(18, 6))
ax1 = plt.subplot(131)
ax1.imshow(data, vmin=-100, vmax=100, cmap='gray', interpolation='nearest', origin='lower')
ax1.scatter(local_maxi_coords[:,1], local_maxi_coords[:,0], s=2, edgecolors='r', label='max within 5 px')
ax1.set_title('magnetogram & local maxima (markers)')

ax2 = plt.subplot(132, sharex = ax1, sharey = ax1)
ax2.imshow(markers, cmap=cmap, interpolation='nearest', origin='lower')
ax2.imshow(borders.astype(np.uint8), vmin=0, vmax=1, origin='lower', cmap='gray', alpha=0.3)

plt.title('Dilated markers & watershed boundaries')

ax3= plt.subplot(133, sharex = ax1, sharey = ax1)
plt.imshow(labels_ws, cmap=cmap, interpolation='nearest', origin='lower')
plt.title('Watershed labeled connected components')

plt.tight_layout()

