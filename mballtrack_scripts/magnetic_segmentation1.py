
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import random_walker, find_boundaries
from scipy import ndimage

# Generate an initial image with two overlapping circles
x, y = np.indices((80, 80))
x1, y1, x2, y2 = 28, 28, 44, 52
r1, r2 = 16, 20
mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1 ** 2
mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2 ** 2
image = np.logical_or(mask_circle1, mask_circle2)
# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance
# to the background
distance = ndimage.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
markers = measure.label(local_maxi)

labels_ws = watershed(-distance, markers, mask=image)
borders_ws = find_boundaries(labels_ws)
#labels_ws[borders_ws] = labels_ws.max()+1

markers2 = markers.copy()
markers2[~image] = -1
labels_rw = random_walker(image, markers2)
labels_rw[markers2==-1] =0
borders_rw = find_boundaries(labels_rw)
#labels_rw[borders_rw] = labels_rw.max()+1

plt.figure(figsize=(7, 10))
plt.subplot(321)
plt.imshow(image, cmap='gray', interpolation='nearest')
plt.title('image')
plt.subplot(322)
plt.imshow(-distance, interpolation='nearest', cmap='jet')
plt.title('distance map')
plt.subplot(323)
plt.imshow(markers, cmap='nipy_spectral', interpolation='nearest')
plt.title('markers for watershed')
plt.subplot(324)
plt.imshow(labels_ws, cmap='nipy_spectral', interpolation='nearest')
plt.title('labels from watershed')
plt.subplot(325)
plt.imshow(markers2, cmap='nipy_spectral', interpolation='nearest')
plt.title('markers for random walker')
plt.subplot(326)
plt.imshow(labels_rw, cmap='nipy_spectral', interpolation='nearest')
plt.title('labels from random walker')
plt.tight_layout()

