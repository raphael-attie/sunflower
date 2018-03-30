import numpy as np
import cv2
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
#import fitstools

# datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
# data = fitstools.fitsread(datafile, tslice=0).astype(np.float32)


img = cv2.imread('/Users/rattie/Dev/sdo_tracking_framework/image_processing/coins.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal.
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# Label the connected components in the sure_fg array. Each connected will have a different positive integer
# Background will be at 0
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0 but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
# Apply watershed
wmarkers = cv2.watershed(img, markers.copy())
img[wmarkers == - 1] = [255, 0, 0]



plt.figure(0, figsize=(19,10))
plt.subplot(241)
plt.imshow(gray, cmap='gray')
#plt.colorbar()
plt.title('Gray-scaled image')

plt.subplot(242)
plt.imshow(thresh, cmap='gray')
#plt.colorbar()
plt.title('Otsu-thresholded binarization')

plt.subplot(243)
plt.imshow(opening, cmap='gray')
#plt.colorbar()
plt.title('White noise removal with opening')

plt.subplot(244)
plt.imshow(sure_bg, cmap='gray')
#plt.colorbar()
plt.title('Sure background in black')

plt.subplot(245)
plt.imshow(dist_transform, cmap='gray')
plt.title('Distance transformed (opening)')

plt.subplot(246)
plt.imshow(sure_fg, cmap='gray')
plt.title("Sure foreground")

plt.subplot(247)
plt.imshow(markers, cmap='jet')
plt.title('Labelled components (unknown = 0)')

plt.subplot(248)
plt.imshow(wmarkers, cmap='jet')
plt.title('Watershed results. Boundary = -1')

plt.tight_layout()

plt.figure(1)
plt.imshow(img)
#plt.colorbar()
plt.title('Gray-scaled image')
plt.tight_layout()
