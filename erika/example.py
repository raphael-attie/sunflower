import matplotlib.pyplot as plt
import numpy as np

from erika.roipoly import roipoly

# create image
img = np.ones((100, 100)) * range(0, 100)

# show the image
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
plt.title("left click: line segment         right click: close region")

# let user draw first ROI
ROI1 = roipoly(roicolor='r') #let user draw first ROI

# show the image with the first ROI
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
ROI1.displayROI()
plt.title('draw second ROI')

# let user draw second ROI
ROI2 = roipoly(roicolor='b') #let user draw ROI

# show the image with both ROIs and their mean values
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
[x.displayROI() for x in [ROI1, ROI2]]
[x.displayMean(img) for x in [ROI1, ROI2]]
plt.title('The two ROIs')
plt.show()

# show ROI masks
plt.imshow(ROI1.getMask(img) + ROI2.getMask(img),
          interpolation='nearest', cmap="Greys")
plt.title('ROI masks of the two ROIs')
plt.show()


