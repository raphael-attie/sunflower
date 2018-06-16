import matplotlib
matplotlib.use('macosx')
import numpy as np
from sunpy.net.helioviewer import HelioviewerClient
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
from matplotlib.colors import Normalize
from datetime import datetime, timedelta
import os
import glymur

data_datetime = datetime(2012, 8, 31, 18, 30, 0)

directory = '/Users/rattie/Data/AIA'
hv = HelioviewerClient()
wlgth = ['304', '171', '193']
#filepng = [hv.download_png('2012/08/31 18:30:00', 2.4, "[SDO,AIA,AIA,%s,1,100]"%w, x0=0, y0=0, width=1024, height=1024, directory='/Users/rattie/Data/AIA', overwrite=True) for w in wlgth]

filejp2 = [hv.download_jp2(data_datetime, observatory='SDO', instrument='AIA', detector='AIA', measurement=wlgth[0], directory=directory, overwrite=True),
           hv.download_jp2(data_datetime, observatory='SDO', instrument='AIA', detector='AIA', measurement=wlgth[1], directory=directory, overwrite=True),
           hv.download_jp2(data_datetime, observatory='SDO', instrument='AIA', detector='AIA', measurement=wlgth[2], directory=directory, overwrite=True)]

# img304png = cv2.imread(filepng[0])
# img304png = img304png.mean(axis=2)

jp2 = [glymur.Jp2k(f) for f in filejp2]

img304 = jp2[0][:]
img171 = jp2[1][:]
img193 = jp2[2][:]

vmin = 40
vmax = 254
img304N, img171N, img193N = [Normalize(vmin, vmax)(img) for img in [img304, img171, img193]]

r = 1.5
g = 1.2
b = 0.7
imgstack1 = np.stack([np.clip(img304N, 0, 1), np.clip(img171N, 0, 1), np.clip(img193N, 0, 1)], axis=-1)
imgstack2 = np.stack([np.clip(img304N*r, 0, 1), np.clip(img171N*g, 0, 1), np.clip(img193N*b, 0, 1)], axis=-1)

plt.figure(figsize=(18,10))
plt.subplot(121)
plt.imshow(imgstack1)
plt.title('R, G, B = 304, 171, 193')
plt.axis('off')
plt.subplot(122)
plt.imshow(imgstack2)
plt.title('balance RGB = [%0.1f, %0.1f, %0.1f]'%(r,g,b))
plt.axis('off')
plt.tight_layout()
