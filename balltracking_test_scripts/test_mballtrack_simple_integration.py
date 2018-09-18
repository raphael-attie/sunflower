"""
Test magnetic balltracking without emergence detection and without balls replacement
This checks whether the balls do settle in and track the local minima
"""

from importlib import reload
import multiprocessing
import time
from functools import partial
import matplotlib
matplotlib.use('macosx')
#matplotlib.use('agg')
import numpy as np
import fitstools
import matplotlib.pyplot as plt
import balltracking.balltrack as blt
import balltracking.mballtrack as mblt
from skimage.feature import peak_local_max
DTYPE = np.float32


datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
data = fitstools.fitsread(datafile, tslice=0).astype(np.int32).astype(DTYPE)
surface = mblt.prep_data(data)
mag_thresh=50
ballspacing = 10
intsteps = 30
nt = 100

# Get a BT instance with default parameters
mbt_p = mblt.MBT(data.shape, nt=nt, rs=2, am=1, dp=0.3, td=5, ballspacing=ballspacing, intsteps=intsteps, mag_thresh=mag_thresh, polarity=1, datafiles=datafile)
#mbt.track_all_frames_debug()
mbt_p.track_all_frames()


# Visualization
i = 50
data = fitstools.fitsread(datafile, tslice=i).astype(np.int32).astype(DTYPE)
pos_p = mbt_p.ballpos[...,i]

plt.figure(figsize=(13, 10))
plt.imshow(data, cmap='gray', vmin=-100, vmax=100)
plt.colorbar()

mask_maxi = data > mag_thresh
ymaxi, xmaxi = np.array(peak_local_max(data, indices=True, footprint=np.ones((mbt_p.ballspacing, mbt_p.ballspacing)), labels=mask_maxi)).T
plt.scatter(xmaxi, ymaxi, s=12, facecolors='none', edgecolors='green')


maskp = pos_p[0, :] > 0
plt.scatter(pos_p[0, maskp], pos_p[1, maskp],
            marker='.', s=12, color='red')


# plt.text(x + 1, y + 1, '%d' % target, fontsize=8)
# plt.axis([0, data.shape[1], 0, data.shape[0]])
plt.axis([0, 512, 0, 512])
plt.xlabel('Lambert cyl. X')
plt.ylabel('Lambert cyl. Y')
plt.tight_layout()


plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mballtrack/frame_%02d.png' % i)
plt.close()
