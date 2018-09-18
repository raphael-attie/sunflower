"""
Test magnetic balltracking for convergeance to the local minima within one single frame
"""

from importlib import reload
import matplotlib
matplotlib.use('macosx')
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
intsteps = 15
nt = 15
# Get a BT instance with default parameters
mbt = mblt.MBT(data.shape, nt=nt, rs=2, am=0.5, dp=0.5, td=1, intsteps=intsteps, mag_thresh=mag_thresh, datafiles=datafile)
#mbt.track_all_frames_debug()
mbt.initialize()
startpos = np.array([mbt.xstart, mbt.ystart, mbt.zstart])

pos = []
vel = []
force = []
pos.append(startpos)
vel.append(mbt.vel)
force.append(mbt.force)
for i in range(mbt.intsteps):
    posi, veli, forcei = blt.integrate_motion(mbt, surface, return_copies=True)
    pos.append(posi)
    vel.append(veli)
    force.append(forcei)

data = fitstools.fitsread(datafile, tslice=0).astype(np.int32).astype(DTYPE)
surface = mblt.prep_data(data)
mask_maxi = data >= mbt.mag_thresh
ymaxi, xmaxi = np.array(peak_local_max(data, indices=True, footprint=np.ones((5, 5)), labels=mask_maxi)).T

## Visualization
for i in range(mbt.intsteps+1):

    plt.figure(1, figsize=(13,10))
    plt.imshow(data, cmap='gray', vmin=-100, vmax=100)
    plt.colorbar()
    plt.scatter(xmaxi, ymaxi, s=12, facecolors='none', edgecolors='green')
    plt.scatter(pos[i][0, :], pos[i][1, :], marker='.', s=12, color='red')
    #plt.text(x + 1, y + 1, '%d' % target, fontsize=8)
    plt.axis([0, data.shape[1], 0, data.shape[0]])
    plt.tight_layout()

    plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mballtrack/frame_%02d.png'%i)
    plt.close()


