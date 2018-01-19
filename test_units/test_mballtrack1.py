from importlib import reload
import matplotlib
matplotlib.use('macosx')
import numpy as np
import fitstools
import matplotlib.pyplot as plt
import balltracking.balltrack as blt
import balltracking.mballtrack as mblt

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
data = fitstools.fitsread(datafile, tslice=0).astype(np.float32)

mag_thresh=20
ypos0, xpos0 = np.where(data > mag_thresh)

# Get a BT instance with default parameters
mbt = mblt.MBT(data.shape, nt=20, datafiles=datafile)

## Test initialization
#mbt.initialize()
## Test tracking positive flux

mbt.track_all_frames()

plt.figure(0, figsize=(10,10))
plt.imshow(data, vmin=-100, vmax=100, cmap='gray')
plt.plot(mbt.xstart, mbt.ystart, ls='none', marker='.', color='red', markerfacecolor='none')
plt.plot(mbt.pos[0, mbt.new_valid_balls_mask], mbt.pos[1, mbt.new_valid_balls_mask], ls='none', marker='+', color='red', markerfacecolor='none')
plt.tight_layout()
