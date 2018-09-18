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
intsteps = 30
nt = 15
# Get a BT instance with default parameters
mbt = mblt.MBT(data.shape, nt=nt, intsteps=intsteps, mag_thresh=mag_thresh, datafiles=datafile)

#mbt.initialize()
mbt.track_all_frames()
#startpos = np.array([mbt.xstart, mbt.ystart, mbt.zstart])

# x,y = 155, 338
# target = mblt.get_balls_at(x, y, mbt.xstart, mbt.ystart)

# pos = []
# vel = []
# force = []
# pos.append(startpos)
# vel.append(mbt.vel)
# force.append(mbt.force)
# for i in range(mbt.intsteps):
#     posi, veli, forcei = blt.integrate_motion(mbt, surface, return_copies=True)
#     pos.append(posi)
#     vel.append(veli)
#     force.append(forcei)

data = fitstools.fitsread(datafile, tslice=0).astype(np.int32).astype(DTYPE)
surface = mblt.prep_data(data)
mask_maxi = data >= mbt.mag_thresh
ymaxi, xmaxi = np.array(peak_local_max(data, indices=True, footprint=np.ones((5, 5)), labels=mask_maxi)).T

plt.figure(0, figsize=(13,9))
plt.imshow(data, cmap='gray', vmin=-100, vmax=100)
plt.colorbar()
plt.scatter(xmaxi, ymaxi,s=12, facecolors='none', edgecolors='green')
plt.scatter(mbt.xstart, mbt.ystart, marker='.', s=12, color='red')
#plt.text(x + 1, y + 1, '%d' % target, fontsize=8)

#plt.colorbar()
plt.axis([0, data.shape[1], 0, data.shape[0]])

plt.tight_layout()

#plt.show()
plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mballtrack/frame_00.png')
plt.close()

for i in range(nt):

    data = fitstools.fitsread(datafile, tslice=i).astype(np.int32).astype(DTYPE)
    surface = mblt.prep_data(data)
    mask_maxi = data >= mbt.mag_thresh
    ymaxi, xmaxi = np.array(peak_local_max(data, indices=True, footprint=np.ones((5, 5)), labels=mask_maxi)).T

    plt.figure(1, figsize=(13,9))
    plt.imshow(data, cmap='gray', vmin=-100, vmax=100)
    plt.colorbar()
    #plt.scatter(pos[i][0,:], pos[i][1, :], c=pos[i][2,:], marker='.', s=12, cmap='jet', vmin=surface.min()-2*mbt.rs, vmax=surface.max())
    # if i == 1:
    #     plt.scatter(mbt.xstart, mbt.ystart, marker='.', s=12, color='green')
    plt.scatter(xmaxi, ymaxi, s=12, facecolors='none', edgecolors='green')
    plt.scatter(mbt.ballpos[0, :, i], mbt.ballpos[1, :, i], marker='.', s=12, color='red')


    #plt.text(x + 1, y + 1, '%d' % target, fontsize=8)

    #plt.colorbar()
    plt.axis([0, data.shape[1], 0, data.shape[0]])

    plt.tight_layout()

    #plt.show()
    plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mballtrack/frame_%02d.png'%(i+1))
    plt.close()
