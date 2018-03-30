from importlib import reload
import matplotlib
matplotlib.use('macosx')
import numpy as np
import balltracking.balltrack as blt
import fitstools
from datetime import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt

# def balltrack_parallel()

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'

# Use 80 frames (1 hr)
nframes = 80
# Load the series
images = fitstools.fitsread(datafile, tslice=slice(0,nframes)).astype(np.float32)

### Ball parameters

# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2
# Smoothing for Euler maps
fwhm = 15
# Calibration factors
a_top = 1.41
a_bottom = 1.30

### time ranges for the euler map
trange = np.arange(0, nframes)

### Lanes parameters
nsteps = 30
maxstep = 4

# Get a BT instance with the above parameters
bt_tf = blt.BT(images.shape[0:2], nframes, rs, dp, sigma_factor=sigma_factor, mode='top', direction='forward', data=images)
bt_tb = blt.BT(images.shape[0:2], nframes, rs, dp, sigma_factor=sigma_factor, mode='top', direction='backward', data=images)
bt_bf = blt.BT(images.shape[0:2], nframes, rs, dp, sigma_factor=sigma_factor, mode='bottom', direction='forward', data=images)
bt_bb = blt.BT(images.shape[0:2], nframes, rs, dp, sigma_factor=sigma_factor, mode='bottom', direction='backward', data=images)



if __name__ == '__main__':

    startTime = datetime.now()

    with Pool(processes=4) as pool:
        bt_tf, bt_tb, bt_bf, bt_bb = pool.map(blt.track_all_frames, [bt_tf, bt_tb, bt_bf, bt_bb])

    ballpos_top = np.concatenate((bt_tf.ballpos, bt_tb.ballpos), axis=1)
    ballpos_bottom = np.concatenate((bt_bf.ballpos, bt_bb.ballpos), axis=1)

    vx_top, vy_top, wplane_top = blt.make_velocity_from_tracks(ballpos_top, images.shape[0:2], trange, fwhm)
    vx_bottom, vy_bottom, wplane_top = blt.make_velocity_from_tracks(ballpos_bottom, images.shape[0:2], trange, fwhm)

    vx_top *= a_top
    vy_top *= a_top
    vx_bottom *= a_bottom
    vy_bottom *= a_bottom

    vx = 0.5*(vx_top + vx_bottom)
    vy = 0.5*(vy_top + vy_bottom)

    print(" --- %s seconds" % (datetime.now() - startTime))

    #lanes_top = blt.make_lanes(vx_top, vy_top, nsteps, maxstep)
    #lanes_bottom = blt.make_lanes(vx_bottom, vy_bottom, nsteps, maxstep)
    lanes = blt.make_lanes(vx, vy, nsteps, maxstep)


    # plt.figure(0)
    # plt.imshow(lanes_top, cmap='gray_r', origin='lower')
    # plt.title('Lanes from top')
    # plt.tight_layout()
    # plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/lanes/lanes_top.png')
    #
    # plt.imshow(lanes_bottom, cmap='gray_r', origin='lower')
    # plt.title('Lanes from bottom')
    # plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/lanes/lanes_bottom.png')
    #
    # plt.imshow(lanes, cmap='gray_r', origin='lower')
    # plt.title('Lanes from top-bottom average')
    # plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/lanes/lanes_top_bottom.png')





