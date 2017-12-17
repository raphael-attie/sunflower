from importlib import reload
import numpy as np
import balltracking.balltrack as blt
import fitstools
from datetime import datetime
from multiprocessing import Pool




datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'

# Load 1st image
image = fitstools.fitsread(datafile, tslice=0).astype(np.float32)
header = fitstools.fitsheader(datafile)
### Ball parameters
# Use 80 frames (1 hr)
nt = 80
# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2
# Get a BT instance with the above parameters
bt_tf = blt.BT(datafile, image.shape, nt, rs, dp, sigma_factor=sigma_factor, mode='top', direction='forward')
bt_tb = blt.BT(datafile, image.shape, nt, rs, dp, sigma_factor=sigma_factor, mode='top', direction='backward')

if __name__ == '__main__':

    startTime = datetime.now()

    with Pool(processes=2) as pool:
        bt_tf, bt_tb = pool.map(blt.track_all_frames, [bt_tf, bt_tb])

    # Flip the time axis of the backward tracking
    # ballpos dimensions: [xyz, # balls, time]
    bt_tb.ballpos = np.flip(bt_tb.ballpos, 2)
    # Concatenate the above on balls axis. I'm simply adding the balls of the backward tracking to the forward tracking.
    # This makes a finer sampling of the data surface and reduces the number of empty bins
    ballpos_t = np.concatenate((bt_tf.ballpos, bt_tb.ballpos), axis=1)

    print(" --- %s seconds" % (datetime.now() - startTime))


