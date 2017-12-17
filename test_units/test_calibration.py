# TODO: Test calibration procedure on a single tracking

from importlib import reload
import matplotlib
#matplotlib.use('macosx')
import numpy as np
import balltracking.balltrack as blt
import fitstools
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import cm

plt.ioff()



datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'

# Load a series of nt images
nt = 80
images = fitstools.fitsread(datafile, tslice=slice(0,nt)).astype(np.float32)
header = fitstools.fitsheader(datafile)
dims = [header['ZNAXIS1'], header['ZNAXIS2']]
# Set npts drift rates
npts = 10
rotation_rates = np.linspace(-0.2, 0.2, npts)
# Make the series of drifting image for 1 rotation rate
drift_images = blt.drift_series(images, (rotation_rates[0], 0))

### Ball parameters
# Use 80 frames (1 hr)
nt = 80
# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2
# Setup BT objects for forward and backward tracking.
bt_tf = blt.BT(dims, nt, rs, dp, sigma_factor=sigma_factor, mode='top', direction='forward', data=drift_images)
bt_tb = blt.BT(dims, nt, rs, dp, sigma_factor=sigma_factor, mode='top', direction='backward', data=drift_images)
# Track
_=blt.track_all_frames(bt_tf)
_=blt.track_all_frames(bt_tb)

ballpos = np.concatenate((bt_tf.ballpos, bt_tb.ballpos), axis=1)

# Get flow maps from tracked positions
trange = np.arange(0, nt)
fwhm = 15
vx, vy, wplane = blt.make_velocity_from_tracks(ballpos, dims, trange, fwhm)