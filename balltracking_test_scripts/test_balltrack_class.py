import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import balltracking.balltrack as blt
import fitstools
from datetime import datetime



datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/test_balltrack_class'
### Ball parameters
# Use 80 frames (1 hr)
nframes = 80
# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2

# Load the nt images
images = fitstools.fitsread(datafile, tslice=slice(0,nframes)).astype(np.float32)

## Calibration parameters
# Set npts drift rates
npts = 10
vx_rates = np.linspace(-0.2, 0.2, npts)
rotation_rates = np.stack((vx_rates, np.zeros(npts)),axis=1).tolist()
trange = np.arange(nframes)

# Smoothing
fwhm = 15
dims = images.shape[0:2]

ballpos_top, ballpos_bottom = blt.balltrack_all(nframes, rs, dp, sigma_factor, datafile, outputdir, ncores=1)

ballpos = ballpos_top
trange = [0, nframes]
vx, vy, _ = blt.make_velocity_from_tracks(ballpos_top, dims, trange, fwhm)
