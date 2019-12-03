from importlib import reload
import matplotlib
matplotlib.use('macosx')
import numpy as np
import balltracking.balltrack as blt
import fitstools


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
bt = blt.BT(nt, rs, dp, sigma_factor=sigma_factor, mode='top', direction='forward', datafiles=datafile)

bt.initialize()
n = 0

if bt.direction == 'forward':
    if bt.data is None:
        image = fitstools.fitsread(bt.datafiles, tslice=n).astype(np.float32)
    else:
        image = bt.data[:, :, n]
elif bt.direction == 'backward':
    if bt.data is None:
        image = fitstools.fitsread(bt.datafiles, tslice=bt.nt - 1 - n).astype(np.float32)
    else:
        image = bt.data[:, :, bt.nt - 1 - n]

surface = blt.prep_data(image, bt.mean, bt.sigma, sigma_factor=bt.sigma_factor)

if bt.mode == 'bottom':
    surface = -surface

# The current position "pos" and velocity "vel" are attributes of bt.
# They are integrated in place.
for _ in range(bt.intsteps):
    blt.integrate_motion(bt, surface)


