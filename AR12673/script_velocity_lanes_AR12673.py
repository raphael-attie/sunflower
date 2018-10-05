import os
from importlib import reload
import matplotlib
matplotlib.use('macosx')
import numpy as np
import balltracking.balltrack as blt
import fitstools
from datetime import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/python_balltracking'
# Go from 1st frame at Sep 1st 00:00:00 till ~ Sep 3 18:00:00
nframes = int((3600*24*2 + 18*3600)/45) # 5280 frames
#nframes = int(80 * 5)

# Load the series
image = fitstools.fitsread(datafile, tslice=0).astype(np.float32)
dims = image.shape
### Ball parameters

# Multiplier to the standard deviation.
sigma_factor = 2
# Smoothing for Euler maps
fwhm = 15
# Calibration factors
# cal_top = 1.41
# cal_bottom = 1.30
# These new values change the speed within the error bar but this potentially reduces systematic tracking error.
cal_top = 1.49
cal_bottom = 1.35

### time windows for the euler map
tavg = 160
tstep = 80

### Lanes parameters
nsteps = 50
maxstep = 4


ballpos_top = np.load(os.path.join(outputdir,'ballpos_top.npy'))
ballpos_bottom = np.load(os.path.join(outputdir, 'ballpos_bottom.npy'))

startTime = datetime.now()

blt.make_euler_velocity_lanes(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, nframes, tavg, tstep, fwhm, nsteps, maxstep, outputdir)

total_time = datetime.now() - startTime
print("total time: %s" %total_time)





