import os
import numpy as np
import balltracking.balltrack as blt
import fitstools
from datetime import datetime


datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/mtrack_20100901_120034_TAI_20100902_120034_TAI_LambertCylindrical_continuum.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/python_balltracking/'
# Go from 1st frame at Sep 1st 00:00:00 till ~ Sep 3 18:00:00
nframes = 1920 # 5280 frames
#nframes = int(80 * 5)

# Load one slice of the series
image = fitstools.fitsread(datafile, tslice=0).astype(np.float32)
dims = image.shape
### Ball parameters

# Multiplier to the standard deviation.
sigma_factor = 2
# Smoothing for Euler maps
fwhm = 15
# Calibration factors
cal_top = 1.41
cal_bottom = 1.31

### time windows for the euler map
tavg = 80
tstep = 40

### Lanes parameters
nsteps = 20
maxstep = 4


ballpos_top = np.load(os.path.join(outputdir,'ballpos_top.npy'))
ballpos_bottom = np.load(os.path.join(outputdir, 'ballpos_bottom.npy'))

startTime = datetime.now()

blt.make_euler_velocity_lanes(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, nframes, tavg, tstep, fwhm, nsteps, maxstep, outputdir)


total_time = datetime.now() - startTime
print("total time: %s" %total_time)





