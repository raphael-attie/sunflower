import os
import matplotlib
matplotlib.use('agg')
import numpy as np
import balltracking.balltrack as blt
import fitstools
from datetime import datetime

datafile = '/Users/rattie/Data/SDO/HMI/continuum/Lat_63/mtrack_20110627_200034_TAI_20110628_000033_TAI_Postel_060.4_63.0_continuum.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/continuum/Lat_63'
nframes = 320

# Load one slice of the series
image = fitstools.fitsread(datafile, tslice=0).astype(np.float32)
dims = image.shape
# Smoothing for Euler maps
fwhm = 15
# Calibration factors
cal_top = 1.41
cal_bottom = 1.30

### time windows for the euler map
tavg = 320
tstep = 320

### Lanes parameters
nsteps = 50
maxstep = 4


ballpos_top = np.load(os.path.join(outputdir,'ballpos_top.npy'))
ballpos_bottom = np.load(os.path.join(outputdir, 'ballpos_bottom.npy'))

startTime = datetime.now()

blt.make_euler_velocity_lanes(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, nframes, tavg, tstep, fwhm, nsteps, maxstep, outputdir)

total_time = datetime.now() - startTime
print("total time: %s" %total_time)





