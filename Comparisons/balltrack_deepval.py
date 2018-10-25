import glob
import numpy as np
import balltracking.balltrack as blt
import fitstools

datafiles = sorted(glob.glob('/Users/rattie/Data/Ben/SteinSDO/SDO_int*.fits'))
outputdir = '/Users/rattie/Data/Ben/SteinSDO/balltracking/'

### Balltrack parameters

# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2
# Duration (nb of frames)
nframes = 80
# Select only that subset of nframes files
selected_files = datafiles[0:nframes]
# Load the nt images
images = fitstools.fitsread(selected_files)
# Must make even dimensions for the fast fourier transform
images2 = np.zeros([264, 264, images.shape[2]])
images2[0:263, 0:263, :] = images.copy()
images2[263, :] = images.mean()
images2[:, 263] = images.mean()

ballpos_top, ballpos_bottom = blt.balltrack_all(nframes, rs, dp, sigma_factor, data=images2, outputdir=outputdir, ncores=4)

# Velocity parameters
# Calibration factors
cal_top = 1.60
cal_bottom = 1.44
# Get dimension from one slice of the series
dims = images2.shape[0:2]
### Lanes parameters
nsteps = 50
maxstep = 4

# Time average and spatial smoothing size
tavg = 80
fwhm = 15
tstep = tavg/2

_ = blt.make_euler_velocity_lanes(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, nframes, tavg, tstep, fwhm, nsteps, maxstep, outputdir)


# Time average and spatial smoothing size
tavg = 30
fwhm = 7
tstep = tavg/2

_ = blt.make_euler_velocity_lanes(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, nframes, tavg, tstep, fwhm, nsteps, maxstep, outputdir)
