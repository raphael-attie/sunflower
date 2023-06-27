import os
from pathlib import Path
import numpy as np

### Balltrack parameters
bt_params_top = {
    'rs': 2,
    'intsteps': 3,
    'ballspacing': 1,
    'am' : 0.3,
    'dp' : 0.1,
    'sigma_factor' : 1.5,
    'fourier_radius': 2,
}

bt_params_bottom = {
    'rs': 2,
    'intsteps': 3,
    'ballspacing': 1,
    'am': 0.3,
    'dp': 0.1,
    'sigma_factor': 1.5,
    'fourier_radius': 2,
}

### Calibration parameters
# Read images from disk
images = None
read_drift_images = True
# Step size between each offset velocity
dv = 0.04
# Vector of offset velocities
vx_rates = np.arange(-0.2, 0.21, dv)
# Set the middle one to zero, for having a non-drifted flow
vx_rates[int(len(vx_rates) / 2)] = 0
# If True, must provide directories, one per drift rate.
drift_dirs = sorted(list(Path(os.environ['DATA3'], 'sanity_check/stein_series/calibration3').glob('drift*')))
# Output directory for the balltracking results
outputdir = Path(os.environ['DATA3'], 'sanity_check/stein_series/calibration3')
# Output directory for the calibration results
outputdir_cal = outputdir
# Number of frames to process for each drifted series
nframes = 60
# Time indices [start, end[ within the series of images to consider in the calibration.
# To take them all, just put [0, nframes]
trange = [0, nframes]
################################################################
# Parameters for the averaging with Lagrange to Euler conversion
#################################################################
# FWHM for the spatial gaussian smooth
fwhm = 7
# Dimensions of the input image.
dims = [263, 263]
# To avoid edge effects, set some cropping parameters, for the metrics of the flow field (correlation, rmse, etc...)
trim = int(vx_rates.max() * nframes + fwhm + 2)
fov_slices = np.s_[trim:dims[0] - trim, trim:dims[1] - trim]

# Save the arrays of ball positions to disk?
save_ballpos_list = False
# Reprocess existing calibration results?
reprocess_existing = True

verbose = False
# Multithread?
nthreads = 1

