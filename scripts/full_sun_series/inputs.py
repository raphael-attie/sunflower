import os
from pathlib import Path
from collections import OrderedDict
import numpy as np
import balltracking.balltrack as blt
# the multiprocessing start method can only bet set once
use_multiprocessing = True

run_balltracking = True

seriesname = 'JSOC_postel_x0_y0'
# For time series presented as one 3D fits file, path must be read as a string.
datafiles = sorted(list(Path(os.environ['DATA'], 'HMI', 'full_sun',
                             '2023_11_17', seriesname).glob('*.fits')))
outputdir = Path(os.environ['DATA'], 'HMI', 'full_sun', '2023_11_17', seriesname, 'balltracking')
# Balltracking parameters
bt_params = {
    'rs': 2,  # Ball radius
    'intsteps': 3,  # Number of integration steps between images
    'ballspacing': 1,  # Minimum spacing between balls
    'am': 0.3,  # Characteristic acceleration
    'dp': 0.2,  # Characteristic depth of floatation
    'sigma_factor': 1.0,  # The resulting standard deviation of the image intensity will be equal to that number.
    'fourier_radius': 4,  # Width of the high-pass Fourier filter, in the Fourier domain (k-space).
    # Time range (start, end) of the series of images to use for tracking.
    'trange': (0, 320),
    'verbose': True
}
# Flow series averaging
# Time average in nb of frames
navg = 80  # 30 min minimum with HMI @45s cadence
# Time step in number of frames between averaged flow maps. Use dt < navg for having smoother transitions
dt = navg//2
# Lanes parameters
# Nb of integration steps
nsteps = 40

##########################
# Calibration parameters
##########################
run_calibration = True
# If the drift images do not exist yet, have them created
make_drift_images = False
# Run balltracking (True) or re-use balltracked positions from a previous run?
reprocess_bt = True
# Set the vector of offset velocities (drift rates), define them independently the x and y direction
vx_rates = np.arange(-0.2, 0.21, 0.04)
vy_rates = np.zeros(len(vx_rates))
# Set the middle one to zero, for having a non-drifted flow
vx_rates[int(len(vx_rates) / 2)] = 0

cal_args = {
    # Drift rates
    'vx_rates': vx_rates,
    'vy_rates': vy_rates,
    # Which files of the time series to process in the calibration. It will determine which images to drift.
    'trange': [0, 80],
    # FWHM for the spatial gaussian smooth
    'fwhm': 7,
    # in-memory series of images to load
    'images': None,
    # Output directory for the calibration results, can be different from the main balltracking results.
    'outputdir_cal': Path(outputdir, 'hmi_drifted')
}

cal_opt_args = {
    # Velocity component(s) where the drift is applied. Can be 'x', 'y' or 'xy' for both.
    'component': 'x',
    # Smoothing kernel: 'gaussian', 'boxcar', or 'both'
    'kernel': 'gaussian',
    # Set whether we read images from disk (True) or use `images` in-memory
    'read_drift_images': True,
    ######
    # Provide the drift parameters. Even if the drifted images are read from disk, must provide what was used for the
    # calibration fit.
    ######
    # Save the arrays of ball positions to disk?
    'save_ballpos_list': True,
    'verbose': True,
    # number of cpus to use for parallelization
    'ncpus': 11
}


