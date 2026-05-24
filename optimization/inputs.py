import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import OrderedDict
import numpy as np
import balltracking.balltrack as blt
# the multiprocessing start method can only bet set once
use_multiprocessing = False # os.getenv('USE_MULTIPROCESSING', '0').lower().strip() in ('1', 'true', 'yes', 'y', 'on')
# number of cpus to use for parallelization
default_cpus = os.cpu_count() if use_multiprocessing else 1
ncpus = int(os.getenv('MAX_CPUS', default_cpus))# 32
# multiprocessing.set_start_method('spawn')
# TODO: check directory content to not overwrite files that will have the same index

if 'DATA' not in os.environ:
    print("ERROR: The 'DATA' environment variable is not set. Please set it to the root directory containing your FITS files before running the script.", file=sys.stderr)
    sys.exit(1)

# directory for the input data files
inputdir = Path(os.environ['DATA'], 'SteinSDO/')
datafiles = sorted(list(inputdir.glob('SDO_int*.fits')))

# directory for the balltracking results
outputdir = Path(os.environ['DATA']) / 'SteinSDO' / 'correlations'
# Run balltracking (True) or re-use balltracked positions from a previous run?
run_balltracking = True

# TODO: See if we can order Pandas rows so that the index do not depend anymore on the order the grid search
# Create the gridded list for the parameter sweep
bt_params = OrderedDict({
    'rs': 2,    # Ball radius
    'intsteps': [3, 4, 5],   # Number of integration steps between images
    'ballspacing': 1,  # Minimum spacing between balls
    'am': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # Characteristic acceleration
    'dp': [0.1, 0.15, 0.2, 0.25, 0.3],    # Characteristic depth of floatation
    'sigma_factor': [1.0, 1.25, 1.5, 1.75, 2],  # The target standard deviation of the Z-height of the output data surface
    'fourier_radius': [0, 1, 2, 3, 4, 5],  # Width of high-pass Fourier filter (k-space). Adapt to instruments, image resolution, ...
    'trange': (0, 60),  # Time range (1st index, last index+1) of the series of images to use for tracking.
    'verbose': True
})
bt_params_list = blt.get_bt_params_list(bt_params)

################################
# Flow maps parameters
################################
maps_params = {
    'generate_lanes': True,  # Toggle creation of the supergranular maps
    'navg': 40,  # in nb of frame ~ must translate to ~30 min minimum with HMI @45s cadence
    'dt': 20,  # Time step in number of frames between averaged flow maps. Use dt < navg for having smoother transitions
    'nsteps': 40,  # Nb of integration steps for the supergranular boundary mapping
    'kernel': 'gaussian',  # Smoothing kernel: 'gaussian', 'boxcar', or 'both'
    'fwhm': 7,   # spatial gaussian smooth of the Euler dense flow maps
}

##########################
# Calibration parameters
##########################
# The calibration gives the velocity magnitude multiplication factors from a linear fit
# on rigidly drifting images at known rates.
# It is necessary for any new set of data and/or new set of input parameters.
# Note that the multiplication factors for the top-side tracking and bottom-side tracking are different,
# which why it is necessary to run that calibration even if you are not analyzing the velocities in physical units.
# The calibration can be long to run, depending on the data volume.

# If the drift images do not exist yet, create them. True will overwrite if they already exist
make_drift_images = True
# Set the vector of offset velocities (drift rates), define them independently the x and y direction
vx_rates = np.arange(-0.2, 0.21, 0.04)
# Set the middle one to zero, for having a non-drifted flow (optional, but encouraged)
vx_rates[int(len(vx_rates) / 2)] = 0
# vy_rates typically set to zeros, but calibration can be tested on both axes at the same time
vy_rates = np.zeros(len(vx_rates))
# Velocity scale to convert from px/frame interval to m/s
v_scale = 368000 / 60 

# Positional arguments passed to blt.Calibrator()
cal_args = {
    'trange': bt_params['trange'],  # [first, last[ Indices of images to drift and track in the time series
    'vx_rates': vx_rates,  # Drift rates x-axis
    'vy_rates': vy_rates,  # Drift rates y-axis
    'fwhm': maps_params['fwhm'],   # for the spatial gaussian smooth during the calibration
    'images': None,  # in-memory series of images. If None, read directly from disk (more ram-friendly)
    'outputdir_cal': outputdir  # can be different from the balltracking output dir.
}

# Optional arguments passed to blt.Calibrator()
cal_opt_args = {
    'component': 'x',  # Velocity component(s) where the drift is applied. Can be 'x', 'y' or 'xy' for both.
    'kernel': maps_params['kernel'],  # Smoothing kernel: 'gaussian', 'boxcar', or 'both'
    'save_ballpos_list': True,  # Save the arrays of ball positions to disk?
    'verbose': True,
    'ncpus': 1  # number of cpus to use for parallelization over the drift rates, <= len(vx_rates).
}
