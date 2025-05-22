import os
from pathlib import Path
import numpy as np

use_multiprocessing = False

# Set whether to run balltracking and calibration
# If False, will go through either just the calibration and/or the Euler Flow map creation, assuming
# Balltracking and calibration have been run before and output files available at the expected location (see outputdir)
run_balltracking = True
run_calibration = True
# Paths to FITS files (replace with whatever applies to you).
# Time series presented as one 3D fits "cube" is supported. Make sure to sort your files even if you have consistent
# names as 'glob' does not do it by default. (print them out to make sure you have the in the right oder).
datafiles = '/Users/rattie/data/Mashael/intensity_30frames.fits'
# Output directory for the Balltracking algorithm
outputdir = Path('/Users/rattie/data/Mashael/balltracking_simulation')

##########################
# Balltracking parameters
##########################

bt_params = {
    'rs': 6,  # Ball radius
    'intsteps': 4,  # Number of integration steps between images
    'ballspacing': 3,  # Minimum spacing between balls
    'am': 0.3,  # Characteristic acceleration
    'dp': 0.2,  # Characteristic depth of floatation
    'sigma_factor': 1.0,  # The resulting standard deviation of the image intensity will be equal to that number.
    'fourier_radius': 4,  # Width of high-pass Fourier filter (k-space). Adapt to instruments, image resolution, ...
    'trange': (0, 29),  # Time range (1st index, last index + 1) of the series of images to use for tracking.
    'verbose': True
}

################################
# Flow maps parameters
################################
maps_params = {
    'generate_lanes': True,  # Toggle creation of the supergranular maps
    'im_dims': [1024, 1024],  # Image dimension [width, height] in pixels
    'navg': 14,  # in nb of frame ~ must translate to ~30 min minimum with HMI @45s cadence
    'dt': 5,  # Time step in number of frames between averaged flow maps. Use dt < navg for having smoother transitions
    'nsteps': 40,  # Nb of integration steps for the supergranular boundary mapping
    'kernel': 'gaussian',  # Smoothing kernel: 'gaussian', 'boxcar', or 'both'
    'fwhm': 21,   # spatial gaussian smooth of the Euler dense flow maps
    'hdu_n': 0,  # index of the header in the FITS header data unit. Often 0, but 1 for RICE-compressed from JSOC
    'use_headers': True,
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

# If the drift images do not exist yet, create them
make_drift_images = True
# Set the vector of offset velocities (drift rates), define them independently the x and y direction
vx_rates = np.arange(-0.2, 0.21, 0.04)
# Set the middle one to zero, for having a non-drifted flow (optional, but encouraged)
vx_rates[int(len(vx_rates) / 2)] = 0
# vy_rates typically set to zeros, but calibration can be tested on both axes at the same time
vy_rates = np.zeros(len(vx_rates))

cal_args = {
    'vx_rates': vx_rates,  # Drift rates x-axis
    'vy_rates': vy_rates,  # Drift rates y-axis
    'trange': [0, 29],  # Indices of images to drift.
    'fwhm': maps_params['fwhm'],   # for the spatial gaussian smooth during the calibration
    'images': None,  # in-memory series of images. If None, read directly from disk (more ram-friendly)
    'outputdir_cal': Path(outputdir, 'hmi_drifted')  # can be different from the balltracking output dir.
}

cal_opt_args = {
    'component': 'x',  # Velocity component(s) where the drift is applied. Can be 'x', 'y' or 'xy' for both.
    'kernel': maps_params['kernel'],  # Smoothing kernel: 'gaussian', 'boxcar', or 'both'
    'read_drift_images': True,  # Set whether we read images from disk (True) or use `images` in-memory
    'save_ballpos_list': True,  # Save the arrays of ball positions to disk?
    'verbose': True,
    'ncpus': 1  # # number of cpus to use for parallelization over the drift rates, <= len(vx_rates).
}

