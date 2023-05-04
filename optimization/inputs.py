import os
from pathlib import Path
from collections import OrderedDict
import numpy as np
import balltracking.balltrack as blt
# the multiprocessing start method can only bet set once
use_multiprocessing = False
# number of cpus to use for parallelization
ncpus = 32
# multiprocessing.set_start_method('spawn')
# TODO: check directory content to not overwrite files that will have the same index
# Output directory for the balltracking results
outputdir = Path(os.environ['DATA3'], 'sanity_check/stein_series/calibration3')
# Output directory for the calibration results
outputdir_cal = outputdir
# Run balltracking (True) or re-use balltracked positions from a previous run?
reprocess_bt = True
nframes = 60
# Time range [start, end[ within the series of images to consider in the calibration.
# To take them all, just put [0, nframes]
trange = [0, nframes]

# TODO: See if we can order Pandas rows so that the index do not depend anymore on the order the grid search
# Create the gridded list for the parameter sweep
bt_params = OrderedDict({
    'rs': 2,
    'intsteps': [3, 4, 5, 6],
    'ballspacing': [1, 2],
    'am': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'dp': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
    'sigma_factor': [1.0, 1.25, 1.5, 1.75, 2],
    'fourier_radius': np.arange(0, 5)
})
bt_params_list = blt.get_bt_params_list(bt_params)
# If restricting only to a subset
# bt_params_list = [params for params in bt_params_list if ((params['am'] < 0.5 or
#
#                                                           params['dp'] < 0.2))]

# Testing another subset with one task per cpu in parallel
bt_params_list = bt_params_list[0:ncpus]

##########################
# Calibration parameters
##########################

# Read images from disk?
read_drift_images = True
# If True, must provide directories, one per drift rate.
drift_dirs = sorted(list(Path(os.environ['DATA3'], 'sanity_check/stein_series/calibration3').glob('drift*')))
# Alternatively, load the images
# imfiles = sorted(glob.glob(os.path.join(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits')))[0:nframes]
# images = fitstools.fitsread(imfiles)
# Prepare images for Ray object store - shared resource
# drifted_images = [blt.create_drift_series(images, dr) for dr in drift_rates]
# drifted_images_id = ray.put(drifted_images)
######
# Provide the drift parameters. Even if the drifted images are read from disk, must provide what was used for the
# calibration fit.
######
# Step size between each offset velocity
dv = 0.04
# Vector of offset velocities
vx_rates = np.arange(-0.2, 0.21, dv)
# Set the middle one to zero, for having an non-drifted flow
vx_rates[int(len(vx_rates) / 2)] = 0
# Stack those values, with vy at 0. Can be changed to have a drift an y-axis as well
drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()
###
# Parameters for the averaging with Lagrange to Euler conversion
###
# FWHM for the spatial gaussian smooth
fwhm = 7
# Dimensions of the input image.
dims = [263, 263]
# To avoid edge effects, set some cropping parameters, for the metrics of the flow field (correlation, rmse, etc...)
trim = int(vx_rates.max() * nframes + fwhm + 2)
fov_slices = np.s_[trim:dims[0] - trim, trim:dims[1] - trim]

