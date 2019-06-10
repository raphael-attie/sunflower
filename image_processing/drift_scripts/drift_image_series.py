"""
This script drift image series for the calibration of flow tracking algorithms like Balltracking, FLCT, DeepVel, ...
Will shift images using different drift rates, interpolating in Fourier space and write FITS files in
separate directories, one for each drift rate value.
"""
import os, glob
import numpy as np
import fitstools
import balltracking.balltrack as blt
# input data, list of files
# glob.glob does not order numbered files by defaultl, the order is as appeared in the file system.
datafiles = sorted(glob.glob('/Users/rattie/Data/Ben/SteinSDO/SDO_int*.fits'))
# output directory for the drifting images
outputdir = '/Users/rattie/Data/Ben/SteinSDO/calibration3/unfiltered/'
# factor of upscaling from HMI resolution to original resolution
upfac = 1008 / 263


# nthreads is the number of parallel threads to use to parallelize the creation of the drifts series.
# Use nthreads = 1 in case of errors. Keep [nthreads <= # of cores - 2] for best efficiency,
# also it cannot be to high due to I/O limitation (depending on your SSD or HHD writing capacities)
nthreads = 4
# Number of frames to consider
nframes = 80 # ~1 hour
selected_files = datafiles[0:nframes]
# Load them
images = fitstools.fitsread(selected_files)
# If dimensions are odd, must make even for the fast fourier transform of numpy.
if images.shape[0] % 2 == 0 or images.shape[1] % 2:

    new_xsize = images.shape[1] + images.shape[1] % 2
    new_ysize = images.shape[0] + images.shape[0] % 2

    images2 = np.zeros([new_ysize, new_xsize, images.shape[2]])
    images2[0:images.shape[0], 0:images.shape[1], :] = images.copy()
    # Pad last row and column with the mean value instead of zeros to minimize tracking discontinuity with Balltracking
    images2[images.shape[0], :] = images.mean()
    images2[:, images.shape[1]] = images.mean()
else:
    # This is probably passed as a numpy view instead of a copy.
    images2 = images
## Set the drift rates.
# Step size between drifts.
# 0.02 px/frame was the value for images of size 263 x 263. Multiply by upscale factor as needed
dv = 0.02 * upfac
# Drifts rates maximum was 0.2 px/frame on 263 x 263 images @ ~368 km/px
vx_rates = np.arange(-0.2*upfac, 0.2*upfac + dv, dv)
# Enforce a strictly zero value for the drift at the middle of the series, otherwise round-offs from above propagate as a non-zero value
vx_rates[int(len(vx_rates) / 2)] = 0
ndrifts = len(vx_rates)
# Drift on x-axis only, i.e., zero on the y-axis.
drift_rates = np.stack((vx_rates, np.zeros(ndrifts)), axis=1).tolist()
# Define the sub-directories where each drift series will be saved
subdirs = [os.path.join(outputdir, 'drift_{:01d}'.format(i)) for i in range(ndrifts)]
# Instantiate the calibrator object.
cal = blt.Calibrator(images2, drift_rates, nframes, outputdir=outputdir,
                     output_prep_data=False, use_existing=False, tracking=False, normalization=False,
                     filter_function=None, subdirs=subdirs,
                     nthreads=nthreads)
# Run the drifts
cal.drift_all_rates()

print('done!')