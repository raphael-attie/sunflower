import os, glob
import numpy as np
from scipy.io import readsav
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
import fitstools
import balltracking.balltrack as blt

# input data, list of files
# glob.glob does not order numbered files by defaultl, the order is as appeared in the file system.
datafiles = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/mtrack_20100901_120034_TAI_20100902_120034_TAI_LambertCylindrical_continuum.fits'
# output directory for the drifting images
outputdir = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/calibration/'
### Ball parameters
# Use 80 frames (1 hr)
nframes = 80
# Ball radius
rs = 2
# depth factor
dp = 0.3
# Multiplier to the standard deviation.
sigma_factor = 1  # 1#2
# Select only a subset of nframes files
images = fitstools.fitsread(datafiles, tslice=slice(0, nframes))


def filter_function(image):
    fimage = blt.filter_image(image)
    fimage2 = fimage + 500
    return fimage2


npts = 9
vx_rates = np.linspace(-0.2, 0.2, npts)
drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()

subdirs = [os.path.join(outputdir, 'drift_no_filtering_{:01d}'.format(i)) for i in range(len(drift_rates))]
cal = blt.Calibrator(images, drift_rates, nframes, rs, dp, sigma_factor, outputdir,
                     output_prep_data=False, use_existing=False, tracking=False, normalization=False,
                     filter_function=None, subdirs=subdirs,
                     nthreads=5)

cal.drift_all_rates()

subdirs_filtered = [os.path.join(outputdir, 'drift_filt_non_norm_{:01d}'.format(i)) for i in range(len(drift_rates))]
cal_filtered = blt.Calibrator(images, drift_rates, nframes, rs, dp, sigma_factor, outputdir,
                     output_prep_data=False, use_existing=False, tracking=False, normalization=False,
                     filter_function=filter_function, subdirs=subdirs_filtered,
                     nthreads=5)

cal_filtered.drift_all_rates()
