import os,glob
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import balltracking.balltrack as blt
import fitstools
from datetime import datetime
import graphics

# input data, list of files
# glob.glob does not order numbered files by defaultl, the order is as appeared in the file system.
datafiles = sorted(glob.glob('/Users/rattie/Data/Ben/SteinSDO/SDO_int*.fits'))
# output directory for the drifting images
outputdir = '/Users/rattie/Data/Ben/SteinSDO/calibration/'
plotdir = os.path.join('/Users/rattie/Data/Ben/SteinSDO/plots/')
### Ball parameters
# Use 80 frames (1 hr)
nframes = 80
# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2

# Select only a subset of nframes files
selected_files = datafiles[0:nframes]
# # Write png files
# for file in selected_files:
#     graphics.fits_to_jpeg(file, '/Users/rattie/Data/Ben/SteinSDO/intensity_png/')

# Load the nt images
images = fitstools.fitsread(selected_files)
# Must make even dimensions for the fast fourier transform
images2 = np.zeros([264, 264, images.shape[2]])
images2[0:263, 0:263, :] = images.copy()
images2[263, :] = images.mean()
images2[:, 263] = images.mean()

## Calibration parameters
# Set npts drift rates
npts = 9
vx_rates = np.linspace(-0.2, 0.2, npts)
drift_rates = np.stack((vx_rates, np.zeros(npts)),axis=1).tolist()

# Smoothing
fwhm = 7
dims = images.shape[0:2]

# List of index tuples [ np_s[x_axis, y_axis], ] to average the means over different subfields of view of the flow map. Useful to mask out a sunspot
# If just one, do not forget to add a comma "," at the end.
# Here we need to exclude the region that got circularly shifted by the Fourier phase shit ~ 23 pixels in both horizontal directions (left-right)
fov_slices = [np.s_[23:263-23, 0:263],
              ]

if __name__ == '__main__':

    startTime = datetime.now()

    cal = blt.Calibrator(images2, drift_rates, nframes, rs, dp, sigma_factor, fwhm, outputdir,
                         output_prep_data=False, use_existing=True,
                         nthreads=5)

    ballpos_top_list, ballpos_bottom_list = cal.balltrack_all_rates()

    print("\nProcessing time: %s seconds\n" % (datetime.now() - startTime))

    # ## Get flow maps from tracked positions
    #
    trange = [0, nframes]

    xrates = np.array(drift_rates)[:,0]
    a_top, vxfit_top, vxmeans_top, vxs_top, vys_top = blt.fit_calibration(ballpos_top_list, xrates, trange, fwhm, images.shape[0:2], fov_slices, return_flow_maps=True)
    a_bottom, vxfit_bottom, vxmeans_bottom, vxs_bottom, vys_bottom = blt.fit_calibration(ballpos_bottom_list, xrates, trange, fwhm, images.shape[0:2], fov_slices, return_flow_maps=True)

    filename_suffix = 'tavg{:d}_fwhm{:d}'.format(nframes, fwhm)

    fitstools.writefits(vxs_top[4], os.path.join(outputdir, 'vx_top_{}.fits'.format(filename_suffix)))
    fitstools.writefits(vxs_bottom[4], os.path.join(outputdir, 'vx_bottom_{}.fits'.format(filename_suffix)))

    fitstools.writefits(vys_bottom[4], os.path.join(outputdir, 'vy_top_{}.fits'.format(filename_suffix)))
    fitstools.writefits(vys_bottom[4], os.path.join(outputdir, 'vy_bottom_{}.fits'.format(filename_suffix)))

    vx_cal = 0.5*(vxs_top[4] * a_top + vxs_bottom[4]*a_bottom)
    vy_cal = 0.5*(vys_top[4] * a_top + vys_bottom[4]*a_bottom)

    fitstools.writefits(vx_cal, os.path.join(outputdir, 'vx_{}_cal.fits'.format(filename_suffix)))
    fitstools.writefits(vy_cal, os.path.join(outputdir, 'vy_{}_cal.fits'.format(filename_suffix)))






