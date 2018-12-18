import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import balltracking.balltrack as blt
import fitstools
from datetime import datetime


# input data
datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/mtrack_20100901_120034_TAI_20100902_120034_TAI_LambertCylindrical_continuum.fits'
# output directory for the drifting images
outputdir = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/calibration/'
### Ball parameters
# Use 80 frames (1 hr)
nframes = 80
# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2

# Load the nt images
images = fitstools.fitsread(datafile, tslice=slice(0,nframes)).astype(np.float32)

## Calibration parameters
# Set npts drift rates
npts = 10
vx_rates = np.linspace(-0.2, 0.2, npts)
drift_rates = np.stack((vx_rates, np.zeros(npts)),axis=1).tolist()
trange = np.arange(nframes)

# Smoothing
fwhm = 15
dims = images.shape[0:2]

fov_slices = [np.s_[4:-4, 30:-30],]

if __name__ == '__main__':

    startTime = datetime.now()

    cal = blt.Calibrator(images, drift_rates, nframes, rs, dp, sigma_factor, fwhm, outputdir, use_existing=True, nthreads=4)

    ballpos_top_list, ballpos_bottom_list = cal.balltrack_all_rates()

    print("\nProcessing time: %s seconds\n" % (datetime.now() - startTime))

    ## Get flow maps from tracked positions

    trange = [0, nframes]
    fwhm = 15

    xrates = np.array(drift_rates)[:,0]
    a_top, vxfit_top, vxmeans_top = blt.fit_calibration(ballpos_top_list, xrates, trange, fwhm, images.shape[0:2], fov_slices)
    a_bottom, vxfit_bottom, vxmeans_bottom = blt.fit_calibration(ballpos_bottom_list, xrates, trange, fwhm, images.shape[0:2], fov_slices)

    plt.figure(0)
    plt.plot(vxmeans_top, vx_rates, 'r.', label='data top', zorder=3)
    plt.plot(vxmeans_bottom, vx_rates, 'g+', label='data bottom', zorder=3)
    plt.plot(vxmeans_top, vxfit_top, 'b-', label=r'$\alpha_t$ =%0.2f' %a_top, zorder=2)
    plt.plot(vxmeans_bottom, vxfit_bottom, 'k-', label=r'$\alpha_b$ =%0.2f' % a_bottom, zorder=2)

    plt.xlabel('Balltracked <Vx> (px/frame)')
    plt.ylabel('Drift <Vx> (px/frame)')
    plt.grid('on')
    plt.legend()

    plt.savefig(os.path.join(outputdir,'calibration.png'))




