import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import balltracking.balltrack as blt
import fitstools
from datetime import datetime


# input data
datafile = '/Users/rattie/Data/SDO/HMI/continuum/Lat_63/mtrack_20110627_200034_TAI_20110628_000033_TAI_Postel_060.4_63.0_continuum.fits'
# output directory for the drifting images
outputdir = '/Users/rattie/Data/SDO/HMI/continuum/Lat_63/'
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

fov_slices = [np.s_[10:200, 30:-30],
              np.s_[350:500, 30:-30]]

unit = 368000/45

if __name__ == '__main__':

    startTime = datetime.now()

    cal = blt.Calibrator(images, drift_rates, nframes, rs, dp, sigma_factor, outputdir, use_existing=False, output_prep_data=False, nthreads=4)

    ballpos_top_list, ballpos_bottom_list = cal.balltrack_all_rates()

    print("\nProcessing time: %s seconds\n" % (datetime.now() - startTime))

    ## Get flow maps from tracked positions

    trange = [0, nframes]
    fwhm = 15

    xrates = np.array(drift_rates)[:,0]
    a_top, vxfit_top, vxmeans_top, residuals_top = blt.fit_calibration(ballpos_top_list, xrates, trange, fwhm, images.shape[0:2], fov_slices)
    a_bottom, vxfit_bottom, vxmeans_bottom, residuals_bottom = blt.fit_calibration(ballpos_bottom_list, xrates, trange, fwhm, images.shape[0:2], fov_slices)

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

    plt.close('all')
    plt.figure(1)
    width = 150
    plt.bar(vx_rates * unit, residuals_top * unit, width = width, color='black', label='top-side tracking')
    plt.bar(vx_rates * unit + width, residuals_bottom * unit, width=width, color='gray', label='bottom-side tracking')
    plt.xlabel('Drift <Vx> (m/s)')
    plt.ylabel('Absolute residual error on <Vx> (m/s)')
    plt.ylim([0, 10])
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, 'residuals_top_bottom.png'), dpi=180)



