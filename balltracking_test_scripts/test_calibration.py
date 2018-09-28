import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import balltracking.balltrack as blt
import fitstools
from datetime import datetime



datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'

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
rotation_rates = np.stack((vx_rates, np.zeros(npts)),axis=1).tolist()
trange = np.arange(nframes)

# Smoothing
fwhm = 15
dims = images.shape[0:2]

if __name__ == '__main__':

    startTime = datetime.now()

    # pool = multiprocessing.Pool(processes=4)
    # # Use partial to give process_calibration_series() the constant input "samples"
    # process_calibration_partial = partial(process_calibration_series, samples=images)
    # ballpos_top_list, ballpos_bottom_list = zip(*pool.map(process_calibration_partial, rotation_rates))

    #btop, bbot = blt.process_calibration_series(rotation_rates[0], nframes, rs, dp, sigma_factor, images)
    # ballpos = btop
    # vx, vy, wplane = blt.make_velocity_from_tracks(ballpos, dims, trange, fwhm)

    ballpos_top_list, ballpos_bottom_list = blt.loop_calibration_series(rotation_rates, images, nframes, rs, dp, sigma_factor,
                                                                        nthreads=5)

    print("\nProcessing time: %s seconds\n" % (datetime.now() - startTime))

    ## Get flow maps from tracked positions

    trange = [0, nframes]
    fwhm = 15

    xrates = np.array(rotation_rates)[:,0]
    a_top, vxfit_top, vxmeans_top = blt.fit_calibration(ballpos_top_list, xrates, trange, fwhm, images.shape[0:2])
    a_bottom, vxfit_bottom, vxmeans_bottom = blt.fit_calibration(ballpos_bottom_list, xrates, trange, fwhm, images.shape[0:2])

    plt.figure(0)
    plt.plot(vxmeans_top, vx_rates, 'r.', label='data top', zorder=3)
    plt.plot(vxmeans_bottom, vx_rates, 'g+', label='data bottom', zorder=3)
    plt.plot(vxmeans_top, vxfit_top, 'b-', label=r'$\alpha_t$ =%0.2f' %a_top, zorder=2)
    plt.plot(vxmeans_bottom, vxfit_bottom, 'k-', label=r'$\alpha_b$ =%0.2f' % a_bottom, zorder=2)

    plt.xlabel('Balltracked <Vx> (px/frame)')
    plt.ylabel('Drift <Vx> (px/frame)')
    plt.grid('on')
    plt.legend()
    plt.show()

    #plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/calibration/calibration_12673_sequential.png')




