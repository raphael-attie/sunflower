import numpy as np
import balltracking.balltrack as blt
import fitstools
from datetime import datetime
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

def process_calibration_series(rotation_rate, samples):

    # number of frames to process
    nt = samples.shape[2]
    # Ball radius
    rs = 2
    # depth factor
    dp = 0.2
    # Multiplier to the standard deviation.
    sigma_factor = 2

    # Make the series of drifting image for 1 rotation rate
    drift_images = blt.drift_series(samples, rotation_rate)
    # Balltrack forward and backward
    ballpos_top, _, _ = blt.balltrack_all(nt, rs, dp, sigma_factor=sigma_factor, mode='top', data=drift_images)
    ballpos_bottom, _, _ = blt.balltrack_all(nt, rs, dp, sigma_factor=sigma_factor, mode='bottom', data=drift_images)

    return ballpos_top, ballpos_bottom

# def parallel_runs(rotation_rate_list, samples):
#
#     pool = multiprocessing.Pool(processes=4)
#
#     # Use partial to give process_calibration_series() the constant input "samples"
#     process_calibration_partial = partial(process_calibration_series, samples=samples)
#     result_list = pool.map(process_calibration_partial, rotation_rate_list)
#
#     return result_list

def fit_calibration(ballpos_list, trange, fwhm):

    vxs, vys, wplanes = zip(*[blt.make_velocity_from_tracks(ballpos, images.shape[0:2], trange, fwhm) for ballpos in ballpos_list])
    # Select an ROI that contains valid data. At least one should exclude edges as wide as the ball radius.
    # This one also excludes the sunspot in the middle. Beware of bias due to differential rotation!

    vxmeans1 = np.array([vx[10:200, 30:-30].mean() for vx in vxs])
    vxmeans2 = np.array([vx[350:500, 30:-30].mean() for vx in vxs])
    vxmeans = 0.5*(vxmeans1 + vxmeans2)

    p = np.polyfit(vx_rates, vxmeans, 1)
    a = 1 / p[0]
    vxfit = a * (vxmeans - p[1])
    return a, vxfit, vxmeans

def calibrate_parallel(images, rotation_rates, nthreads=2):
    pool = multiprocessing.Pool(processes=nthreads)
    # Use partial to give process_calibration_series() the constant input "samples"
    process_calibration_partial = partial(process_calibration_series, samples=images)
    ballpos_top_list, ballpos_bottom_list = zip(*pool.map(process_calibration_partial, rotation_rates))
    return ballpos_top_list, ballpos_bottom_list

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'

### Ball parameters
# Use 80 frames (1 hr)
nframes = 80

# Load the nt images
images = fitstools.fitsread(datafile, tslice=slice(0,nframes)).astype(np.float32)

## Calibration parameters
# Set npts drift rates
npts = 10
vx_rates = np.linspace(-0.2, 0.2, npts)
rotation_rates = np.stack((vx_rates, np.zeros(npts)),axis=1).tolist()


if __name__ == '__main__':

    startTime = datetime.now()

    # pool = multiprocessing.Pool(processes=4)
    # # Use partial to give process_calibration_series() the constant input "samples"
    # process_calibration_partial = partial(process_calibration_series, samples=images)
    # ballpos_top_list, ballpos_bottom_list = zip(*pool.map(process_calibration_partial, rotation_rates))


    ballpos_top_list, ballpos_bottom_list = calibrate_parallel(images, rotation_rates, nthreads=4)

    print("\nProcessing time: %s seconds\n" % (datetime.now() - startTime))

    ## Get flow maps from tracked positions

    trange = np.arange(0, nframes)
    fwhm = 15

    a_top, vxfit_top, vxmeans_top = fit_calibration(ballpos_top_list, trange, fwhm)
    a_bottom, vxfit_bottom, vxmeans_bottom = fit_calibration(ballpos_bottom_list, trange, fwhm)

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

    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/calibration/calibration.png')




