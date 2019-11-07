import os, glob
import numpy as np
import fitstools
import balltracking.balltrack as blt
import multiprocessing
from multiprocessing import Pool
from functools import partial
import time

def balltrack_calibration(drift_rates, filter_radius, ballspacing, fwhm, intsteps, fov_slices, kernel, reprocess_bt, use_existing, outputdir):
    # Load the nt images
    dims = [264, 264]
    start_time = time.time()

    if reprocess_bt:
        cal = blt.Calibrator(None, drift_rates, nframes, rs, dp, sigma_factor,
                             filter_radius=filter_radius, ballspacing=ballspacing,
                             outputdir=outputdir,
                             intsteps=intsteps,
                             output_prep_data=False, use_existing=use_existing,
                             nthreads=5)

        ballpos_top_list, ballpos_bottom_list = cal.balltrack_all_rates()

        print(start_time - time.time())
    else:
        print('Load existing tracked data at all rates')
        ballpos_top_list = np.load(os.path.join(outputdir, 'ballpos_top_list.npy'))
        ballpos_bottom_list = np.load(os.path.join(outputdir, 'ballpos_bottom_list.npy'))


    trange = [0, nframes]
    xrates = np.array(drift_rates)[:, 0]
    a_top, vxfit_top, vxmeans_top, residuals_top = blt.fit_calibration(ballpos_top_list, xrates, trange, fwhm,
                                                                       dims, fov_slices, kernel,
                                                                       return_flow_maps=False)
    a_bottom, vxfit_bottom, vxmeans_bottom, residuals_bottom = blt.fit_calibration(ballpos_bottom_list, xrates, trange,
                                                                                   fwhm,
                                                                                   dims, fov_slices, kernel,
                                                                                   return_flow_maps=False)

    return a_top, vxfit_top, vxmeans_top, residuals_top, a_bottom, vxfit_bottom, vxmeans_bottom, residuals_bottom


if __name__ == '__main__':


    # the multiprocessing start method can only bet set once.
    multiprocessing.set_start_method('spawn')
    # directory hosting the drifted data (9 series)
    drift_parent_dir = '/Users/rattie/Data/Ben/SteinSDO/calibration/unfiltered/'
    # output directory for the drifting images
    outputdir = '/Users/rattie/Data/Ben/SteinSDO/optimization_fourier_radius_3px_finer_grid/'

    reprocess_bt = True
    use_existing = True
    ### Ball parameters
    # Use 80 frames (1 hr)
    nframes = 80
    # Ball radius
    rs = 2
    # Get series of all other input parameters
    dp = 0.3
    # TODO: Look again at the correlation matrix and see where the calibration should be attempted again.

    sigma_factor = 1.5
    intsteps = 5
    # Space between balls on initial grid
    ballspacing = 3
    ### Fourier filter radius
    f_radius_l = np.arange(0, 21)
    ### Velocity smoothing
    fwhm = 7
    kernel = 'boxcar'
    ##########################################
    ## Calibration parameters
    # Set npts drift rates
    npts = 9
    vx_rates = np.linspace(-0.2, 0.2, npts)
    drift_rates = np.stack((vx_rates, np.zeros(npts)), axis=1).tolist()
    imsize = 263  # actual size is 264 but original size was 263 then padded to 264 to make it even for the Fourier transform
    trim = int(vx_rates.max() * nframes + fwhm + 2)
    # The balltracking fov needs to be in a list, even for just 1 element.
    fov_slices = np.s_[trim:imsize - trim, trim:imsize - trim]

    a_top_l = []
    a_bot_l = []
    for filter_radius in f_radius_l:
        a_top, vxfit_top, vxmeans_top, residuals_top, a_bottom, vxfit_bottom, vxmeans_bottom, residuals_bottom = \
            balltrack_calibration(drift_rates, filter_radius, ballspacing, fwhm, intsteps, fov_slices, kernel, reprocess_bt, use_existing, drift_parent_dir)

        print('Calibration for filter_radius = ', filter_radius)
        print('a_top = ', a_top)
        print('a_bottom = ', a_bottom)
        print(' ')
        a_top_l.append(a_top)
        a_bot_l.append(a_bottom)

    cal_factors = np.array([a_top_l, a_bot_l])
    np.save(os.path.join(outputdir, 'cal_factors_sigma_factor{:1.2f}_{:s}.npy'.format(sigma_factor, kernel)), cal_factors)

