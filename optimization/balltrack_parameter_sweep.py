import os, glob
import numpy as np
import fitstools
import balltracking.balltrack as blt
import multiprocessing
from multiprocessing import Pool
from functools import partial
import time
import pandas as pd

nprocesses = 6

if __name__ == '__main__':

    # the multiprocessing start method can only bet set once.
    multiprocessing.set_start_method('spawn')
    # directory hosting the drifted data (9 series)
    drift_parent_dir = '/Users/rattie/Data/sanity_check/stein_series/'
    # output directory for the drifting images
    outputdir = '/Users/rattie/Data/sanity_check/stein_series/calibration/'
    reprocess_bt = True
    nframes = 40
    trange = [0, nframes]

    ### Ball parameters
    bt_params = {'rs': 2}
    # Parameter sweep
    intsteps = [3,4,5]
    ballspacing = [1, 2, 3, 4]
    dp_l = [0.1, 0.2, 0.3, 0.4, 0.5]
    sigma_factor_l = [1, 1.25, 1.5, 1.75, 2]
    ### Fourier filter radius
    f_radius_l = np.arange(0, 21)
    bt_params_list = blt.get_bt_params_list(bt_params, ('intsteps', 'ballspacing', 'dp', 'sigma_factor', 'f_radius'), (intsteps, ballspacing, dp_l, sigma_factor_l, f_radius_l))

    df = pd.DataFrame(bt_params_list)


    ### Velocity smoothing
    fwhm = 7
    kernel = 'boxcar'
    ##########################################
    ## Calibration parameters
    # Set npts drift rates
    dv = 0.04
    vx_rates = np.arange(-0.2, 0.21, dv)
    vx_rates[int(len(vx_rates) / 2)] = 0
    vx_labels = ['vx_{:02d}'.format(i) for i in range(len(vx_rates))]

    drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()
    imsize = 263  # actual size is 264 but original size was 263 then padded to 264 to make it even for the Fourier transform
    trim = int(vx_rates.max() * nframes + fwhm + 2)
    fov_slices = np.s_[trim:imsize - trim, trim:imsize - trim]

    a_top_l = []
    a_bot_l = []
    dims = [imsize, imsize]
    for bt_params in bt_params_list:
        a_top, vxfit_top, vxmeans_top, residuals_top, a_bottom, vxfit_bottom, vxmeans_bottom, residuals_bottom = \
            blt.balltrack_calibration(drift_rates, trange, fov_slices, reprocess_bt, drift_parent_dir, bt_params, kernel, fwhm, dims)

        a_top_l.append(a_top)
        a_bot_l.append(a_bottom)

    calibrate_partial = partial(blt.balltrack_calibration, drift_rates=drift_rates, trange=trange, fov_slices=fov_slices,
                                reprocess_bt=reprocess_bt, outputdir=drift_parent_dir, kernel=kernel, fwhm=fwhm, dims=dims)

    pool = Pool(processes=nprocesses)
    vxmeans_l, a_avg_l, vxfit_avg_l, residuals_l = zip(*pool.map(calibrate_partial, bt_params_list))
    pool.close()
    pool.join()

    # Save the results in a dataframe
    df['a_top'] = a_top_l
    df['a_bot'] = a_bot_l

    cal_factors = np.array([a_top_l, a_bot_l])
    np.save(os.path.join(outputdir, 'cal_factors_parameter_sweep.npy'), cal_factors)

