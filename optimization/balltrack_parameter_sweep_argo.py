import os
import numpy as np
import balltracking.balltrack as blt
from functools import partial
from collections import OrderedDict
import time

if __name__ == '__main__':

    os.chdir(os.path.expanduser('~/dev/sdo-tracking-framework'))
    # directory hosting the drifted data (9 series)
    drift_dir = os.path.join(os.environ['DATA'], 'sanity_check/stein_series/')
    # output directory for the drifting images
    outputdir = os.path.join(drift_dir, 'calibration')
    reprocess_bt = True
    nframes = 30
    trange = [0, nframes]
    # Ball parameters
    bt_params = OrderedDict({'rs': 2})
    # Parameter sweep
    intsteps = [3,4,5]
    ballspacing = [1, 2, 3, 4]
    dp_l = [0.2, 0.25, 0.3, 0.35, 0.4]
    sigma_factor_l = [1, 1.25, 1.5, 1.75, 2]
    # Fourier filter radius
    f_radius_l = np.arange(0, 16)
    bt_params_list = blt.get_bt_params_list(bt_params, ('intsteps', 'ballspacing', 'dp', 'sigma_factor', 'f_radius'),
                                            (intsteps, ballspacing, dp_l, sigma_factor_l, f_radius_l))
    # Velocity smoothing
    fwhm = 7
    kernel = 'boxcar'
    ##########################################
    # Calibration parameters
    # Set npts drift rates
    dv = 0.04
    vx_rates = np.arange(-0.2, 0.21, dv)
    vx_rates[int(len(vx_rates) / 2)] = 0
    vx_labels = ['vx_{:02d}'.format(i) for i in range(len(vx_rates))]

    drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()
    imsize = 263
    dims = [imsize, imsize]
    trim = int(vx_rates.max() * nframes + fwhm + 2)
    fov_slices = np.s_[trim:imsize - trim, trim:imsize - trim]

    calibrate_partial = partial(blt.balltrack_calibration,
                                drift_rates=drift_rates,
                                trange=trange,
                                fov_slices=fov_slices,
                                reprocess_bt=reprocess_bt,
                                drift_dir=drift_dir,
                                outputdir=outputdir,
                                kernel=kernel,
                                fwhm=fwhm,
                                dims=dims,
                                basename='im_shifted',
                                write_ballpos_list=False)

    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    calibrate_partial(bt_params_list[job_id])





