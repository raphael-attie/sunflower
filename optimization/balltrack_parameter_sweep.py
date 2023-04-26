import glob
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import balltracking.balltrack as blt
import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial
from collections import OrderedDict
from pathlib import Path
import fitstools

if __name__ == '__main__':
    # the multiprocessing start method can only bet set once
    multiprocessing.set_start_method('spawn')
    # output directory
    # TODO: check directory content to not overwrite files that will have the same index
    outputdir = Path(os.environ['DATA3'], 'sanity_check/stein_series/calibration3')
    outputdir.mkdir(parents=True, exist_ok=True)
    reprocess_bt = True
    nframes = 60
    trange = [0, nframes]
    imfiles = sorted(glob.glob(os.path.join(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits')))[0:nframes]
    images = fitstools.fitsread(imfiles)
    # Ball parameters
    # TODO: See if we can order Pandas rows so that the index do not depend anymore on the order the grid search
    bt_params = OrderedDict({'rs': 2})
    # Parameter sweep
    intsteps = [3, 4, 5, 6]
    ballspacing = [1, 2]
    am_l = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dp_l = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    sigma_factor_l = [1.0, 1.25, 1.5, 1.75, 2]
    # Fourier filter radius
    f_radius_l = np.arange(0, 5)
    bt_params_list = blt.get_bt_params_list(bt_params,
                                            ('intsteps', 'ballspacing', 'am', 'dp', 'sigma_factor', 'fourier_radius'),
                                            (intsteps, ballspacing, am_l, dp_l, sigma_factor_l, f_radius_l))
    bt_params_list = [params for params in bt_params_list if ((params['am'] < 0.5 or
                                                              params['dp'] < 0.2))]
    bt_params_list = bt_params_list[0:10]
    ##########################################
    ## Calibration parameters
    # Set npts drift rates
    dv = 0.04
    vx_rates = np.arange(-0.2, 0.21, dv)
    vx_rates[int(len(vx_rates) / 2)] = 0
    vx_labels = ['vx_{:02d}'.format(i) for i in range(len(vx_rates))]
    drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()


    fwhm = 7
    imsize = 263  # actual size is 264 but original size was 263 then padded to 264 to make it even for the Fourier transform
    trim = int(vx_rates.max() * nframes + fwhm + 2)
    fov_slices = np.s_[trim:imsize - trim, trim:imsize - trim]
    dims = [imsize, imsize]
    # Prepare partial function for parallel pool & map, which can accept only the list of parameters as its argument
    calibrate_partial = partial(blt.full_calibration,
                                images=images,
                                drift_rates=drift_rates,
                                trange=trange,
                                fov_slices=fov_slices,
                                reprocess_bt=reprocess_bt,
                                reprocess_fit=True,
                                outputdir=outputdir,
                                fwhm=fwhm,
                                dims=dims,
                                outputdir2=outputdir,
                                save_ballpos_list=True,
                                verbose=True,
                                nthreads=1)

    with Pool(processes=1) as pool:
        pool.map(calibrate_partial, bt_params_list)

# At the end of this parallel job, use "parameter_sweep_velocity_calibration.py" to aggregate everything

