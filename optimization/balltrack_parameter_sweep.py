import os
import balltracking.balltrack as blt
import numpy as np
import multiprocessing
from functools import partial
from collections import OrderedDict
from pathlib import Path
from time import time
# import ray
# from ray.util.multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

if __name__ == '__main__':
    # the multiprocessing start method can only bet set once
    multiprocessing.set_start_method('spawn')
    # ray.init(num_cpus=32)
    # TODO: check directory content to not overwrite files that will have the same index
    outputdir = Path(os.environ['DATA3'], 'sanity_check/stein_series/calibration3')
    outputdir.mkdir(parents=True, exist_ok=True)
    reprocess_bt = True
    nframes = 60
    trange = [0, nframes]
    # imfiles = sorted(glob.glob(os.path.join(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits')))[0:nframes]
    # images = fitstools.fitsread(imfiles)
    drift_dirs = sorted(list(Path(os.environ['DATA3'], 'sanity_check/stein_series/calibration3').glob('drift*')))
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
    # bt_params_list = [params for params in bt_params_list if ((params['am'] < 0.5 or
    #                                                           params['dp'] < 0.2))]
    bt_params_list = bt_params_list[0:32]
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

    # Prepare images for Ray object store - shared resource
    # drifted_images = [blt.create_drift_series(images, dr) for dr in drift_rates]
    # drifted_images_id = ray.put(drifted_images)

    # Prepare partial function for parallel pool & map, which can accept only the list of parameters as its argument
    calibrate_partial = partial(blt.full_calibration,
                                drift_rates=drift_rates,
                                drift_dirs=drift_dirs,
                                read_drift_images=True,
                                trange=trange,
                                fov_slices=fov_slices,
                                reprocess_bt=reprocess_bt,
                                reprocess_fit=True,
                                outputdir=outputdir,
                                fwhm=fwhm,
                                dims=dims,
                                outputdir2=outputdir,
                                save_ballpos_list=False,
                                verbose=False,
                                nthreads=1)

    # with Pool(processes=32) as pool:
    #     pool.map(calibrate_partial, bt_params_list)

    start = time()
    with ProcessPoolExecutor(max_workers=32) as executor:
        for idx in executor.map(calibrate_partial, bt_params_list):
            print(f'Processed index {idx}')
    end = time()
    print('Elapsed time: ', (end - start)/60)
    
# At the end of this parallel job, use "parameter_sweep_velocity_calibration.py" to aggregate everything

