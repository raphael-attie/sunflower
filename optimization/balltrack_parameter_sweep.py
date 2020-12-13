import glob, os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import balltracking.balltrack as blt
from multiprocessing import Pool
from functools import partial
from collections import OrderedDict
import fitstools

if __name__  == '__main__':

    # output directory
    outputdir = os.path.join(os.environ['DATA'], 'sanity_check/stein_series/calibration2')
    reprocess_bt = True
    nframes = 60
    trange = [0, nframes]
    image_files = sorted(glob.glob(os.path.join(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits')))[0:nframes]

    # Ball parameters
    bt_params = OrderedDict({'rs': 2})
    # Parameter sweep
    intsteps = [4, 5, 6]
    ballspacing = [1, 2]
    dp_l = [0.2, 0.25, 0.3, 0.35, 0.4]
    sigma_factor_l = [1.0, 1.25, 1.5, 1.75, 2]
    # Fourier filter radius
    f_radius_l = np.arange(0, 10)
    bt_params_list = blt.get_bt_params_list(bt_params,
                                            ('intsteps', 'ballspacing', 'dp', 'sigma_factor', 'fourier_radius'),
                                            (intsteps, ballspacing, dp_l, sigma_factor_l, f_radius_l))
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

    dims = [imsize, imsize]

    images = fitstools.fitsread(image_files, cube=False)

    # Prepare partial function for parallel pool & map.
    calibrate_partial = partial(blt.balltrack_calibration,
                                images=images,
                                drift_rates=drift_rates,
                                trange=trange,
                                fov_slices=fov_slices,
                                reprocess_bt=reprocess_bt,
                                outputdir=outputdir,
                                kernel=kernel,
                                fwhm=fwhm,
                                dims=dims,
                                save_ballpos_list=False,
                                nthreads=1)

    print('Running balltracking on {:d} params lists'.format(len(bt_params_list)))
    # df = calibrate_partial(bt_params_list[0])

    #map(calibrate_partial, bt_params_list[0:6])

    # the multiprocessing start method can only bet set once.
    # multiprocessing.set_start_method('spawn')
    pool = Pool(processes=50)
    pool.map(calibrate_partial, bt_params_list)
    pool.close()
    pool.join()

# At the end of this parallel job, use "parameter_sweep_velocity_calibration.py" to aggregate everything

