import os, glob
import numpy as np
import balltracking.balltrack as blt
from functools import partial
from collections import OrderedDict
import fitstools

def my_arange(start, stop, step):
    """
    Convenience function, workaround for float truncation error propagating within numpy arange
    """
    return step * np.arange(start / step, stop / step)


if __name__ == '__main__':



    # output directory for the drifting images
    outputdir = os.path.join(os.environ['DATA'], 'sanity_check/stein_series/calibration_unit_test')
    # Select the index (ARGO batch job id) to test.
    job_id = 3021

    reprocess_bt = True
    nframes = 30
    # load image data
    image_files = sorted(glob.glob(os.path.join(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits')))[0:nframes]

    trange = [0, nframes]
    # Ball parameters
    bt_params = OrderedDict({'rs': 2})
    # Parameter sweep
    intsteps = [3, 4, 5]
    ballspacing = [1, 2]
    dp_l = my_arange(0.2, 0.31, 0.01)  # [0.2, 0.25, 0.3, 0.35, 0.4]
    sigma_factor_l = [1.0, 1.25, 1.5, 1.75, 2]
    # Fourier filter radius
    f_radius_l = np.arange(0, 10)
    bt_params_list = blt.get_bt_params_list(bt_params,
                                            ('intsteps', 'ballspacing', 'dp', 'sigma_factor', 'fourier_radius'),
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

    images = fitstools.fitsread(image_files, cube=False)

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
                                save_ballpos_list=True,
                                verbose=True)


    _ = calibrate_partial(bt_params_list[job_id])




