import os, glob
import numpy as np
import balltracking.balltrack as blt
from functools import partial
from collections import OrderedDict
import time
import pandas as pd
import fitstools

if __name__ == '__main__':

    # directory hosting the data
    drift_dir = os.path.join(os.environ['DATA'], 'sanity_check/stein_series/')
    # output directory for the drifting images
    outputdir = os.path.join(drift_dir, 'calibration')
    nframes = 60
    trange = [0, nframes]
    # load image data
    image_files = sorted(glob.glob(os.path.join(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits')))[0:nframes]

    # Ball parameters
    bt_params = OrderedDict({'rs': 2,
                             'intsteps': 5,
                             'ballspacing': 2,
                             'dp': 0.25,
                             'sigma_factor': 1.5,
                             'fourier_radius': 1.0,
                             'nframes': nframes,
                             'index':'single'})

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

    images = fitstools.fitsread(image_files, cube=False)
    # Velocity smoothing
    # fwhm = 7
    # kernel = 'boxcar'
    cal_dicts = []
    for fwhm in [7,9,11,13,15]:
        for kernel in ['boxcar', 'gaussian']:
            print(f'fwhm = {fwhm} ; kernel = {kernel}')
            bt_params['index'] = f'{kernel}_fwhm{fwhm}'

            trim = int(vx_rates.max() * nframes + fwhm + 2)
            fov_slices = np.s_[trim:imsize - trim, trim:imsize - trim]

            calibrate_partial = partial(blt.balltrack_calibration,
                                        images=images,
                                        drift_rates=drift_rates,
                                        trange=trange,
                                        fov_slices=fov_slices,
                                        reprocess_bt=True,
                                        outputdir=outputdir,
                                        kernel=kernel,
                                        fwhm=fwhm,
                                        dims=dims,
                                        basename='im_shifted',
                                        save_ballpos_list=True,
                                        nthreads=6)

            cal_dict = calibrate_partial(bt_params)
            cal_dicts.append(cal_dict)

    df = pd.DataFrame(cal_dicts)
    df.to_csv(os.path.join(outputdir, 'calibration_fwhm_kernel.csv'))





