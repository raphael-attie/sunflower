import balltracking.balltrack as blt
import os, glob
import numpy as np
import pandas as pd
from collections import OrderedDict
from multiprocessing import Pool


if __name__ == '__main__':

    # Get the intensity files
    datafiles = sorted(glob.glob(os.path.join(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits')))
    outputdir = os.path.join(os.environ['DATA'], 'Ben/SteinSDO/balltrack2')
    reprocess_bt = True

    # Ball parameters
    bt_params_top = OrderedDict({'rs': 2,
                                 'ballspacing': 2,
                                 'intsteps': 5,
                                 'dp': 0.25,
                                 'sigma_factor': 1.5,
                                 'fourier_radius': 1.0})

    bt_params_bottom = OrderedDict({'rs': 2,
                                    'ballspacing': 2,
                                    'intsteps': 5,
                                    'dp': 0.25,
                                    'sigma_factor': 1.5,
                                    'fourier_radius': 1.0})

    bt_params = {'top':bt_params_top, 'bottom':bt_params_bottom,
                 'nframes': 364,
                 'outputdir':outputdir,
                 'output_prep_data':False,
                 'verbose': False}


    if reprocess_bt:
        b_top, b_bot = blt.balltrack_all(bt_params, datafiles=datafiles, ncores=4)
    else:
        with np.load(os.path.join(outputdir, 'ballpos.npz')) as bt_pos:
            b_top = bt_pos['ballpos_top']
            b_bot = bt_pos['ballpos_bottom']

    # Make velocity flow fields

    caldf = pd.read_csv(os.path.expanduser('~/Data/sanity_check/stein_series/correlation_dataframe.csv'))
    row_top = caldf[(caldf.rs == bt_params_top['rs'])
                & (caldf.ballspacing==bt_params_top['ballspacing'])
                & (caldf.intsteps==bt_params_top['intsteps'])
                & np.isclose(caldf.dp, bt_params_top['dp'])
                & np.isclose(caldf.sigma_factor, bt_params_top['sigma_factor'])
                & np.isclose(caldf.fourier_radius, bt_params_top['fourier_radius'])]

    row_bottom = caldf[(caldf.rs == bt_params_bottom['rs'])
                    & (caldf.ballspacing == bt_params_bottom['ballspacing'])
                    & (caldf.intsteps == bt_params_bottom['intsteps'])
                    & np.isclose(caldf.dp, bt_params_bottom['dp'])
                    & np.isclose(caldf.sigma_factor, bt_params_bottom['sigma_factor'])
                    & np.isclose(caldf.fourier_radius, bt_params_bottom['fourier_radius'])]


    cal_top = row_top.p_top_0.values[0]
    cal_bot = row_bottom.p_bot_0.values[0]
    print('Using calibration factors of (top / bottom):')
    print(cal_top)
    print(cal_bot)


    im_dims = [263, 263]
    for fwhm in [7,9,11,13,15]:
        for kernel in ['boxcar', 'gaussian']:

            tranges = [[0, nt] for nt in range(30, bt_params['nframes']+1, 5)]
            dims=[263, 263]

            def make_velocity_wrapper(trange):
                vx_t, vy_t = blt.make_euler_velocity(b_top, b_bot, cal_top, cal_bot, im_dims, trange, fwhm, kernel)
                npzfile = os.path.join(bt_params['outputdir'], 'vxy_{:s}_fwhm_{:d}_avg_{:d}.npz'.format(kernel, fwhm, trange[1]))
                print('saving ', npzfile)
                np.savez_compressed(npzfile,vx=vx_t, vy=vy_t)
                return vx_t, vy_t

            with Pool(processes = 6) as pool:
                vx, vy = zip(*pool.map(make_velocity_wrapper, tranges))
