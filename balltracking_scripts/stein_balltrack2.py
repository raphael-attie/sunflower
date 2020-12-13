import balltracking.balltrack as blt
import os, glob
import numpy as np
import pandas as pd
from collections import OrderedDict
from multiprocessing import Pool


if __name__ == '__main__':
    """ Script generating Stein balltracked data from best parameters seen in 
    comparisons/DeepVelU_FLCT_Balltracking.ipynb """

    # Get the intensity files
    datafiles = sorted(glob.glob(os.path.join(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits')))
    outputdir = os.path.join(os.environ['DATA'], 'Ben/SteinSDO/balltrack3')
    reprocess_bt = False
    ncores_balltrack = 4
    ncores_makevel = 36

    # Ball parameters
    bt_params_top = OrderedDict({'rs': 2,
                                 'ballspacing': 1,
                                 'intsteps': 6,
                                 'dp': 0.3,
                                 'sigma_factor': 2.0,
                                 'fourier_radius': 1.0})

    bt_params_bottom = OrderedDict({'rs': 2,
                                    'ballspacing': 2,
                                    'intsteps': 4,
                                    'dp': 0.20,
                                    'sigma_factor': 2.0,
                                    'fourier_radius': 1.0})

    bt_params = {'top': bt_params_top, 'bottom': bt_params_bottom,
                 'nframes': 364,
                 'outputdir': outputdir,
                 'output_prep_data': False,
                 'verbose': False}


    if reprocess_bt:
        ballpos_top, ballpos_bottom = blt.balltrack_all(bt_params, datafiles=datafiles, ncores=ncores_balltrack)
    else:
        with np.load(os.path.join(outputdir, 'ballpos.npz')) as bt_pos:
            ballpos_top = bt_pos['ballpos_top']
            ballpos_bottom = bt_pos['ballpos_bottom']

    # Make velocity flow fields

    caldf = pd.read_csv(os.path.join(os.environ['DATA'], 'sanity_check/stein_series/correlation_dataframe2.csv'))
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
    print('Using calibration factors of (top / bottom ):')
    print(cal_top)
    print(cal_bot)
    # The time ranges should exclude the one used for the calibration to avoid overfitting and underestimating the error
    dims = [263, 263]
    tranges = [[60, nt] for nt in range(60+30, bt_params['nframes'] + 1, 5)]

    for fwhm in [7, 9, 11, 13, 15]:
        print(f'fwhm = {fwhm}')
        kernel = 'boxcar'

        def make_euler_velocity_wrapper(trange):

            vx_top, vy_top, _ = blt.make_velocity_from_tracks(ballpos_top, dims, trange, fwhm, kernel)
            vx_bottom, vy_bottom, _ = blt.make_velocity_from_tracks(ballpos_bottom, dims, trange, fwhm, kernel)

            vx_top_cal = vx_top * cal_top
            vy_top_cal = vy_top * cal_top
            vx_bot_cal = vx_bottom * cal_bot
            vy_bot_cal = vy_bottom * cal_bot

            vx_cal = 0.5 * (vx_top_cal + vx_bot_cal)
            vy_cal = 0.5 * (vy_top_cal + vy_bot_cal)
            npzfile = os.path.join(bt_params['outputdir'], 'vxy_{:s}_fwhm_{:d}_avg_{:d}.npz'.format(kernel, fwhm, trange[1]))
            print('saving ', npzfile)
            np.savez_compressed(npzfile,vx_cal=vx_cal, vy_cal=vy_cal,
                                vx_top_uncal=vx_top, vy_top_uncal=vy_top,
                                vx_bot_uncal=vx_bottom, vy_bot_uncal=vy_bottom)

            return None

        with Pool(processes=ncores_makevel) as pool:
            pool.map(make_euler_velocity_wrapper, tranges)
