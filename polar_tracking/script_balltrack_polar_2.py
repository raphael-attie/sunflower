import balltracking.balltrack as blt
import os, glob
import numpy as np
from collections import OrderedDict
import pandas as pd
from pathlib import Path

if __name__ == '__main__':

    # Longitudes (Stonyhurst) at starting frame
    # lons = [-7, -67, -77]
    # Longitudes (Carrington):
    lonCRs = [60, 0, 350]
    # Latitudes
    lats = [0, 60, 70]
    lon_lats = [(60.4, 0), (0, 0), (350, 0), (60.4, 60), (60.4, 70)]
    nevents = len(lon_lats)
    basenames_l = ['mtrack_20110627_200034_TAI_20110628_000033_TAI_Postel_{:05.1f}_{:04.1f}_continuum.fits'
                     .format(lon_lat[0], lon_lat[1]) for lon_lat in lon_lats]
    # Get the intensity files
    datacubefiles = [os.path.join(os.environ['DATA'], 'SDO/HMI/polar_study/', basename) for basename in basenames_l]
    outputdirs_l = [os.path.join(os.environ['DATA'], 'SDO/HMI/polar_study/lonCR_{:05.1f}_lat_{:04.1f}'.format(lon_lat[0], lon_lat[1]))
                    for lon_lat in lon_lats]

    # Balltracking parameters
    reprocess_bt = False
    nframes = 320
    ncores_balltrack = 4
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


    for idx in range(0,nevents):
    # idx = 0

        bt_params = {'top': bt_params_top, 'bottom': bt_params_bottom,
                     'nframes': nframes,
                     'outputdir': outputdirs_l[idx],
                     'output_prep_data': False,
                     'verbose': False}

        # Aggregate calibration files
        # These files don't really need to be ordered. This will be taken care for by sorting per index key.
        filelist = sorted(glob.glob(os.path.join(outputdirs_l[idx], 'calibration/param_sweep_*.csv')))

        if reprocess_bt:
            print('Tracking datacube: ', datacubefiles[idx])
            ballpos_top, ballpos_bottom = blt.balltrack_all(bt_params, datafiles=datacubefiles[idx], ncores=ncores_balltrack)
        else:
            with np.load(os.path.join(outputdirs_l[idx], 'ballpos.npz')) as bt_pos:
                ballpos_top = bt_pos['ballpos_top']
                ballpos_bottom = bt_pos['ballpos_bottom']

        # Make velocity flows and lanes
        # Get calibration factors. For extreme longitudes, use the factors of the extreme latitudes
        # because Vx is not resolved in extreme longitudes and they are in high latitudes.
        print('making velocity maps and lanes')
        idx2 = idx
        if idx == 1:
            idx2 = 3
        if idx == 2:
            idx2 = 4
        df_top = pd.read_csv(os.path.join(outputdirs_l[idx2], 'calibration/param_sweep_0.csv'))
        df_bot = pd.read_csv(os.path.join(outputdirs_l[idx2], 'calibration/param_sweep_1.csv'))
        cal_top = df_top['p_top_0'].values[0]
        cal_bot = df_bot['p_bot_0'].values[0]

        dims = [512, 512]
        fwhm = 15
        # Time ranges
        tavgs = range(40, nframes+1, 8)
        tranges = [[0, tavg] for tavg in tavgs]
        # Lanes parameters
        nsteps = 50
        maxstep = 4

        outputdir = Path(outputdirs_l[idx], 'widening_average')
        outputdir.mkdir(exist_ok=True)
        vxl, vyl, lanesl = blt.make_euler_velocity_lanes(
            ballpos_top, ballpos_bottom, cal_top, cal_bot, dims, tranges,
            fwhm, nsteps, maxstep, str(outputdir), kernel='gaussian')

        # Time ranges
        tstarts = np.arange(0, nframes-80, 40)
        tranges = [[t1, t1+80] for t1 in tstarts]

        outputdir = Path(outputdirs_l[idx], 'sliding_average')
        outputdir.mkdir(exist_ok=True)
        vxl, vyl, lanesl = blt.make_euler_velocity_lanes(
            ballpos_top, ballpos_bottom, cal_top, cal_bot, dims, tranges,
            fwhm, nsteps, maxstep, str(outputdir), kernel='gaussian')