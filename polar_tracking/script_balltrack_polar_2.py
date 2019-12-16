import balltracking.balltrack as blt
import os, glob
import numpy as np
from collections import OrderedDict
import pandas as pd

if __name__ == '__main__':

    # Longitudes (Stonyhurst) at starting frame
    # lons = [-7, -67, -77]
    # Longitudes (Carrington):
    lonCRs = [60, 0, 350]
    # Latitudes
    lats = [0, 60, 70]
    lon_lats = [(60.4, 0), (60.4, 60), (60.4, 70), (0, 0), (350, 0)]
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
    # Ball parameters
    bt_params_l = [OrderedDict({'rs': 2,
                             'intsteps': 5,
                             'ballspacing': 2,
                             'dp': 0.25,
                             'sigma_factor': 1.5,
                             'fourier_radius': 1.0,
                             'nframes': nframes,
                             'datafiles': datacubefiles[i],
                             'outputdir': outputdirs_l[i],
                             'ncores': 4,
                             'verbose':False}) for i in range(nevents)]

    for idx in range(3, nevents):
        # idx = 0
        print('Tracking datacube: ', datacubefiles[idx])
        if reprocess_bt:
            b_top, b_bot = blt.balltrack_all(**bt_params_l[idx])
        else:
            b_top = np.load(os.path.join(bt_params_l[idx]['outputdir'], 'ballpos_top.npy'))
            b_bot = np.load(os.path.join(bt_params_l[idx]['outputdir'], 'ballpos_bottom.npy'))

        # Make velocity flows and lanes
        # Get calibration factors. For extreme longitudes, use the factors of the extreme latitudes
        # because Vx is not resolved in extreme longitudes and they are in high latitudes.
        idx2 = idx
        if idx == 3:
            idx2 = 1
        if idx == 4:
            idx2 = 2
        df = pd.read_csv(os.path.join(bt_params_l[idx2]['outputdir'], 'calibration/param_sweep_0.csv'))
        cal_top = df['p_top_0'].values[0]
        cal_bot = df['p_bot_0'].values[0]
        dims = [512,512]
        fwhm = 11
        # Time ranges
        tavgs = range(40,nframes+1,40)
        tranges = [[0, tavg] for tavg in tavgs]
        ### Lanes parameters
        nsteps = 50
        maxstep = 4

        vxl, vyl, lanesl = blt.make_euler_velocity_lanes(b_top, b_bot, cal_top, cal_bot, dims, tranges,
                                      fwhm, nsteps, maxstep, bt_params_l[idx]['outputdir'], kernel='gaussian')
