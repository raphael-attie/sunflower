import balltracking.balltrack as blt
import os, glob
import numpy as np
from collections import OrderedDict
import pandas as pd
from pathlib import PurePath
import fitstools

if __name__ == '__main__':

    fitsfiles = sorted(glob.glob(
        '/Users/rattie/Data/Ben/DeepVelU_AR_Moat_Flows/MURaM_AR_at_SDO_resolution/SDO_ic1_*.fits'))[0:30]
    data = fitstools.fitsread(fitsfiles)
    data = data[0:256, 0:256, :]
    # Get the intensity files
    outputdir = '/Users/rattie/Data/Ben/DeepVelU_AR_Moat_Flows/MURaM_AR_at_SDO_resolution/balltrack'

    # Balltracking parameters
    nframes = data.shape[-1]
    # Ball parameters
    bt_params_l = OrderedDict({'rs': 2,
                             'intsteps': 5,
                             'ballspacing': 2,
                             'dp': 0.25,
                             'sigma_factor': 1.5,
                             'fourier_radius': 1.0,
                             'nframes': nframes,
                             'data': data,
                             'outputdir': outputdir,
                             'ncores': 4,
                             'verbose':False})


        # Make velocity flows and lanes
        # Get calibration factors. For extreme longitudes, use the factors of the extreme latitudes
        # because Vx is not resolved in extreme longitudes and they are in high latitudes.

    b_top, b_bot = blt.balltrack_all(**bt_params_l)
    df = pd.read_csv(os.path.join(outputdir, 'calibration_disk_center.csv'))
    cal_top = df['p_top_0'].values[0]
    cal_bot = df['p_bot_0'].values[0]
    dims = data.shape[0:2]
    fwhm = 7
    # Time ranges
    tranges = [[0, nframes]]
    ### Lanes parameters
    nsteps = 50
    maxstep = 1

    vxl, vyl, lanesl = blt.make_euler_velocity_lanes(b_top, b_bot, cal_top, cal_bot, dims, tranges,
                                  fwhm, nsteps, maxstep, bt_params_l['outputdir'], kernel='gaussian')
