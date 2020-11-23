import balltracking.balltrack as blt
import os
import numpy as np
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path
from functools import partial

if __name__ == '__main__':

    ibisdir = Path(os.environ['DATA'], 'Ben', 'IBIS', 'white_light')
    # Get the intensity files
    datafiles = str(ibisdir.joinpath('ibis.wl.speckle.cropped.fits'))
    outputdir = Path(ibisdir, 'balltrack')
    reprocess_bt = False
    ncores_balltrack = 4

    # Ball parameters
    bt_params_top = OrderedDict({'rs': 8,
                                 'ballspacing': 8,
                                 'intsteps': 5,
                                 'dp': 0.25,
                                 'sigma_factor': 1.0,
                                 'fourier_radius': 10.0})

    bt_params_bottom = bt_params_top

    bt_params = {'top': bt_params_top, 'bottom': bt_params_bottom,
                 'nframes': 832,
                 'outputdir': outputdir,
                 'output_prep_data': False,
                 'verbose': True}

    if reprocess_bt:
        ballpos_top, ballpos_bottom = blt.balltrack_all(bt_params, datafiles=datafiles, ncores=ncores_balltrack)
    else:
        with np.load(os.path.join(outputdir, 'ballpos.npz')) as bt_pos:
            ballpos_top = bt_pos['ballpos_top']
            ballpos_bottom = bt_pos['ballpos_bottom']

    # Make velocity flow fields
    ncores_make_velocity = 10
    cal_top = 1.534
    cal_bot = 1.511

    dims = [920, 920]

    tranges = [[start, start+150] for start in range(0, bt_params['nframes'] - 150, 75)]


    def make_euler_velocity_wrapper(trange, fwhm, kernel):

        vx_top, vy_top, _ = blt.make_velocity_from_tracks(ballpos_top, dims, trange, fwhm, kernel)
        vx_bottom, vy_bottom, _ = blt.make_velocity_from_tracks(ballpos_bottom, dims, trange, fwhm, kernel)

        vx_top_cal = vx_top * cal_top
        vy_top_cal = vy_top * cal_top
        vx_bot_cal = vx_bottom * cal_bot
        vy_bot_cal = vy_bottom * cal_bot

        vx_cal = 0.5 * (vx_top_cal + vx_bot_cal)
        vy_cal = 0.5 * (vy_top_cal + vy_bot_cal)

        lanes = blt.make_lanes(vx_cal, vy_cal, nsteps=40, maxstep=4)

        npzfile = Path(bt_params['outputdir'], f'vxy_{kernel}_fwhm_{fwhm}_trange_{trange[0]:03d}_{trange[1]:03d}.npz')
        print('saving ', npzfile)
        np.savez_compressed(npzfile, vx_cal=vx_cal, vy_cal=vy_cal, lanes=lanes,
                            vx_top_uncal=vx_top, vy_top_uncal=vy_top,
                            vx_bot_uncal=vx_bottom, vy_bot_uncal=vy_bottom)

        return npzfile

    kernel = 'boxcar'

    for fwhm in [37, 74]:

        make_vel = partial(make_euler_velocity_wrapper, fwhm=fwhm,
                           kernel=kernel)

        with Pool(processes=ncores_make_velocity) as pool:
            vfiles = pool.map(make_vel, tranges)
