import balltracking.balltrack as blt
import os, glob
import numpy as np
from collections import OrderedDict
from multiprocessing import Pool

if __name__ == '__main__':

    # Get the intensity files
    datafiles = sorted(glob.glob(os.path.join(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits')))
    reprocess_bt = False

    # Ball parameters
    bt_params = OrderedDict({'rs': 2,
                             'intsteps': 5,
                             'ballspacing': 2,
                             'dp': 0.25,
                             'sigma_factor': 1.5,
                             'fourier_radius': 1.0,
                             'nframes': 364,
                             'datafiles': datafiles,
                             'outputdir': os.path.join(os.environ['DATA'], 'Ben/SteinSDO/balltrack'),
                             'ncores': 4})


    if reprocess_bt:
        b_top, b_bot = blt.balltrack_all(verbose=True, **bt_params)
    else:
        b_top = np.load(os.path.join(bt_params['outputdir'], 'ballpos_top.npy'))
        b_bot = np.load(os.path.join(bt_params['outputdir'], 'ballpos_bottom.npy'))

    # # Make velocity flow fields
    kernel = 'boxcar'
    fwhm = 11
    tranges = [[0, nt] for nt in range(30, bt_params['nframes']+1, 5)]
    cal_top = 1.801
    cal_bot = 1.713
    dims=[263, 263]

    def make_velocity_wrapper(trange):
        vx_t, vy_t = blt.make_euler_velocity(b_top, b_bot, cal_top, cal_bot, [263, 263], trange, fwhm, kernel)
        np.savez_compressed(os.path.join(bt_params['outputdir'], 'vxy_{:s}_fwhm_{:d}_avg_{:d}.npz'.format(kernel, fwhm, trange[1])),
                            vx=vx_t, vy=vy_t)
        return vx_t, vy_t

    with Pool(processes = 6) as pool:
        vx, vy = zip(*pool.map(make_velocity_wrapper, tranges))


    # for trange in tranges:
    #     vx, vy = blt.make_euler_velocity(b_top, b_bot, cal_top, cal_bot, [263, 263], trange, fwhm, kernel)
    #     np.savez_compressed(os.path.join(bt_params['outputdir'], 'vxy_fwhm_{:d}_avg_{:d}.npz'.format(fwhm, trange[1])),
    #                         vx=vx, vy=vy)

