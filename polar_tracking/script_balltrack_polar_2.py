import balltracking.balltrack as blt
import os, glob
import numpy as np
from collections import OrderedDict


if __name__ == '__main__':

    # Get the intensity files
    datacubefile = os.path.join(os.environ['DATA'], 'SDO/HMI/continuum/Lat_63/mtrack_20110627_200034_TAI_20110628_000033_TAI_Postel_060.4_63.0_continuum.fits')
    # Ball parameters
    bt_params = OrderedDict({'rs': 2,
                             'intsteps': 4,
                             'ballspacing': 2,
                             'dp': 0.3,
                             'sigma_factor': 1.75,
                             'fourier_radius': 1.0,
                             'nframes': 320,
                             'datafiles': datacubefile,
                             'outputdir': os.path.join(os.environ['DATA'], 'SDO/HMI/continuum/Lat_63/balltrack'),
                             'ncores': 4})


    b_top, b_bot = blt.balltrack_all(**bt_params)

    # # Make velocity flow fields
    # kernel = 'gaussian'
    # fwhm = 10
    # trange = [0, bt_params['nframes']]
    # cal_top = 1.74
    # cal_bot = 1.71
    #
    # for trange in tranges:
    #     vx, vy = blt.make_euler_velocity(b_top, b_bot, cal_top, cal_bot, [263, 263], trange, fwhm, kernel)
    #     np.savez_compressed(os.path.join(bt_params['outputdir'], 'vxy_fwhm_{:d}_avg_{:d}.npz'.format(fwhm, trange[1])),
    #                         vx=vx, vy=vy)

