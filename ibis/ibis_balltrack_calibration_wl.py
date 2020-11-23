import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import balltracking.balltrack as blt
from collections import OrderedDict
import fitstools
from pathlib import Path
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Description for my parser")
    parser.add_argument("-r", "--rs", help="Sphere radius", required=False, type=int, default=2)
    parser.add_argument("-s", "--sigma_factor", help="multiplier on sigma", required=False, type=float, default=1.5)
    parser.add_argument("-f", "--fourier_radius", help="fourier filtering (px)", required=False, type=float, default=2)
    args = parser.parse_args()
    print(args)

    # directory hosting the data
    ibisdir = Path(os.environ['DATA'], 'Ben', 'IBIS', 'white_light')
    # Get enough frames to make it to 1 hour of observations with dt = 12s
    nframes = 300
    trange = [0, nframes]
    # load image data
    fitsfile = str(ibisdir.joinpath('ibis.wl.speckle.cropped.fits'))

    # Ball parameters
    bt_params = OrderedDict({'rs': args.rs,
                             'intsteps': 5,
                             'ballspacing': args.rs,
                             'dp': 0.25,
                             'sigma_factor': args.sigma_factor,
                             'fourier_radius': args.fourier_radius,
                             'nframes': nframes,
                             'index': 'single'})

    # output directory for the drifting images
    outputdir = Path(ibisdir, f'calibration_rs{args.rs}_'
                              f'sigmaf{args.sigma_factor}_'
                              f'fourier{args.fourier_radius}')

    ##########################################
    # Calibration parameters
    # Set npts drift rates - use values of stein multiply by factor 5 (as we have 5x smaller pixel size)
    dv = 0.1
    vx_rates = np.arange(-0.5, 0.51, dv)
    vx_rates[int(len(vx_rates) / 2)] = 0
    vx_labels = ['vx_{:02d}'.format(i) for i in range(len(vx_rates))]
    drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()
    imsize = 920

    images = fitstools.fitsread(fitsfile, cube=True)
    # Velocity smoothing over 2.5 Mm
    fwhm = 37
    kernel = 'boxcar'
    bt_params['index'] = f'{kernel}_fwhm{fwhm}'

    trim = int(vx_rates.max() * nframes + fwhm + 2)
    fov_slices = np.s_[trim:imsize - trim, trim:imsize - trim]

    cal_dict = blt.balltrack_calibration(bt_params,
                                         images=images,
                                         drift_rates=drift_rates,
                                         trange=trange,
                                         fov_slices=fov_slices,
                                         reprocess_bt=True,
                                         outputdir=outputdir,
                                         kernel=kernel,
                                         fwhm=fwhm,
                                         dims=[imsize, imsize],
                                         save_ballpos_list=True,
                                         verbose=True,
                                         nthreads=11)

