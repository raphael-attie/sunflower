import os
import numpy as np
import balltracking.balltrack as blt
import fitstools
from collections import OrderedDict
from functools import partial


if __name__ == '__main__':

    # Longitudes (Stonyhurst) at starting frame
    # lons = [-7, -67, -77]
    # Longitudes (Carrington):
    # lonCRs = [60, 0, 350]
    # Latitudes
    # lats = [0, 60, 70]
    lon_lats = [(60.4, 0), (0, 0), (350, 0), (60.4, 60), (60.4, 70)]
    nevents = len(lon_lats)
    basenames_l = [f'mtrack_20110627_200034_TAI_20110628_000033_TAI_Postel_{lon_lat[0]:05.1f}_{lon_lat[1]:04.1f}_continuum.fits'
                   for lon_lat in lon_lats]
    # Get the intensity files
    datacubefiles = [os.path.join(os.environ['DATA'], 'SDO/HMI/polar_study/', basename) for basename in basenames_l]
    outputdirs_l = [os.path.join(
        os.environ['DATA'], f'SDO/HMI/polar_study/lonCR_{lon_lat[0]:05.1f}_lat_{lon_lat[1]:04.1f}/calibration')
                    for lon_lat in lon_lats]

    reprocess_bt = True
    nframes = 60
    trange = [0, nframes]
    # Ball parameters
    bt_params_top = OrderedDict({'rs': 2,
                                 'ballspacing': 1,
                                 'intsteps': 6,
                                 'dp': 0.3,
                                 'sigma_factor': 2.0,
                                 'fourier_radius': 1.0,
                                 'index': 0})

    bt_params_bottom = OrderedDict({'rs': 2,
                                    'ballspacing': 2,
                                    'intsteps': 4,
                                    'dp': 0.20,
                                    'sigma_factor': 2.0,
                                    'fourier_radius': 1.0,
                                    'index': 1})

    # Calibration parameters
    # Set npts drift rates
    dv = 0.04
    vx_rates = np.arange(-0.2, 0.21, dv)
    vx_rates[int(len(vx_rates) / 2)] = 0
    vx_labels = ['vx_{:02d}'.format(i) for i in range(len(vx_rates))]
    drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()
    # Smoothing
    fwhm = 7
    dims = [512, 512]
    trim = int(vx_rates.max() * nframes + fwhm + 2)
    fov_slices = np.s_[trim:dims[0] - trim, trim:dims[1] - trim]
    kernel = 'gaussian'

    for idx in range(1, nevents):
        # idx = 0
        print('Calibrating flows on datacube: ', datacubefiles[idx])
        # Load the nt images
        images = fitstools.fitsread(datacubefiles[idx], tslice=slice(*trange)).astype(np.float32)
        outputdir = outputdirs_l[idx]


        # Must provide source images to create drift images. Do not provide the images if the drift images already exist
        calibrate_partial = partial(blt.balltrack_calibration,
                                    images=images,
                                    drift_rates=drift_rates,
                                    trange=trange,
                                    fov_slices=fov_slices,
                                    reprocess_bt=reprocess_bt,
                                    outputdir=outputdir,
                                    kernel=kernel,
                                    fwhm=fwhm,
                                    dims=dims,
                                    save_ballpos_list=False,
                                    nthreads=11)

        calibrate_partial(bt_params_top)
        calibrate_partial(bt_params_bottom)

    ## Get flow maps from tracked positions
    #
    #
    # plt.figure(0)
    # plt.plot(vxmeans_top, vx_rates, 'r.', label='data top', zorder=3)
    # plt.plot(vxmeans_bottom, vx_rates, 'g+', label='data bottom', zorder=3)
    # plt.plot(vxmeans_top, vxfit_top, 'b-', label=r'$\alpha_t$ =%0.2f' %a_top, zorder=2)
    # plt.plot(vxmeans_bottom, vxfit_bottom, 'k-', label=r'$\alpha_b$ =%0.2f' % a_bottom, zorder=2)
    #
    # plt.xlabel('Balltracked <Vx> (px/frame)')
    # plt.ylabel('Drift <Vx> (px/frame)')
    # plt.grid('on')
    # plt.legend()
    #
    # plt.savefig(os.path.join(outputdir,'calibration.png'))
    #
    # plt.close('all')
    # plt.figure(1)
    # width = 150
    # plt.bar(vx_rates * unit, residuals_top * unit, width = width, color='black', label='top-side tracking')
    # plt.bar(vx_rates * unit + width, residuals_bottom * unit, width=width, color='gray', label='bottom-side tracking')
    # plt.xlabel('Drift <Vx> (m/s)')
    # plt.ylabel('Absolute residual error on <Vx> (m/s)')
    # plt.ylim([0, 10])
    # plt.grid(True, ls=':')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(outputdir, 'residuals_top_bottom.png'), dpi=180)



