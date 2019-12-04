import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import balltracking.balltrack as blt
import fitstools
from collections import OrderedDict


if __name__ == '__main__':
    # Get the intensity files
    datacubefile = os.path.join(os.environ['DATA'],
                                'SDO/HMI/continuum/Lat_63/mtrack_20110627_200034_TAI_20110628_000033_TAI_Postel_060.4_63.0_continuum.fits')
    # directory hosting the drifted data
    drift_dir = os.path.join(os.environ['DATA'], 'SDO/HMI/continuum/Lat_63/calibration')
    outputdir = drift_dir
    # Ball parameters
    bt_params = OrderedDict({'rs': 2,
                             'intsteps': 4,
                             'ballspacing': 2,
                             'dp': 0.3,
                             'sigma_factor': 1.75,
                             'fourier_radius': 1.0,
                             'nframes': 80,
                             'datafiles': datacubefile,
                             'outputdir': os.path.join(os.environ['DATA'], 'SDO/HMI/continuum/Lat_63/balltrack'),
                             'index': 0,
                             'ncores': 1})

    reprocess_bt = False
    # Load the nt images
    images = fitstools.fitsread(datacubefile, tslice=slice(0, bt_params['nframes'])).astype(np.float32)
    # Calibration parameters
    # Set npts drift rates
    dv = 0.04
    vx_rates = np.arange(-0.2, 0.21, dv)
    vx_rates[int(len(vx_rates) / 2)] = 0
    drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()
    trange = [0, bt_params['nframes']]

    # Smoothing
    fwhm = 11
    dims = images.shape[0:2]
    trim = int(vx_rates.max() * bt_params['nframes'] + fwhm + 2)
    fov_slices = np.s_[trim:dims[0] - trim, trim:dims[1] - trim]
    kernel = 'gaussian'

    cal = blt.balltrack_calibration(bt_params, drift_rates, trange, fov_slices, reprocess_bt, drift_dir, outputdir,
                                    kernel, fwhm, dims, nthreads=4, verbose=True)


    ## Get flow maps from tracked positions
    #
    # trange = [0, bt_params['nframes']]
    # fwhm = 15
    #
    # xrates = np.array(drift_rates)[:,0]
    # a_top, vxfit_top, vxmeans_top, residuals_top = blt.fit_calibration(ballpos_top_list, xrates, trange, fwhm, images.shape[0:2], fov_slices)
    # a_bottom, vxfit_bottom, vxmeans_bottom, residuals_bottom = blt.fit_calibration(ballpos_bottom_list, xrates, trange, fwhm, images.shape[0:2], fov_slices)
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



