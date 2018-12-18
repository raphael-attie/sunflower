import matplotlib
matplotlib.use('agg')
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import fitstools
import balltracking.balltrack as blt
import multiprocessing
from multiprocessing import Pool
from functools import partial


def run_calibration(args, fwhm):


    intsteps, dp, sigma_factor = args
    outputdir_args = os.path.join(outputdir,
                                  'intsteps_{:d}_dp_{:0.1f}_sigmaf_{:0.2f}'.format(intsteps, dp, sigma_factor))

    ballpos_top_list = np.load(os.path.join(outputdir_args,'ballpos_top_list.npy') )
    ballpos_bottom_list = np.load(os.path.join(outputdir_args, 'ballpos_top_list.npy'))

    xrates = np.array(drift_rates)[:, 0]
    a_top, vxfit_top, vxmeans_top = blt.fit_calibration(ballpos_top_list, xrates, trange, fwhm,
                                                        imshape, fov_slices,
                                                        return_flow_maps=False)
    a_bottom, vxfit_bottom, vxmeans_bottom = blt.fit_calibration(ballpos_bottom_list, xrates, trange, fwhm,
                                                                 imshape, fov_slices,
                                                                 return_flow_maps=False)
    # Fit the averaged calibrated balltrack velocity
    vxmeans_bt = (vxmeans_top * a_top + vxmeans_bottom * a_bottom) / 2
    vxmeans_bt -= vxmeans_bt[4]
    p = np.polyfit(vx_rates, vxmeans_bt, 1)
    a_avg = 1 / p[0]
    vxfit_avg = a_avg * (vxmeans_bt - p[1])
    # Calculate residuals
    bt_residuals = np.abs(vxmeans_bt - vxfit_avg)

    return vxmeans_bt, a_avg, vxfit_avg, bt_residuals




outputdir = '/Users/rattie/Data/Ben/SteinSDO/calibration/'

npts = 9
nframes = 80
trange = [0, nframes]
vx_rates = np.linspace(-0.2, 0.2, npts)
drift_rates = np.stackrun_ca((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()
imshape = [264, 264]
imsize = 263  # actual size is imshape = 264 but original size was 263 then padded to 264 to make it even for the Fourier transform


fwhms = [7, 15]
trim = int(vx_rates.max() * nframes + max(fwhms) + 2)
# FOV
# Here we need to exclude the region that got circularly shifted by the Fourier phase shit ~ 23 pixels in both horizontal directions (left-right)
fov_slices = [np.s_[trim:imsize - trim, trim:imsize - trim], ]


# At FWHM 15