import os, glob
import numpy as np
import fitstools
import balltracking.balltrack as blt
import multiprocessing
from multiprocessing import Pool
from functools import partial



def run_balltrack(args):
    intsteps, dp, sigma_factor = args
    outputdir_args = os.path.join(outputdir,
                                  'intsteps_{:d}_dp_{:0.1f}_sigmaf_{:0.2f}'.format(intsteps, dp, sigma_factor))

    drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()

    cal = blt.Calibrator(images2, drift_rates, nframes, rs, dp, sigma_factor, outputdir,
                         outputdir2=outputdir_args,
                         intsteps=intsteps,
                         output_prep_data=False,
                         use_existing=use_existing,
                         nthreads=4)

    cal.balltrack_all_rates(return_ballpos=False)


def run_calibration(args, fwhm, outputdir, vx_rates, trange, imshape, fov_slices):

    intsteps, dp, sigma_factor = args
    outputdir_args = os.path.join(outputdir,
                                  'intsteps_{:d}_dp_{:0.1f}_sigmaf_{:0.2f}'.format(intsteps, dp, sigma_factor))

    print('Processing data in {:s}'.format(outputdir_args))

    ballpos_top_list = np.load(os.path.join(outputdir_args, 'ballpos_top_list.npy'))
    ballpos_bottom_list = np.load(os.path.join(outputdir_args, 'ballpos_bottom_list.npy'))


    drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()
    xrates = np.array(drift_rates)[:, 0]
    a_top, vxfit_top, vxmeans_top, residuals_top = blt.fit_calibration(ballpos_top_list, xrates, trange, fwhm,
                                                        imshape, fov_slices,
                                                        return_flow_maps=False)
    a_bottom, vxfit_bottom, vxmeans_bottom, residuals_bottom = blt.fit_calibration(ballpos_bottom_list, xrates, trange, fwhm,
                                                                 imshape, fov_slices,
                                                                 return_flow_maps=False)
    # Fit the averaged calibrated balltrack velocity
    vxmeans = (vxfit_top + vxfit_bottom) / 2
    p = np.polyfit(vx_rates, vxmeans, 1)
    a_avg = 1 / p[0]
    vxfit_avg = a_avg * (vxmeans - p[1])
    # Calculate residuals
    residuals = np.abs(vxmeans - vx_rates)

    np.savez(os.path.join(outputdir_args, 'results_fwhm_{:d}.npz'.format(fwhm)),
             vxmeans=vxmeans, a_avg=a_avg, vxfit_avg=vxfit_avg, residuals=residuals,
             residuals_top=residuals_top, residuals_bottom=residuals_bottom)

    print('Saved data in {:s}'.format(outputdir_args))

    return vxmeans, a_avg, vxfit_avg, residuals


if __name__ == '__main__':


    # the multiprocessing start method can only bet set once.
    multiprocessing.set_start_method('spawn')

    # Set if we run balltrack, or just the flow map creation
    process_balltrack = False
    # Set if we use existing drifted images
    use_existing = True

    unit = 368000 / 60
    unit_str = '[m/s]'

    # input data, list of files
    # glob.glob does not order numbered files by defaultl, the order is as appeared in the file system.
    datafiles = sorted(glob.glob('/Users/rattie/Data/Ben/SteinSDO/SDO_int*.fits'))
    # output directory for the drifting images
    outputdir = '/Users/rattie/Data/Ben/SteinSDO/calibration/'
    plotdir = os.path.join('/Users/rattie/Data/Ben/SteinSDO/plots/')
    ### Ball parameters
    # Use 80 frames (1 hr)
    nframes = 80
    trange = [0, nframes]
    # Ball radius
    rs = 2

    # Select only a subset of nframes files
    selected_files = datafiles[0:nframes]

    npts = 9
    imshape = [264, 264]
    imsize = 263  # actual size is imshape = 264 but original size was 263 then padded to 264 to make it even for the Fourier transform
    vx_rates = np.linspace(-0.2, 0.2, npts)

    # Load the nt images
    images = fitstools.fitsread(selected_files)
    # Must make even dimensions for the fast fourier transform
    images2 = np.zeros([264, 264, images.shape[2]])
    images2[0:263, 0:263, :] = images.copy()
    images2[263, :] = images.mean()
    images2[:, 263] = images.mean()


    ##########################################
    intsteps_l = [3, 4, 5]
    dp_l = [0.1, 0.2, 0.3, 0.4, 0.5]
    sigma_factor_l = [1, 1.25, 1.5, 1.75, 2]

    # Build argument list for parallelization of run_balltrack()
    mesh_intsteps, mesh_dp, mesh_sigma_factor = np.meshgrid(intsteps_l, dp_l, sigma_factor_l, indexing='ij')
    intsteps_ravel = np.ravel(mesh_intsteps)
    dp_ravel = np.ravel(mesh_dp)
    sigmaf_ravel = np.ravel(mesh_sigma_factor)

    args_list = [list(a) for a in zip(intsteps_ravel, dp_ravel, sigmaf_ravel)]

    if process_balltrack:
        # Test on the 1st triplet of the series
        #ballpos_top_list, ballpos_bottom_list = run_balltrack(args_list[0])
        # Run on all triplets
        for args in args_list:
            run_balltrack(args)


    fwhms = [7, 15]
    trim = int(vx_rates.max() * nframes + max(fwhms) + 2)
    # FOV
    # Here we need to exclude the region that got circularly shifted by the Fourier phase shit ~ 23 pixels in both horizontal directions (left-right)
    fov_slices = [np.s_[trim:imsize - trim, trim:imsize-trim], ]

    # At FWHM 7
    fwhm = 7

    calibrate_partial = partial(run_calibration, fwhm=fwhm, outputdir=outputdir, vx_rates=vx_rates, trange=trange, imshape=imshape, fov_slices=fov_slices)

    pool = Pool(processes=4)
    vxmeans_l, a_avg_l, vxfit_avg_l, residuals_l = zip(*pool.map(calibrate_partial, args_list))
    pool.close()
    pool.join()

    vxmeans_la, a_avg_la, vxfit_avg_la, residuals_la = np.array(vxmeans_l), np.array(a_avg_l), np.array(vxfit_avg_l), np.array(residuals_l)
    np.savez(os.path.join(outputdir, 'all_results_fwhm_{:d}_simulation'.format(fwhm)),
             vxmeans_la=vxmeans_la, a_avg_la=a_avg_la, vxfit_avg_la=vxfit_avg_la, residuals_la=residuals_la)


