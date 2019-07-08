import os, glob
import numpy as np
import fitstools
import balltracking.balltrack as blt
import multiprocessing
from multiprocessing import Pool
from functools import partial


def wrapper_balltrack(args, data, filter_radius, outputdir):

    nt, rs, dp, sigma_factor, intsteps = args
    outputdir_args = os.path.join(outputdir,
                                  'rs{:0.1f}_dp{:0.1f}_sigmaf{:0.2f}_intsteps{:d}_nt{:d}'.format(rs, dp, sigma_factor, intsteps, nt))
    print(outputdir_args)
    _, _ = blt.balltrack_all(nt, rs, dp, sigma_factor, intsteps, outputdir_args, fourier_radius=filter_radius, data=data, ncores=1)
    return outputdir


if __name__ == '__main__':


    # the multiprocessing start method can only bet set once.
    multiprocessing.set_start_method('spawn')
    nthreads = 4
    # input data, list of files
    # glob.glob does not order numbered files by defaultl, the order is as appeared in the file system.
    datafiles = sorted(glob.glob('/Users/rattie/Data/Ben/SteinSDO/SDO_int*.fits'))
    # output directory for the drifting images
    outputdir = '/Users/rattie/Data/Ben/SteinSDO/optimization_fourier_radius_3px'
    plotdir = os.path.join('/Users/rattie/Data/Ben/SteinSDO/optimization/plots/')
    ### Ball parameters
    # Use 80 frames (1 hr)
    nframes = 80
    # Fourier filtering
    filter_radius = 3
    # Ball radius
    rs = 2
    # Get series of all other input parameters
    dp_l = [0.1, 0.2, 0.3, 0.4, 0.5]
    sigma_factor_l = [1, 1.25, 1.5, 1.75, 2]
    intsteps_l = [3, 4, 5]
    # Build argument list for parallelization of run_balltrack()
    mesh_nt, mesh_rs, mesh_dp, mesh_sigma_factor, mesh_intsteps = np.meshgrid(nframes, rs, dp_l, sigma_factor_l, intsteps_l, indexing='ij')
    args_list = [list(a) for a in zip(mesh_nt.ravel(), mesh_rs.ravel(), mesh_dp.ravel(), mesh_sigma_factor.ravel(), mesh_intsteps.ravel())]
    # Select only a subset of nframes files
    selected_files = datafiles[0:nframes]
    imshape = [264, 264]
    imsize = 263  # actual size is imshape = 264 but original size was 263 then padded to 264 to make it even for the Fourier transform
    # Load the nt images
    images = fitstools.fitsread(selected_files)
    # Must make even dimensions for the fast fourier transform
    images2 = np.zeros([imshape[0], imshape[1], images.shape[2]])
    images2[0:imsize, 0:imsize, :] = images.copy()
    images2[imsize, :, :] = images.mean()
    images2[:, imsize, :] = images.mean()
    ##########################################

    # Run Balltrack  on all triplets in parallel
    # Test on the 1st triplet of the series
    #ballpos_top_list, ballpos_bottom_list = run_balltrack(args_list[0])
    # Run on all triplets
    # wrapper_partial = partial(wrapper_balltrack, data=images, outputdir=outputdir)
    # outputdir = wrapper_partial(args_list[0])

    with Pool(processes=nthreads) as pool:
        wrapper_partial = partial(wrapper_balltrack, data=images2, filter_radius=filter_radius, outputdir=outputdir)
        outputdirs = pool.map(wrapper_partial, args_list)

