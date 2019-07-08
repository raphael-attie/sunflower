import os, glob
import numpy as np
import fitstools
import balltracking.balltrack as blt
import multiprocessing
from multiprocessing import Pool
from functools import partial


def wrapper_balltrack(filter_radius, ballspacing, args, data, outputdir):

    nt, rs, dp, sigma_factor, intsteps = args
    outputdir_filter = os.path.join(outputdir, 'filter_radius{:d}'.format(filter_radius))
    print(outputdir_filter)
    _, _ = blt.balltrack_all(nt, rs, dp, sigma_factor, intsteps, outputdir_filter, fourier_radius=filter_radius, ballspacing=ballspacing, data=data, ncores=1)
    return outputdir


if __name__ == '__main__':


    # the multiprocessing start method can only bet set once.
    multiprocessing.set_start_method('spawn')
    nthreads = 4
    # input data, list of files
    # glob.glob does not order numbered files by defaultl, the order is as appeared in the file system.
    datafiles = sorted(glob.glob('/Users/rattie/Data/Ben/SteinSDO/SDO_int*.fits'))
    ### Ball parameters
    # Use 80 frames (1 hr)
    nframes = 80
    # Ball radius
    rs = 2
    # Get series of all other input parameters
    dp = 0.3
    sigma_factor = 1.5
    intsteps = 5
    # Space between balls on initial grid
    ballspacing = 3
    ### Fourier filter radius
    f_radius_l = np.arange(0, 21)
    # output directory for the drifting images
    outputdir = '/Users/rattie/Data/Ben/SteinSDO/optimization_fourier_radius_3px_finer_grid/sigma_factor_{:1.2f}'.format(sigma_factor)
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

    with Pool(processes=nthreads) as pool:
        wrapper_partial = partial(wrapper_balltrack, args=(nframes, rs, dp, sigma_factor, intsteps), ballspacing=ballspacing, data=images2, outputdir=outputdir)
        outputdirs = pool.map(wrapper_partial, f_radius_l)

