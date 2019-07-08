import os, glob
import numpy as np
import balltracking.balltrack as blt
import fitstools
from multiprocessing import Pool
from functools import partial


def wrapper_velocity(args, fwhm, kernel, outputdir):
    nt, rs, dp, sigma_factor, intsteps = args
    ballpos_dir = os.path.join(outputdir,
                                  'rs{:0.1f}_dp{:0.1f}_sigmaf{:0.2f}_intsteps{:d}_nt{:d}'.format(rs, dp, sigma_factor, intsteps, nt))
    print(ballpos_dir)
    ballpos_top = np.load(os.path.join(ballpos_dir, 'ballpos_top.npy'))
    ballpos_bottom = np.load(os.path.join(ballpos_dir, 'ballpos_bottom.npy'))

    dims = [264, 264]
    fov = np.s_[0:263, 0:263]
    trange = [0, nframes]
    vx_top, vy_top, _ = blt.make_velocity_from_tracks(ballpos_top, dims, trange, fwhm, kernel=kernel)
    vx_top, vy_top = vx_top[fov], vy_top[fov]

    vx_bottom, vy_bottom, _ = blt.make_velocity_from_tracks(ballpos_bottom, dims, trange, fwhm, kernel=kernel)
    vx_bottom, vy_bottom = vx_bottom[fov], vy_bottom[fov]

    fitstools.writefits(vx_top, os.path.join(ballpos_dir, 'vxtop_fwhm{:d}_kernel_{:s}.fits'.format(fwhm, kernel)))
    fitstools.writefits(vy_top, os.path.join(ballpos_dir, 'vytop_fwhm{:d}_kernel_{:s}.fits'.format(fwhm, kernel)))
    fitstools.writefits(vx_bottom, os.path.join(ballpos_dir, 'vxbottom_fwhm{:d}_kernel_{:s}.fits'.format(fwhm, kernel)))
    fitstools.writefits(vy_bottom, os.path.join(ballpos_dir, 'vybottom_fwhm{:d}_kernel_{:s}.fits'.format(fwhm, kernel)))

    return ballpos_dir


# parent directory for the ballpos and velocity directories
outputdir = '/Users/rattie/Data/Ben/SteinSDO/optimization_fourier_radius_3px_finer_grid/'
### Ball parameters
# Use 80 frames (1 hr)
nframes = 80
# Ball radius
rs = 2
# Get series of all other input parameters
dp_l = [0.1, 0.2, 0.3, 0.4, 0.5]
sigma_factor_l = [1, 1.25, 1.5, 1.75, 2]
intsteps_l = [3, 4, 5]
# Build argument list for parallelization of run_balltrack()
mesh_nt, mesh_rs, mesh_dp, mesh_sigma_factor, mesh_intsteps = np.meshgrid(nframes, rs, dp_l, sigma_factor_l, intsteps_l,
                                                                          indexing='ij')
args_list = [list(a) for a in
             zip(mesh_nt.ravel(), mesh_rs.ravel(), mesh_dp.ravel(), mesh_sigma_factor.ravel(), mesh_intsteps.ravel())]



# Smoothing for Euler maps
fwhm = 7
nthreads = 4
with Pool(processes=nthreads) as pool:
    wrapper_partial = partial(wrapper_velocity, fwhm=fwhm, kernel='boxcar', outputdir=outputdir)
    outputdirs = pool.map(wrapper_partial, args_list)


# with Pool(processes=nthreads) as pool:
#     wrapper_partial = partial(wrapper_velocity, fwhm=fwhm, kernel='gaussian', outputdir=outputdir)
#     outputdirs = pool.map(wrapper_partial, args_list)
