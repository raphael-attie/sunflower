import os
import numpy as np
import balltracking.balltrack as blt
import fitstools

def wrapper_velocity(filter_radius, fwhm, kernel, outputdir):

    ballpos_dir = os.path.join(outputdir, 'filter_radius{:d}'.format(filter_radius))
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
outputdir = '/Users/rattie/Data/Ben/SteinSDO/optimization_fourier_radius_3px_finer_grid/sigma_factor_1.50'
### Ball parameters
# Use 80 frames (1 hr)
nframes = 80
### Fourier filter radius
f_radius_l = np.arange(0, 21)
# Smoothing for Euler maps
fwhm = 7

for radius in f_radius_l:
    vel_dir = wrapper_velocity(radius, fwhm, 'boxcar', outputdir)

