import glob, os
import fitsio
import numpy as np
import matplotlib
matplotlib.use('macosx')
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import balltracking.balltrack as blt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from importlib import reload

def get_vel(frame_number):
    vel_x = fitsio.read(vx_files[frame_number]).astype(np.float32)
    vel_y = fitsio.read(vy_files[frame_number]).astype(np.float32)
    return vel_x, vel_y


def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])



def integrate_lanes_series(step, doprint=False):

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(lanes[step], cmap='Blues', origin='lower', vmin=0, vmax=50)
    ax.plot(pos[0][step, ::gap, ::gap], pos[1][step, ::gap, ::gap], ls='None', marker='.', ms=2, color='red')
    ax.plot(pos[0][0, ::gap, ::gap], pos[1][0, ::gap, ::gap], ls='None', marker='.', ms=2, color='green', markerfacecolor='None')

    x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))
    quiv = ax.quiver(x[::quiv_res, ::quiv_res], y[::quiv_res, ::quiv_res], vx[::quiv_res, ::quiv_res], vy[::quiv_res, ::quiv_res],
                     units='xy', scale=quiver_scale, width=shaft_width, headwidth=headwidth, headlength=headlength)

    plt.axis([100, 400, 100, 400])
    plt.xlabel('x [px]', fontsize=FS - 2)
    plt.ylabel('y [px]', fontsize=FS - 2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    cax.tick_params(labelsize=FS - 2)
    cax.set_ylabel('Distance from initial position [px]', fontsize=FS)
    ax.tick_params(labelsize=FS - 2)
    if step == 0:
        ax.set_title('Initial position. (Integration step: %d)' % step, fontsize=FS)
    else:
        ax.set_title('Integration step: %d' % step, fontsize=FS)
    plt.tight_layout()


    if doprint:
        fname = os.path.join(fig_dir, 'lanes_%d.png' % step)
        fig.savefig(fname, dpi=300)



tracking_dir = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/python_balltracking'
fig_dir = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/lanes/'

### Velocity field parameters
fwhm = 15
tavg = 160
tstep = 80
# Units
px_meter = 0.03 * 3.1415/180 * 6.957e8
ms_unit = px_meter / 45
# Get velocity files
vx_files = glob.glob(os.path.join(tracking_dir,'vx_fwhm%d_tavg%d_[0-9]*.fits'%(fwhm, tavg)))
vy_files = glob.glob(os.path.join(tracking_dir,'vy_fwhm%d_tavg%d_[0-9]*.fits'%(fwhm, tavg)))

frame_nb = 3
vx, vy = get_vel(frame_nb)

### Lanes parameters
nsteps = 40
maxstep = 4

lanes, pos = blt.make_lanes_visualization(vx, vy, nsteps, maxstep)

vx *= ms_unit
vy *= ms_unit
div = divergence(np.array([vx, vy]))


# Plot the lanes
FS = 16
gap = 12

# Quiver plot parameters
quiv_res = 10
quiver_scale = 50
shaft_width = 1.
headwidth = 3
headlength = 4

fig, ax = plt.subplots(1, 1, figsize=(9, 7))
im = ax.imshow(div, origin='lower', cmap='Blues')
x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))
quiv = ax.quiver(x[::quiv_res, ::quiv_res], y[::quiv_res, ::quiv_res], vx[::quiv_res, ::quiv_res], vy[::quiv_res, ::quiv_res],
                 units='xy', scale=quiver_scale, width=shaft_width, headwidth=headwidth, headlength=headlength)

plt.axis([100, 400, 100, 400])
plt.xlabel('x [px]', fontsize=FS - 2)
plt.ylabel('y [px]', fontsize=FS - 2)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax)
cax.tick_params(labelsize=FS - 2)
cax.set_ylabel(r'Divergence [$s^{-1}$]', fontsize=FS)
ax.tick_params(labelsize=FS - 2)
ax.set_title('Divergence field')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'divergence.png'))

# integrate_lanes_series(0)
# integrate_lanes_series(1)
# integrate_lanes_series(2)
# integrate_lanes_series(10)
integrate_lanes_series(1)

for step in range(nsteps+1):
    integrate_lanes_series(step, doprint=True)
    plt.close()
    #integrate_lanes_series(nsteps)

