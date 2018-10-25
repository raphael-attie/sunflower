import os
import fitsio
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.filters import gaussian_filter


def quiver_plot(vx, vy, step, vunit=1, scale=0.05, width=0.1, headwidth=3, headlength=4, title='', axis=None):
    v_slice = np.s_[::step, ::step]
    # Grid of the flow fields, first define the default, then slice it to match anchors spaced by "step" pixels
    x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))
    x = x[v_slice]
    y = y[v_slice]
    # Slice the velocity to match anchor spacing
    vx_s = vx[v_slice] * vunit
    vy_s = vy[v_slice] * vunit

    # norm of velocity for color mapping
    #vnorm = np.sqrt(vx_s ** 2 + vy_s ** 2)

    ###############################################################################################
    ############################  Without color mapping ###########################################
    ###############################################################################################
    fig, axs = plt.subplots(figsize=(10, 10))
    axs.quiver(x, y, vx_s, vy_s,
               units='xy', scale=scale, width=width, headwidth=headwidth, headlength=headlength)
    plt.gca().set_aspect('equal')
    plt.axis([0, vx.shape[1], 0, vx.shape[0]])
    if axis is not None:
        plt.axis(axis)
    plt.title(title)

    plt.tight_layout()
    plt.show()

    return fig


datadir = '/Users/rattie/Data/Ben/SteinSDO'
plotdir = '/Users/rattie/Data/Ben/SteinSDO/plots/'

# Load the data whose filenames are based on parameters used in Balltracking
# Nb of frames used
nframes = 30
# Smoothing size
fwhm = 7
# File name scheme suffix
filename_suffix = 'fwhm{:d}_tavg{:d}_000'.format(fwhm, nframes)

# Load velocity fields from balltracking
vx_bt = fitsio.read(os.path.join(datadir, 'balltracking', 'vx_{}.fits'.format(filename_suffix)))
vy_bt = fitsio.read(os.path.join(datadir, 'balltracking', 'vy_{}.fits'.format(filename_suffix)))
# Load velocity fields from LCT
vx_lct = fitsio.read(os.path.join(datadir, 'TimeAverage_290040-291720', 'LCT_vx1_290040-291720.fits'))
vy_lct = fitsio.read(os.path.join(datadir, 'TimeAverage_290040-291720', 'LCT_vy1_290040-291720.fits'))
# Load velocity fields from Stein
vx_stein = fitsio.read(os.path.join(datadir, 'TimeAverage_290040-291720', 'SteinSDO_vx1_290040-291720.fits'))
vy_stein = fitsio.read(os.path.join(datadir, 'TimeAverage_290040-291720', 'SteinSDO_vy1_290040-291720.fits'))
# Smooth Stein like it was smoothed with LCT and Balltrack
sigma = fwhm / 2.35
vx_stein_sm = gaussian_filter(vx_stein, sigma=sigma, order=0)
vy_stein_sm = gaussian_filter(vy_stein, sigma=sigma, order=0)


#### Prepare quiver plot the flow fields.
#### More info for customization at https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.quiver.html

fig_bt = quiver_plot(vx_bt, vy_bt, 1, axis=[51, 101, 51, 101], vunit=368000/60, scale=300, width=0.1, headwidth=3, headlength=4, title='balltracking')
fig_bt.savefig(os.path.join(plotdir, 'quiver_plot_{}_bw_BT_zoom.png'.format(filename_suffix)))

fig_lct = quiver_plot(vx_lct, vy_lct, 1, axis=[50, 100, 50, 100], vunit=1000, scale=300, width=0.1, headwidth=3, headlength=4, title='LCT')
fig_lct.savefig(os.path.join(plotdir, 'quiver_plot_{}_bw_LCT_zoom.png'.format(filename_suffix)))

fig_stein = quiver_plot(vx_stein_sm, vy_stein_sm, 1, axis=[50, 100, 50, 100], vunit=1, scale=300, width=0.1, headwidth=3, headlength=4, title='Stein')
fig_stein.savefig(os.path.join(plotdir, 'quiver_plot_{}_bw_stein_zoom.png'.format(filename_suffix)))


###############################################################################################
############################ With color mapping by norm of velocity ###########################
###############################################################################################
# fig, axs = plt.subplots(figsize=(10, 10))
# quiv = axs.quiver(x, y, vx_s, vy_s, vnorm,
#            units='xy', scale=0.01, width=0.5, headwidth=4, headlength=4, cmap='magma')
#
# plt.gca().set_aspect('equal')
# plt.axis([0, vx.shape[1], 0, vx.shape[0]])
# # Create a nicely aligned container for the colorbar
# divider = make_axes_locatable(axs)
# cax = divider.append_axes("right", size="3%", pad=0.20)
# # Plot the colorbar in that container
# plt.colorbar(quiv, cax=cax)
# cax.set_ylabel('|V| [px / frame interval]')
#
# plt.tight_layout()
# plt.show()
# plt.savefig(os.path.join(plotdir, 'quiver_plot_{}_colored.png'.format(filename_suffix)))
#

