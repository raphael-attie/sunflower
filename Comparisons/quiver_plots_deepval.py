import os
import fitsio
import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

datadir = '/Users/rattie/Data/Ben/SteinSDO/calibration/'
plotdir = '/Users/rattie/Data/Ben/SteinSDO/plots/'

# Load the data whose filenames are based on parameters used in Balltracking
# Nb of frames used
nframes = 30
# Smoothing size
fwhm = 15
# File name scheme suffix
filename_suffix = 'tavg{:d}_fwhm{:d}'.format(nframes, fwhm)

# Load velocity fields
vx = fitsio.read(os.path.join(datadir, 'vx_{}_cal.fits'.format(filename_suffix)))
vy = fitsio.read(os.path.join(datadir, 'vy_{}_cal.fits'.format(filename_suffix)))


#### Prepare quiver plot the flow fields.
#### More info for customization at https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.quiver.html

# Spacing between arrow anchors (in pixels)
step = 5
# Parametrize the velocity slicing for simpler expressions later on.
v_slice = np.s_[::step, ::step]
# Grid of the flow fields, first define the default, then slice it to match anchors spaced by "step" pixels
x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))
x = x[v_slice]
y = y[v_slice]
# Slice the velocity to match anchor spacing
vx_s = vx[v_slice]
vy_s = vy[v_slice]
# norm of velocity for color mapping
vnorm = np.sqrt(vx_s ** 2 + vy_s ** 2)

###############################################################################################
############################  Without color mapping ###########################################
###############################################################################################
fig, axs = plt.subplots(figsize=(10, 10))
axs.quiver(x, y, vx_s, vy_s,
           units='xy', scale=0.01, width=0.5, headwidth=4, headlength=4)
plt.gca().set_aspect('equal')
plt.axis([0, vx.shape[1], 0, vx.shape[0]])
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(plotdir, 'quiver_plot_{}_BW.png'.format(filename_suffix)))

###############################################################################################
############################ With color mapping by norm of velocity ###########################
###############################################################################################
fig, axs = plt.subplots(figsize=(10, 10))
quiv = axs.quiver(x, y, vx_s, vy_s, vnorm,
           units='xy', scale=0.01, width=0.5, headwidth=4, headlength=4, cmap='magma')

plt.gca().set_aspect('equal')
plt.axis([0, vx.shape[1], 0, vx.shape[0]])
# Create a nicely aligned container for the colorbar
divider = make_axes_locatable(axs)
cax = divider.append_axes("right", size="3%", pad=0.20)
# Plot the colorbar in that container
plt.colorbar(quiv, cax=cax)
cax.set_ylabel('|V| [px / frame interval]')

plt.tight_layout()
plt.show()
plt.savefig(os.path.join(plotdir, 'quiver_plot_{}_colored.png'.format(filename_suffix)))


