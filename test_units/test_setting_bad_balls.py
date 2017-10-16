from importlib import reload
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import balltracking.balltrack as blt
import fitstools
import fitsio
import filters
import cython_modules.interp as cinterp
from timeit import timeit

#%gui qt
#from mayavi import mlab

plt.ioff()

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

file = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/series_continuum/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum_00000.fits'
#file = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/series_continuum_calibration/calibration/drift0.20/drift_0001.fits'
# Get the header
h       = fitstools.fitsheader(file)
# Get the 1st image
image   = fitsio.read(file).astype(np.float32).copy(order='C')
# Filter image
ffilter_hpf = filters.han2d_bandpass(image.shape[0], 0, 5)
fdata_hpf = filters.ffilter_image(image, ffilter_hpf)
sigma = fdata_hpf[1:200, 1:200].std()

surface = blt.rescale_frame(fdata_hpf, 2*sigma).astype(np.float32)


nt = 15
rs = 2
dp = 0.2
bt = blt.BT(image.shape, nt, rs, dp)
# Initialize ball positions
bt.initialize_ballpos(surface)

# integrate motion over some time steps. Enough to have overpopulated cells (bad balls).
pos, _, _ = [np.array(v).squeeze().swapaxes(0,1) for v in zip(*[blt.integrate_motion2(bt.pos_t, bt.vel_t, bt, surface) for i in range(nt)])]



# Display image, first positions, and last positions

plt.figure(1)
# image
plt.imshow(surface, cmap='gray', origin='lower', vmin=-1, vmax=1)
# initial positions
plt.plot(bt.xstart, bt.ystart, 'r.', markersize=4)
plt.axis([0, 60, 0, 60])

# # Last position
# plt.plot(pos[0, -1, :], pos[1, -1, :], 'c.', markersize=4)
# Current position
for i in range(nt-1):
    plt.plot(pos[0,i,:], pos[1,i,:], 'b.', markersize=2)
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/fig_b_%02d.png'%i)


# Mask of the bad balls at a given time tstop
tstop = 14
bad_balls = blt.get_bad_balls(pos[:,tstop,:], bt)
bad_pos = pos[:, tstop, bad_balls]

x = bad_pos[0,:]
y = bad_pos[1,:]

xcoarse, ycoarse, _ = blt.coarse_grid_pos(bt, x, y)

chess_board = blt.fill_coarse_grid(bt, x, y)

# fig = plt.figure(2, figsize=(8,8))
# #ax = fig.add_subplot(1,1,1)
# # Show the chess board with and setup proper axis
# plt.imshow(chess_board, cmap='gray_r', origin='lower', vmin=0, vmax=1, extent=(0,bt.nx-1, 0, bt.ny-1))

#mlab.text3d(250, 250, 1, 'blah 3D', color=red)


# initial positions
#plt.plot(bt.xstart, bt.ystart, 'r.', markersize=4)
# for i in range(nt-1):
#     plt.plot(pos[0,i,:], pos[1,i,:], 'b.', markersize=2)
plt.axis([-1, 60, -1, 60])
# Last position
plt.plot(pos[0, tstop, :], pos[1, tstop, :], 'c.', markersize=4)

# bad position
plt.plot(x, y, 'yo', markersize=6, markerfacecolor='none')
plt.plot(np.floor(x/bt.ballspacing)*bt.ballspacing, np.floor(y/bt.ballspacing)*bt.ballspacing, 'ro', markersize=6, markerfacecolor='none')

# Overaly the coarse grid lines
#Spacing between each line
intervals = bt.ballspacing
loc = plticker.MultipleLocator(base=intervals)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.grid(which='major', axis='both', linestyle='-')


plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/chessboard.png')



# ball labels
bnumbers = np.char.mod('%d', np.arange(bt.nballs))
for x,y,s in zip(pos[0, tstop, :], pos[1, tstop, :], bnumbers):
    plt.text(x, y, s, color='red', clip_on=True)
