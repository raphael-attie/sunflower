from importlib import reload
import numpy as np
import matplotlib
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import balltracking.balltrack as blt
import fitstools
import fitsio
import filters
from timeit import timeit

#%gui qt
#from mayavi import mlab

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
# Rescale image to a data surface
surface = blt.rescale_frame(fdata_hpf, 2*sigma).astype(np.float32)

### Ball parameters
# Nb of intermediate steps
nt = 15
# Ball radius
rs = 2
# depth factor
dp = 0.2
# Get a BT instance with the above parameters
bt = blt.BT(image.shape, nt, rs, dp)
# Initialize ball positions
bt.initialize_ballpos(surface)

# integrate motion over some time steps. Enough to have overpopulated cells (bad balls).
pos, _, _ = [np.array(v).squeeze().swapaxes(0,1) for v in zip(*[blt.integrate_motion2(bt.pos, bt.vel, bt, surface) for i in range(nt)])]


# Mask of the bad balls at a given time tstop
tstop = 14
bad_balls = blt.get_bad_balls(pos[:,tstop,:], bt)
bad_pos = pos[:, tstop, bad_balls]

x = bad_pos[0,:]
y = bad_pos[1,:]

# Fill the coarse grid at the cells that map to the bad positions
chess_board = blt.fill_coarse_grid(bt, x, y)

# Replace bad balls and get the new positions that fills the empty coarse grid cells
xnew, ynew = blt.replace_bad_balls(pos[:,tstop,:], bad_balls, surface, bt)

### Display ###

fig = plt.figure(2, figsize=(8,8))
ax = fig.add_subplot(1,1,1)
# Show the chess board with and setup proper axis
plt.imshow(chess_board, cmap='coolwarm', origin='lower', vmin=-1, vmax=1, extent=(0,bt.nx-1, 0, bt.ny-1))
plt.plot(pos[0, :, :], pos[1, :, :], 'b.')

plt.axis([0, 60, 0, 60])
# Last position
plt.plot(pos[0, tstop, :], pos[1, tstop, :], 'c.', markersize=4)

# bad position
plt.plot(x, y, 'yo', markersize=6, markerfacecolor='none')
#plt.plot(np.floor(x/bt.ballspacing)*bt.ballspacing, np.floor(y/bt.ballspacing)*bt.ballspacing, 'ro', markersize=6, markerfacecolor='none')
# Add the new positions
plt.plot(xnew, ynew, 'go', markersize = 8, markerfacecolor='none')


# Overaly the coarse grid lines
#Spacing between each line
intervals = bt.ballspacing
loc = plticker.MultipleLocator(base=intervals)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.grid(which='major', axis='both', linestyle='-')

plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/chessboard.png')


# # ball labels
# bnumbers = np.char.mod('%d', np.arange(bt.nballs))
# for x,y,s in zip(pos[0, tstop, :], pos[1, tstop, :], bnumbers):
#     plt.text(x, y, s, color='red', clip_on=True)
