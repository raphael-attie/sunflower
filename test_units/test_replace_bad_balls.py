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
image   = fitsio.read(file).astype(np.float32)

### Ball parameters
# Nb of intermediate steps
nt = 15
# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2
# Get a BT instance with the above parameters
bt = blt.BT(image.shape, nt, rs, dp, sigma_factor=sigma_factor)
# Initialize ball positions
bt.initialize(image)
# Get the initial data surface, as calculated in bt.initialize
surface, _, _ = blt.prep_data2(image, sigma_factor = sigma_factor)


# integrate motion over some time steps. Enough to have overpopulated cells (bad balls).
pos, _, _ = [np.array(v).squeeze().swapaxes(0,1) for v in zip(*[blt.integrate_motion(bt.pos, bt.vel, bt, surface, return_copies=True) for _ in range(nt)])]


# Mask of the bad balls at a given time tstop
tstop = 14
bad_balls = blt.get_bad_balls(pos[:,tstop,:], bt)
# Get [x, y] coordinates of these bad balls at time = tstop
xbad = pos[0, tstop, bad_balls]
ybad = pos[1, tstop, bad_balls]
# Fill the coarse grid at the cells that map to the bad positions
chess_board = blt.fill_coarse_grid(bt, xbad, ybad)
# The above fills the chess_board with only bad balls
# TODO: Populate the chessboard with also all the balls. Show it maybe before the latter.


# Use a copy of the pos array, because relocating the bad balls overwrite that input.
pos2 = pos.copy()
# Replace bad balls and get the new positions that fills the empty coarse grid cells
xnew, ynew = blt.replace_bad_balls(pos2[:,tstop,:], surface, bt)

# Check new state of the chessboard
bad_balls_mask = blt.get_bad_balls(pos2[:,tstop,:], bt)
# Get [x, y] coordinates of these bad balls at time = tstop
xbad2 = pos2[0, tstop, bad_balls[np.logical_not(bad_balls_mask)]]
ybad2 = pos2[1, tstop, bad_balls]
# Fill the coarse grid at the cells that map to the bad positions
chess_board2 = blt.fill_coarse_grid(bt, xbad2, ybad2)




### Display ###
print_fig = True

fig = plt.figure(2, figsize=(10, 10))
ax = fig.add_subplot(1,1,1)

# Overlay initial positions
plt.plot(bt.xstart, bt.ystart, 'ro')
plt.axis([0, 60, 0, 60])
plt.imshow(bt.coarse_grid, cmap='gray_r', origin='lower', vmin=0, vmax=3, extent=(0,bt.nx-1, 0, bt.ny-1))

# Overaly the coarse grid lines
#Spacing between each line
intervals = bt.ballspacing
loc = plticker.MultipleLocator(base=intervals)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.grid(which='major', axis='both', linestyle='-')

plt.title('Initial positions & initial chess board')

if print_fig:
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/replace_bad_balls_0.png')

# Overlay all other integrated positions
plt.plot(pos[0, :, :], pos[1, :, :], 'b.')
# Last position
plt.plot(pos[0, tstop, :], pos[1, tstop, :], 'c.', markersize=4)
plt.title('All positions untill tstop = %d'%tstop)

if print_fig:
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/replace_bad_balls_1.png')

# Show the chess board with and setup proper axis
plt.imshow(chess_board, cmap='gray_r', origin='lower', vmin=0, vmax=3, extent=(0,bt.nx-1, 0, bt.ny-1))
plt.title('Chessboard of bad cells')

# bad position
plt.plot(xbad, ybad, 'yo', markersize=6, markerfacecolor='none', mew=2)
#plt.plot(np.floor(x/bt.ballspacing)*bt.ballspacing, np.floor(y/bt.ballspacing)*bt.ballspacing, 'ro', markersize=6, markerfacecolor='none')
plt.title('Bad positions & chess board')

if print_fig:
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/replace_bad_balls_2.png')

# Add the new positions
plt.plot(xnew, ynew, 'o', markeredgecolor='xkcd:green', markersize = 8, markerfacecolor='none', mew=2)
plt.title('New positions')

if print_fig:
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/replace_bad_balls_3.png')

## With new chessboard

# Show the chess board with and setup proper axis
plt.imshow(chess_board2, cmap='gray_r', origin='lower', vmin=0, vmax=3, extent=(0,bt.nx-1, 0, bt.ny-1))
plt.title('New chessboard of bad cells')

if print_fig:
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/replace_bad_balls_4.png')



# # ball labels
# bnumbers = np.char.mod('%d', np.arange(bt.nballs))
# for x,y,s in zip(pos[0, tstop, :], pos[1, tstop, :], bnumbers):
#     plt.text(x, y, s, color='red', clip_on=True)
