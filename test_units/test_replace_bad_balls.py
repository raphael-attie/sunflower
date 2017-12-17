from importlib import reload
import numpy as np
import matplotlib
matplotlib.use('macosx')
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as plticker
from matplotlib import cm


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
# Nb of steps (intermediate steps are ignored, integration on a single frame instead)
nt = 15
# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2
# Get a BT instance with the above parameters
bt = blt.BT(file, image.shape, nt, rs, dp, sigma_factor=sigma_factor)
# Initialize ball positions
bt.initialize(image)
# Get the initial data surface, as calculated in bt.initialize
surface, _, _ = blt.prep_data2(image, sigma_factor = sigma_factor)


# integrate motion over some time steps. Enough to have overpopulated cells (bad balls).
#pos, _, _ = [np.array(v).squeeze().swapaxes(0,1) for v in zip(*[blt.integrate_motion(bt.pos, bt.vel, bt, surface, return_copies=True) for _ in range(nt)])]
pos, _, _ = [np.array(v).squeeze().swapaxes(0,1) for v in zip(*[blt.integrate_motion(bt, surface, return_copies=True) for _ in range(nt)])]


# Mask of the bad balls at a given time tstop
tstop = 14
blt.get_bad_balls(bt)
# Get [x, y] coordinates of these bad balls at time = tstop
xbad = pos[0, tstop, bt.bad_balls_mask]
ybad = pos[1, tstop, bt.bad_balls_mask]

# Initial chess board
chess_board_init = blt.fill_coarse_grid(bt, bt.xstart, bt.ystart)
# Fill the coarse grid at the cells that map to the bad positions
chess_board_all = blt.fill_coarse_grid(bt, pos[0, tstop, :], pos[1, tstop, :])
chess_board_bad = blt.fill_coarse_grid(bt, xbad, ybad)
# Replace bad balls and get the new positions that fills the empty coarse grid cells
xnew, ynew = blt.replace_bad_balls(surface, bt)
# Check new chess board
chess_board_relocation = blt.fill_coarse_grid(bt, xnew, ynew)

# Ignore flagged balls to see new chessboard
validmask = bt.pos[0,:] != -1
valid_pos = bt.pos[:, validmask]
new_chess_board = blt.fill_coarse_grid(bt, valid_pos[0, :], valid_pos[1, :])

gmap = cm.gray_r.from_list('whatever', ('white', 'black'), N=6)

### Display ###
print_fig = True

fig = plt.figure(2, figsize=(12, 12))
ax = fig.add_subplot(1,1,1)

# Overlay initial positions
ax.plot(bt.xstart, bt.ystart, 'ro')
ax.axis([0, 60, 0, 60])
im = ax.imshow(chess_board_init, cmap=gmap, origin='lower', vmin=0, vmax=6, extent=(0,bt.nx-1, 0, bt.ny-1))

ax.set_title('Initial positions & initial chess board')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb1 = plt.colorbar(im, cax=cax)


# Overaly the coarse grid lines
#Spacing between each line
intervals = bt.ballspacing
loc = plticker.MultipleLocator(base=intervals)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.grid(which='major', axis='both', linestyle='-')

if print_fig:
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/replace_bad_balls_0.png')


# Overlay all other integrated positions
ax.plot(pos[0, :, :], pos[1, :, :], 'b.')
# Last position
ax.plot(pos[0, tstop, :], pos[1, tstop, :], 'c.', markersize=4)
ax.set_title('All positions untill tstop = %d'%tstop)

if print_fig:
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/replace_bad_balls_1.png')


# bad position
ax.plot(xbad, ybad, 'yo', markersize=6, markerfacecolor='none', mew=2)
#plt.plot(np.floor(x/bt.ballspacing)*bt.ballspacing, np.floor(y/bt.ballspacing)*bt.ballspacing, 'ro', markersize=6, markerfacecolor='none')

# Show the chess board with just the bad balls
im = ax.imshow(chess_board_all, cmap=gmap, origin='lower', vmin=0, vmax=5, extent=(0,bt.nx-1, 0, bt.ny-1))
ax.set_title('Chessboard & overpopulation')


if print_fig:
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/replace_bad_balls_2.png')

# Add the new positions
ax.plot(xnew, ynew, 'o', markeredgecolor='xkcd:green', markersize = 8, markerfacecolor='none', mew=2)
ax.set_title('New positions')

if print_fig:
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/replace_bad_balls_3.png')

## With new chessboard

# Show the chess board with just the bad balls
im = ax.imshow(chess_board_relocation, cmap=gmap, origin='lower', vmin=0, vmax=5, extent=(0,bt.nx-1, 0, bt.ny-1))
ax.set_title('New positions & associated chessboard')


if print_fig:
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/replace_bad_balls_4.png')

# Show the chess board with just the bad balls
im = ax.imshow(new_chess_board, cmap=gmap, origin='lower', vmin=0, vmax=5, extent=(0,bt.nx-1, 0, bt.ny-1))
ax.set_title('All positions, updated chessboard')


if print_fig:
    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/replace_bad_balls_5.png')


# # ball labels
# bnumbers = np.char.mod('%d', np.arange(bt.nballs))
# for x,y,s in zip(pos[0, tstop, :], pos[1, tstop, :], bnumbers):
#     plt.text(x, y, s, color='red', clip_on=True)
