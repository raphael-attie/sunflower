from importlib import reload
import matplotlib
matplotlib.use('macosx')
import numpy as np
import balltracking.balltrack as blt
import fitstools
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as plticker

plt.ioff()



datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'

# Load 1st image
image = fitstools.fitsread(datafile, tslice=0).astype(np.float32)
header = fitstools.fitsheader(datafile)
### Ball parameters
# Use 80 frames (1 hr)
nt = 80
# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2
# Get a BT instance with the above parameters
bt_tf = blt.BT(image.shape, nt, rs, dp, sigma_factor=sigma_factor, mode='top', direction='forward', datafiles=datafile)
bt_tb = blt.BT(image.shape, nt, rs, dp, sigma_factor=sigma_factor, mode='top', direction='backward', datafiles=datafile)


startTime = datetime.now()

blt.track_all_frames(bt_tf)
blt.track_all_frames(bt_tb)
# Flip the time axis of the backward tracking
# ballpos dimensions: [xyz, # balls, time]
ballposR = np.flip(bt_tb.ballpos, 2)
# Concatenate the above on balls axis. I'm simply adding the balls of the backward tracking to the forward tracking.
# This makes a finer sampling of the data surface and reduces the number of empty bins
ballpos = np.concatenate((bt_tf.ballpos, ballposR.copy()), axis=1)
print( " --- %s seconds" %(datetime.now() - startTime))


ballpos_x = bt_tf.ballpos[0, :, :].copy()
ballpos_y = bt_tf.ballpos[1, :, :].copy()

bt = bt_tf

gmap = cm.gray_r.from_list('whatever', ('white', 'black'), N=6)

fig = plt.figure(2, figsize=(20, 11))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_xlim(xmin=50, xmax=150)
ax1.set_ylim(ymin=50, ymax=150)
ax2.set_xlim(xmin=350, xmax=450)
ax2.set_ylim(ymin=50, ymax=150)
# Overaly the coarse grid lines
# Spacing between each line
# ax1
intervals = bt.ballspacing
locx1 = plticker.MultipleLocator(base=intervals)
locy1 = plticker.MultipleLocator(base=intervals)
ax1.xaxis.set_major_locator(locx1)
ax1.yaxis.set_major_locator(locy1)
ax1.grid(which='major', axis='both', linestyle='-')
ax1.tick_params(labelsize=8)
#ax2
locx2 = plticker.MultipleLocator(base=intervals)
locy2 = plticker.MultipleLocator(base=intervals)
ax2.xaxis.set_major_locator(locx2)
ax2.yaxis.set_major_locator(locy2)
ax2.grid(which='major', axis='both', linestyle='-')
ax2.tick_params(labelsize=8)

plt.tight_layout()


for n in range(nt):
    # Get the time series of chess boards.
    validmask = ballpos_x[:,n] != -1
    valid_pos_x = ballpos_x[validmask, n]
    valid_pos_y = ballpos_y[validmask, n]
    new_chess_board = blt.fill_coarse_grid(bt_tf, valid_pos_x, valid_pos_y)


    ax1.imshow(new_chess_board, cmap=gmap, origin='lower', vmin=0, vmax=6, extent=(0, bt.nx, 0, bt.ny))
    ax1.plot(valid_pos_x, valid_pos_y, 'r.', markersize=4)
    ax1.set_title('All positions, updated chessboard at frame=%d (Bottom Left)' % n)

    ax2.imshow(new_chess_board, cmap=gmap, origin='lower', vmin=0, vmax=6, extent=(0, bt.nx, 0, bt.ny))
    ax2.plot(valid_pos_x, valid_pos_y, 'r.', markersize=4)
    ax2.set_title('All positions, updated chessboard at frame=%d (Bottom Right)' % n)

    plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/tracks/chessboard_frame_%d.png' %n)


# bad_mask = ballpos_x == -1
# ballpos_x[bad_mask] = np.nan
# ballpos_y[bad_mask] = np.nan
#
# ball = np.where( np.logical_and(np.logical_and(ballpos_x[:, 0] >= 0, ballpos_x[:, 0] <= 60), np.logical_and(ballpos_y[:, 0] >= 0, ballpos_y[:, 0] <= 60)) )[0]
#
# # Concatenate initial position arrays and final position at frame 0
# xinit = np.stack((bt_tf.xstart.flatten(), ballpos_x[:,0]), axis=1)
# yinit = np.stack((bt_tf.ystart.flatten(), ballpos_y[:,0]), axis=1)
#
# ### Display ###
# fig = plt.figure(2, figsize=(10, 10))
#
# plt.axis([0, 60, 0, 60])
# plt.grid(which='both')
# plt.plot(xinit[:,0], yinit[:,0], 'o', markersize=5, mew=1, markeredgecolor='blue', markerfacecolor='blue')
# plt.plot(xinit.T, yinit.T, 'b-', marker='.', markersize=2, mew=1, markeredgecolor='black', markerfacecolor='black')
# plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/tracks/positions_frame_0.png')
#
# # Plot positions at the end of each frame
# for i in range(9):
#     plt.plot(ballpos_x[ball, 0:i+1].T, ballpos_y[ball, 0:i+1].T, 'r-', marker='.', markersize=2, mew=1, markeredgecolor='black', markerfacecolor='black')
#     plt.savefig('/Users/rattie/Dev/sdo_tracking_framework/figures/tracks/positions_frame_%d.png'%i)
#
# Make flow maps
dims = (bt_tf.ny, bt_tf.nx)
trange = np.arange(0, nt)
fwhm = 15

startTime = datetime.now()

vxf, vyf, wplanef = blt.make_velocity_from_tracks(bt_tf.ballpos, dims, trange, fwhm)
vxb, vyb, wplaneb = blt.make_velocity_from_tracks(bt_tb.ballpos, dims, trange, fwhm)
vx, vy, wplane = blt.make_velocity_from_tracks(ballpos, dims, trange, fwhm)


print( " --- %s seconds" %(datetime.now() - startTime))

vnormf = np.sqrt(vxf**2 + vyf**2)
vnormb = np.sqrt(vxb**2 + vyb**2)
vnorm = np.sqrt(vx**2 + vy**2)

plt.figure(0, figsize=(18,10))
plt.subplot(241)
plt.imshow(vnormf, origin='lower')
plt.title('Forward')
plt.subplot(242)
plt.imshow(vnormb, origin='lower')
plt.title('Backward')
plt.subplot(243)
plt.imshow((vnormf + vnormb)/2, origin='lower')
plt.title('Average')
plt.subplot(244)
plt.imshow(vnorm, origin='lower')
plt.title('Concatenation')

plt.subplot(245)
plt.imshow(wplanef, origin='lower', vmin = wplanef.min(), vmax = wplanef.max())
plt.subplot(246)
plt.imshow(wplaneb, origin='lower', vmin = wplanef.min(), vmax = wplanef.max())
plt.subplot(247)
plt.imshow(wplanef + wplaneb, origin='lower', vmin = wplanef.min(), vmax = wplanef.max()*2)
plt.subplot(248)
plt.imshow(wplane, origin='lower', vmin = wplanef.min(), vmax = wplanef.max()*2)

plt.tight_layout()


x, y = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))

step = 4

plt.figure(1, figsize=(18,10))
plt.subplot(231)
plt.quiver(x[::step, ::step], y[::step, ::step], vx[::step, ::step], vy[::step, ::step], units='xy', scale=0.005, width=0.5, headwidth=2, headlength=2)
plt.axis([60, 150, 80, 170])
plt.gca().set_aspect('equal')
plt.subplot(232)
plt.quiver(x[::step, ::step], y[::step, ::step], vx[::step, ::step], vy[::step, ::step], units='xy', scale=0.005, width=0.5, headwidth=3, headlength=2)
plt.axis([60, 150, 80, 170])
plt.gca().set_aspect('equal')
plt.subplot(233)
plt.quiver(x[::step, ::step], y[::step, ::step], vx[::step, ::step], vy[::step, ::step], units='xy', scale=0.005, width=0.5, headwidth=3, headlength=4)
plt.axis([60, 150, 80, 170])
plt.gca().set_aspect('equal')

plt.subplot(234)
plt.quiver(x[::step, ::step], y[::step, ::step], vx[::step, ::step], vy[::step, ::step], units='xy', scale=0.005, width=0.5, headwidth=3, headlength=5)
plt.axis([60, 150, 80, 170])
plt.gca().set_aspect('equal')
plt.subplot(235)
plt.quiver(x[::step, ::step], y[::step, ::step], vx[::step, ::step], vy[::step, ::step], units='xy', scale=0.005, width=1, headwidth=3, headlength=5, headaxislength=6)
plt.axis([60, 150, 80, 170])
plt.gca().set_aspect('equal')
plt.subplot(236)
plt.quiver(x[::step, ::step], y[::step, ::step], vx[::step, ::step], vy[::step, ::step], units='xy', scale=0.005, width=1, headwidth=3, headlength=5, headaxislength=8)
plt.axis([60, 150, 80, 170])
plt.gca().set_aspect('equal')


plt.tight_layout()
