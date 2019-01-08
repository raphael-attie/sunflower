import os, glob
import matplotlib
matplotlib.use('macosx')
#matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import balltracking.mballtrack as mblt
from skimage.exposure import rescale_intensity
from datetime import datetime
import pickle
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable

DTYPE = np.float32


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def preprep_function(datajz):
    datajzn = (datajz - datajz.mean()) / datajz.std()
    datajzn *= 100
    return datajzn


def prep_function(datajz):
    """
    Turn the current jz image into a trackable surface. Mean normalization, then shift upward so that the minimum is at
    the zero surface level, then invert and shift upward so that the max is at zero surface level.
    :param datajz:
    :return:
    """

    datajzn = (datajz - datajz.mean()) / datajz.std()
    #datajzn *= 100
    datajzn2 = np.abs(datajzn)

    return datajzn2





datadir = '/Users/rattie/Data/Lars/'
datafiles = sorted(glob.glob(os.path.join(datadir, '10degree*.npz')))
outputdir = os.path.join(datadir, 'mballtrack')

mbt_dict = {"nt":2,
            "rs":8,
            "am":0.5,
            "dp":0.3,
            "td":0.5,
            "prep_function":prep_function,
            "ballspacing":8,
            "intsteps":20,
            "local_min":True,
            "mag_thresh":[0, 30],
            "noise_level":0,
            "track_emergence":False,
            "emergence_box":10,
            "datafiles":datafiles}


### Start processing
mbt = mblt.mballtrack_main_positive(**mbt_dict)


# ### Get a sample
image = mblt.load_data(datafiles, 0)
image2 = (image - image.mean()) / image.std()
data = prep_function(image)





### Visualize

# def plot_step(step, marker='.', ms=4, color='cyan'):
#     mbt = mblt.mballtrack_main_positive(**mbt_dict)
#     maskp = mbt.ballpos[0, :, step] > 0
#     pos = mbt.ballpos[:, maskp, step]
#     ball_nbs = np.arange(mbt.nballs_max)[maskp]
#     tags = ['{:d}'.format(b) for b in ball_nbs]
#     ax1.plot(pos[0, :], pos[1, :], marker=marker, ms=ms, color=color, ls='none', markerfacecolor='none')
#     for i, tag in enumerate(tags):
#         # text coordinate for that tag from ball position
#         tagx, tagy = pos[0, i], pos[1, i]
#         ax1.text(tagx+1, tagy+1, tags[i], color=color, fontsize=8)


def plot_intsteps(marker='.', ms=4, color='cyan'):

    mbt = mblt.MBT(**mbt_dict)
    mbt.track_start_intermediate()

    pos = mbt.ballpos_inter[:, 0:mbt.nballs, :]
    ball_nbs = np.arange(mbt.nballs)
    tags = ['{:d}'.format(b) for b in ball_nbs]
    #ax1.plot(pos[0, :, :], pos[1, :, :], marker=marker, ms=ms, color=color, ls='none', markerfacecolor='none')
    ax1.plot(pos[0, :, -1], pos[1, :, -1], marker='d', ms=4, color='red', ls='none', markerfacecolor='none', label='final position')
    # for i, tag in enumerate(tags):
    #     # text coordinate for that tag from ball position
    #     tagx, tagy = pos[0, i], pos[1, i]
    #     ax1.text(tagx+1, tagy+1, tags[i], color=color, fontsize=8)
    return pos


# Test with different integration steps, plot the last step of each trial
fig = plt.figure(figsize=(10, 10))
dmax = data.max()
dmin = 0
fac = 0.5
ax1 = plt.subplot(111)
im1 = ax1.imshow(data, vmin=dmin, vmax=fac*dmax, cmap='inferno', origin='lower', interpolation='nearest')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im1, cax=cax)
ax1.plot(mbt.xstart, mbt.ystart, marker='+', ms=8, color='cyan', ls='none', label='Initial position')
tags = ['{:d}'.format(b) for b in range(mbt.nballs)]
for i, tag in enumerate(tags):
    tagx, tagy = mbt.xstart[i], mbt.ystart[i]
    ax1.text(tagx+1, tagy+1, tags[i], color='cyan', fontsize = 8)
# ax1.plot(pos[0, maskp], pos[1, maskp], marker='+', ms=2, color='cyan', ls='none')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Balls center at frame 0, nb of integration steps: {:d}, r={:d} px'.format(mbt_dict['intsteps'], mbt_dict['rs']))


mbt_dict['intsteps'] = 20
pos = plot_intsteps(ms=8)
ax1.legend()

fig.tight_layout()

#plot_step(0, color='cyan')
plt.savefig('/Users/rattie/Data/Lars/mballtrack/integration_20steps_rs_{:d}px.png'.format(mbt_dict['rs']), dpi=150)


# mbt_dict['intsteps'] = 40
# pos = plot_intsteps(ms=4, color='orange')
#
# plot_step(1, color='green', marker='o')


# fig = plt.figure(figsize=(18, 8))
#
# ax1 = plt.subplot(131)
# im1 = ax1.imshow(data, vmin=range_minmax[0], vmax=range_minmax[1], cmap='gray', origin='lower', interpolation='nearest')
# line1p, = plt.plot(mbt_p.xstart, mbt_p.ystart, marker='.', ms=2, color='red', ls='none')
# line1n, = plt.plot(mbt_n.xstart, mbt_n.ystart, marker='.', ms=2, color='cyan', ls='none')
#
# ax1.set_xlabel('Lambert cyl. X')
# ax1.set_ylabel('Lambert cyl. Y')
# ax1.set_title('Tracked local extrema at frame 0')
#
# ax2 = plt.subplot(132)
# #im2 = plt.imshow(data_borders_rgb, origin='lower', interpolation='nearest')
# im2 = plt.imshow(np.zeros([*data.shape, 3]), origin='lower', interpolation='nearest')
# line2p, = plt.plot([], [], marker='.', ms=2, color='red', ls='none')
# line2n, = plt.plot([], [], marker='.', ms=2, color='cyan', ls='none')
# ax2.set_xlabel('Lambert cyl. X')
# ax2.set_ylabel('Lambert cyl. Y')
# ax2.set_title('Boundaries of tracked fragments')
#
# ax3 = plt.subplot(133)
# cmap = custom_cmap(mbt_p.nballs + mbt_n.nballs)
# #im3 = plt.imshow(ws_labels, cmap=cmap, vmin=-1, vmax=mbt_p.nballs + mbt_n.nballs, interpolation='nearest', origin='lower')
# im3 = plt.imshow(np.zeros(data.shape), cmap=cmap, vmin=-1, vmax=mbt_p.nballs + mbt_n.nballs, interpolation='nearest', origin='lower')
#
# ax3.set_xlabel('Lambert cyl. X')
# ax3.set_ylabel('Lambert cyl. Y')
# ax3.set_title('Tracked fragments with ball-based color labeling')
# fig.tight_layout()
#
#
# # i = 1816
# # update_fig(i)