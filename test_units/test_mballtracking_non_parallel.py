from importlib import reload
import multiprocessing
import time
from functools import partial
import matplotlib
matplotlib.use('macosx')
#matplotlib.use('qt5agg')
#matplotlib.use('agg')
import numpy as np
import fitstools
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import balltracking.balltrack as blt
import balltracking.mballtrack as mblt
from skimage.feature import peak_local_max
from skimage.exposure import rescale_intensity
from skimage.morphology import thin
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.animation as animation
DTYPE = np.float32

def custom_cmap(nballs):

    # ## Custom colors => This must add a unique color for the background
    colors = plt.cm.Set1_r(np.linspace(0, 1, 9))
    # colors = plt.cm.Dark2(np.linspace(0, 1, 8))
    # # light gray color
    gray = np.array([[220, 220, 220, 255]]) / 255

    cmap = matplotlib.colors.ListedColormap(colors, name='mycmap', N=nballs)
    colors2 = np.array([cmap(i) for i in range(nballs)])
    # Add unique background color
    colors2 = np.concatenate((gray, colors2), axis=0)
    cmap2 = matplotlib.colors.ListedColormap(colors2, name='mycmap2')

    return cmap2

def make_data_borders_rgb(data, borders, in_range):

    bkg_gray = rescale_intensity(np.tile(data[..., np.newaxis], (1, 1, 3)), in_range=in_range,
                                 out_range=np.uint8).astype(np.uint8)

    borders_rgb = bkg_gray.copy()
    # Color positive borders as red
    borders_rgb[borders == 1, 0] = 255
    borders_rgb[borders == 1, 1] = 0
    borders_rgb[borders == 1, 2] = 0
    # Color negative borders as cyan (accounting for color blindness)
    borders_rgb[borders == -1, 0] = 175
    borders_rgb[borders == -1, 1] = 238
    borders_rgb[borders == -1, 2] = 238

    return borders_rgb



def update_fig(i):
    #global mbt_p, mbt_n, ws_list_n, borders_list_n, borders_list_p, borders_list_n

    # Visualize tracking results at different time indices
    pos_p = mbt_p.ballpos[..., i]
    pos_n = mbt_n.ballpos[..., i]

    labels_p = ws_list_p[i]
    borders_p = borders_list_p[i].astype(np.bool)
    # borders_p = thin(borders_p)

    labels_n = ws_list_n[i]
    borders_n = borders_list_n[i]
    # borders_n = thin(borders_n)

    # Merge watershed labels and borders.
    ws_labels, borders = mblt.merge_watershed(labels_p, borders_p, mbt_p.nballs, labels_n, borders_n)

    data = fitstools.fitsread(datafile, tslice=i).astype(DTYPE)
    im1.set_array(data)

    maskp = pos_p[0, :] > 0
    maskn = pos_n[0, :] > 0
    line1p.set_data(pos_p[0, maskp], pos_p[1, maskp])
    line1n.set_data(pos_n[0, maskn], pos_n[1, maskn])
    # xmaxp, ymaxp = mblt.get_local_extrema_ar(data, mblt.prep_data(data), 1, mbt_p.ballspacing, mbt_p.mag_thresh, mbt_p.mag_thresh_sunspots)
    # xmaxn, ymaxn = mblt.get_local_extrema_ar(data, mblt.prep_data(data), -1, mbt_p.ballspacing, mbt_p.mag_thresh, mbt_p.mag_thresh_sunspots)
    # xmax = np.concatenate((xmaxp, xmaxn))
    # ymax = np.concatenate((ymaxp, ymaxn))
    # line1max.set_data(xmax, ymax)

    data_borders_rgb = make_data_borders_rgb(data, borders, range_minmax)

    im2.set_array(data_borders_rgb)
    line2p.set_data(pos_p[0, maskp], pos_p[1, maskp])
    line2n.set_data(pos_n[0, maskn], pos_n[1, maskn])

    im3.set_array(ws_labels)

    ax1.set_title('Tracked local extrema at frame %d'%i)

    #i+=1

    # plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mballtrack/frame_%02d.png' % i)
    # plt.close()
    #line = [line1p, line1n, line1max, line2p, line2n]
    line = [line1p, line1n, line2p, line2n]
    return line

def init():
    im1.set_data(data)
    im2.set_data(np.zeros(data.shape))
    im3.set_data(np.zeros([*data.shape, 3]))
    line1p.set_data(np.ma.array(np.arange(10), mask=True), np.ma.array(np.arange(10), mask=True))
    line1n.set_data(np.ma.array(np.arange(10), mask=True), np.ma.array(np.arange(10), mask=True))
    line2p.set_data(np.ma.array(np.arange(10), mask=True), np.ma.array(np.arange(10), mask=True))
    line2n.set_data(np.ma.array(np.arange(10), mask=True), np.ma.array(np.arange(10), mask=True))
    #line1max.set_data(np.ma.array(np.arange(10), mask=True), np.ma.array(np.arange(10), mask=True))
    # line = [line1p, line1n, line1max, line2p, line2n]
    line = [line1p, line1n, line2p, line2n]
    return line



datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'

mbt_dict = {"nt":50,
            "rs":2,
            "am":0.5,
            "dp":0.3,
            "td":2,
            "ballspacing":8,
            "intsteps":10,
            "mag_thresh":50,
            "mag_thresh_sunspots":800,
            "noise_level":25,
            "track_emergence":True,
            "datafiles":datafile}

#mbt_p = mblt.MBT(polarity=1, **mbt_dict)

# dist = euclidean_distances(np.array([xstart[select], ystart[select]]).T)
# prows, pcols = np.where(dist > 2*mbt_p.ballspacing)
# # Each element in the above prows and pcols are indices in the distance array.

mbt_p, mbt_n = mblt.mballtrack_main(**mbt_dict)

# Flux extraction by markers-based watershed
ws_list_p, markers_list_p, borders_list_p = mblt.watershed_series(mbt_dict['datafiles'], mbt_dict['nt'], mbt_dict['noise_level'], 1, mbt_p.ballpos.astype(np.int32))
ws_list_n, markers_list_n, borders_list_n = mblt.watershed_series(mbt_dict['datafiles'], mbt_dict['nt'], mbt_dict['noise_level'], -1, mbt_n.ballpos.astype(np.int32))

data = fitstools.fitsread(datafile, tslice=0).astype(DTYPE)
ws_labels, borders = mblt.merge_watershed(ws_list_p[0], borders_list_p[0], mbt_p.nballs, ws_list_n[0], borders_list_n[0])

range_minmax = (-200,200)
data_borders_rgb = make_data_borders_rgb(data, borders, range_minmax)



fig = plt.figure(figsize=(18, 8))

ax1 = plt.subplot(131)
im1 = ax1.imshow(mbt_p.image, vmin=range_minmax[0], vmax=range_minmax[1], cmap='gray', origin='lower', interpolation='nearest')
line1p, = plt.plot(mbt_p.xstart, mbt_p.ystart, marker='.', ms=2, color='red', ls='none')
line1n, = plt.plot(mbt_n.xstart, mbt_n.ystart, marker='.', ms=2, color='cyan', ls='none')
# xmaxp, ymaxp = mblt.get_local_extrema_ar(data, mblt.prep_data(data), 1, mbt_p.ballspacing, mbt_p.mag_thresh,mbt_p.mag_thresh_sunspots)
# xmaxn, ymaxn = mblt.get_local_extrema_ar(data, mblt.prep_data(data), -1, mbt_p.ballspacing, mbt_p.mag_thresh,mbt_p.mag_thresh_sunspots)
# xmax = np.concatenate((xmaxp, xmaxn))
# ymax = np.concatenate((ymaxp, ymaxn))
# line1max, = plt.plot([], [], color='red', marker='.', ms=8, ls='none', markerfacecolor='none')

ax1.set_xlabel('Lambert cyl. X')
ax1.set_ylabel('Lambert cyl. Y')
ax1.set_title('Tracked local extrema at frame 0')

ax2 = plt.subplot(132)
#im2 = plt.imshow(data_borders_rgb, origin='lower', interpolation='nearest')
im2 = plt.imshow(np.zeros([*data.shape, 3]), origin='lower', interpolation='nearest')
line2p, = plt.plot([], [], marker='.', ms=2, color='red', ls='none')
line2n, = plt.plot([], [], marker='.', ms=2, color='cyan', ls='none')
ax2.set_xlabel('Lambert cyl. X')
ax2.set_ylabel('Lambert cyl. Y')
ax2.set_title('Boundaries of tracked fragments')

ax3 = plt.subplot(133)
cmap = custom_cmap(mbt_p.nballs + mbt_n.nballs)
#im3 = plt.imshow(ws_labels, cmap=cmap, vmin=-1, vmax=mbt_p.nballs + mbt_n.nballs, interpolation='nearest', origin='lower')
im3 = plt.imshow(np.zeros(data.shape), cmap=cmap, vmin=-1, vmax=mbt_p.nballs + mbt_n.nballs, interpolation='nearest', origin='lower')

ax3.set_xlabel('Lambert cyl. X')
ax3.set_ylabel('Lambert cyl. Y')
ax3.set_title('Tracked fragments with ball-based color labeling')
fig.tight_layout()

# # Create iterable for funcAnimation(). It must contain the series
i = 0
#for i in range(mbt_dict['nt']):
update_fig(i)



interval = 100
ani = animation.FuncAnimation(fig, update_fig, interval=interval, frames=50, blit=True, repeat=False, init_func=init)

plt.show()


# TODO: investigate shape_index to reject some local maxima that should be rejected
# TODO: or see usage of removing the small area, filling the hole, and relabel?
fps = 20
ani.save('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mballtrack/movie_anim_fps%d.mp4'%fps, fps=fps)
