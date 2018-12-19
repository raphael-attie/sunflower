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

DTYPE = np.float32


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



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



# def update_fig(i):
#     #global mbt_p, mbt_n, ws_list_n, borders_list_n, borders_list_p, borders_list_n
#
#     # Visualize tracking results at different time indices
#     pos_p = mbt_p.ballpos[..., i]
#     pos_n = mbt_n.ballpos[..., i]
#     maskp = pos_p[0, :] > 0
#     maskn = pos_n[0, :] > 0
#
#     labels_p = ws_list_p[i]
#     borders_p = borders_list_p[i].astype(np.bool)
#     # borders_p = thin(borders_p)
#
#     labels_n = ws_list_n[i]
#     borders_n = borders_list_n[i]
#     # borders_n = thin(borders_n)
#
#     # Merge watershed labels and borders.
#     ws_labels, borders = mblt.merge_watershed(labels_p, borders_p, mbt_p.nballs, labels_n, borders_n)
#     ws_labels[0, 0] = mbt_p.nballs + mbt_n.nballs
#     #data = fitstools.fitsread(datafile, tslice=i).astype(DTYPE)
#     data = mblt.load_data(datafiles, i)
#     im1.set_array(data)
#
#
#     line1p.set_data(pos_p[0, maskp], pos_p[1, maskp])
#     line1n.set_data(pos_n[0, maskn], pos_n[1, maskn])
#     # xmaxp, ymaxp = mblt.get_local_extrema_ar(data, mblt.prep_data(data), 1, mbt_p.ballspacing, mbt_p.mag_thresh, mbt_p.mag_thresh_sunspots)
#     # xmaxn, ymaxn = mblt.get_local_extrema_ar(data, mblt.prep_data(data), -1, mbt_p.ballspacing, mbt_p.mag_thresh, mbt_p.mag_thresh_sunspots)
#     # xmax = np.concatenate((xmaxp, xmaxn))
#     # ymax = np.concatenate((ymaxp, ymaxn))
#     # line1max.set_data(xmax, ymax)
#
#     data_borders_rgb = make_data_borders_rgb(data, borders, range_minmax)
#
#     im2.set_array(data_borders_rgb)
#     line2p.set_data(pos_p[0, maskp], pos_p[1, maskp])
#     line2n.set_data(pos_n[0, maskn], pos_n[1, maskn])
#
#     im3.set_array(ws_labels)
#
#     ax1.set_title('Tracked local extrema at frame %d'%i)
#
#     #i+=1
#
#     # plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mballtrack/frame_%02d.png' % i)
#     # plt.close()
#     #line = [line1p, line1n, line1max, line2p, line2n]
#     line = [line1p, line1n, line2p, line2n]
#     return line



datadir = '/Users/rattie/Data/Lars/'
datafiles = sorted(glob.glob(os.path.join(datadir, '10degree*.npz')))
outputdir = os.path.join(datadir, 'mballtrack')

prep_function = partial(mblt.prep_data, sqrt_scale=False, invert=False, absolute=True)

mbt_dict = {"nt":2,
            "rs":8,
            "am":0.5,
            "dp":0.3,
            "td":0.5,
            "prep_function":prep_function,
            "ballspacing":8,
            "intsteps":20,
            "mag_thresh":100,
            "mag_thresh_sunspots":800, # not used at the moment
            "noise_level":25,
            "track_emergence":False,
            "emergence_box":10,
            "datafiles":datafiles}


### Start processing
mbt_p, mbt_n = mblt.mballtrack_main(**mbt_dict)
# Save results of tracking
fname = os.path.join(outputdir,'mbt_pn.pkl')
save_object([mbt_p, mbt_n], fname)
# Flux extraction by markers-based watershed
ws_list_p, markers_list_p, borders_list_p = mblt.watershed_series(mbt_dict['datafiles'], mbt_dict['nt'], mbt_dict['noise_level'], 1, mbt_p.ballpos.astype(np.int32))
ws_list_n, markers_list_n, borders_list_n = mblt.watershed_series(mbt_dict['datafiles'], mbt_dict['nt'], mbt_dict['noise_level'], -1, mbt_n.ballpos.astype(np.int32))

fname = os.path.join(outputdir, 'watershed_arrays2.npz')
np.savez(fname,
         ws_list_p=ws_list_p, markers_list_p=markers_list_p, borders_list_p=borders_list_p,
         ws_list_n=ws_list_n, markers_list_n=markers_list_n, borders_list_n=borders_list_n)

# load above saved file as:
# npzfile = np.load(fname)
# List content with: npzfile.files
# Get a specific array named 'a' with:
# a = npzfile['a']

# TODO: Watershed can be marked further with the local minima, which are appended to the balls positions, and removed afterwards.

#
# ### Get a sample
# # data = fitstools.fitsread(datafile, tslice=0).astype(DTYPE)
# data = mblt.load_data(datafiles, 0)
# range_minmax = (-200,200)
#
# ### Visualize
#
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