from importlib import reload
import multiprocessing
import time
from functools import partial
import matplotlib
matplotlib.use('macosx')
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


def plot_mballtrack(data, pos, watershed_labels, borders, nballs, i):

    plt.figure(figsize=(18, 8))

    plt.subplot(131)
    plt.imshow(data, cmap='gray', vmin=-100, vmax=100)
    #plt.colorbar()
    mask = pos[0, :] > 0
    plt.scatter(pos[0, mask], pos[1, mask],marker='.', s=2, color='red')
    plt.axis([0, data.shape[1], 0, data.shape[0]])

    plt.xlabel('Lambert cyl. X')
    plt.ylabel('Lambert cyl. Y')

    plt.subplot(132)

    #bkg_gray = np.full([*data.shape, 3], 220, dtype=np.uint8)
    bkg_gray = rescale_intensity(np.tile(data[..., np.newaxis], (1,1,3)), in_range=(-100,100), out_range=np.uint8).astype(np.uint8)

    borders_rgb = bkg_gray.copy()
    # Color positive borders as red
    borders_rgb[borders == 1, 0] = 255
    borders_rgb[borders == 1, 1] = 0
    borders_rgb[borders == 1, 2] = 0

    #borders_rgb = np.concatenate((borders_red, borders_green, bkg_blue), axis=2)
    #plt.imshow(data, cmap='gray', vmin=-100, vmax=100)
    #plt.imshow(borders, vmin=0, vmax=1, origin='lower', cmap='Blues')
    plt.imshow(borders_rgb, origin='lower')
    plt.scatter(pos[0, mask], pos[1, mask], marker='.', s=2, color='red')
    # plt.text(x + 1, y + 1, '%d' % target, fontsize=8)
    plt.axis([0, data.shape[1], 0, data.shape[0]])
    plt.title('Boundaries of tracked fragments')


    plt.subplot(133)
    cmap = custom_cmap(nballs)
    plt.imshow(watershed_labels, cmap=cmap, vmin=-1, vmax=nballs, interpolation='nearest', origin='lower')

    plt.axis([0, data.shape[1], 0, data.shape[0]])
    plt.title('Tracked fragments with ball-based color labeling)')
    plt.tight_layout()

    plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mballtrack/frame_%02d.png' % i)
    plt.close()



datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'

mbt_dict = {"nt":50,
            "rs":2,
            "am":1,
            "dp":0.3,
            "td":2,
            "intsteps":10,
            "mag_thresh":50,
            "mag_thresh_sunspots": 400,
            "noise_level":20,
            "track_emergence":True,
            "datafiles":datafile}

mbt_n = mblt.MBT(polarity=-1, **mbt_dict)
mbt_n.track_all_frames()

# Flux extraction by markers-based watershed
ws_list_n, markers_list_n, borders_list_n = mblt.watershed_series(mbt_dict['datafiles'], mbt_dict['nt'], mbt_dict['noise_level'], -1, mbt_n.ballpos.astype(np.int32))


for i in range(50):
    # Visualize tracking results at different time indices
    pos_n = mbt_n.ballpos[..., i]
    data = fitstools.fitsread(datafile, tslice=i).astype(np.int32).astype(DTYPE)

    labels = ws_list_n[i]
    borders = borders_list_n[i]
    borders = thin(borders)

    plot_mballtrack(data, pos_n, labels, borders, mbt_n.nballs, i)

