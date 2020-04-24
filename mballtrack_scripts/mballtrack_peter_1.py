import multiprocessing
import time
from functools import partial
import matplotlib
matplotlib.use('agg')
import numpy as np
import fitstools
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import balltracking.mballtrack as mblt
from skimage.exposure import rescale_intensity
import glob
from pathlib import PurePath
DTYPE = np.float32


def custom_cmap():

    # ## Custom colors => This must add a unique color for the background
    colors = plt.cm.Set1_r(np.linspace(0, 1, 9))
    # colors = plt.cm.Dark2(np.linspace(0, 1, 8))
    # # light gray color
    gray = np.array([[220, 220, 220, 255]]) / 255

    nballs = mbt_p.nballs + mbt_n.nballs
    cmap = matplotlib.colors.ListedColormap(colors, name='mycmap', N=nballs)
    colors2 = np.array([cmap(i) for i in range(nballs)])
    # Add unique background color
    colors2 = np.concatenate((gray, colors2), axis=0)
    cmap2 = matplotlib.colors.ListedColormap(colors2, name='mycmap2')

    return cmap2


def plot_mballtrack(args, datafiles, outputdir):
    pos_p, pos_n, watershed_labels, borders, i = args

    data = fitstools.fitsread(datafiles, tslice=i).astype(DTYPE)


    plt.figure(figsize=(18, 8))

    plt.subplot(131)
    plt.imshow(data, cmap='gray', vmin=-100, vmax=100)
    #plt.colorbar()
    maskp = pos_p[0, :] > 0
    maskn = pos_n[0, :] > 0
    plt.scatter(pos_p[0, maskp], pos_p[1, maskp],marker='.', s=2, color='red')
    plt.scatter(pos_n[0, maskn], pos_n[1, maskn],marker='.', s=2, color='cyan')

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
    # Color negative borders as cyan (accounting for color blindness)
    borders_rgb[borders == -1, 0] = 175
    borders_rgb[borders == -1, 1] = 238
    borders_rgb[borders == -1, 2] = 238

    #borders_rgb = np.concatenate((borders_red, borders_green, bkg_blue), axis=2)
    #plt.imshow(data, cmap='gray', vmin=-100, vmax=100)
    #plt.imshow(borders, vmin=0, vmax=1, origin='lower', cmap='Blues')
    plt.imshow(borders_rgb, origin='lower')
    plt.scatter(pos_p[0, maskp], pos_p[1, maskp], marker='.', s=2, color='red')
    # plt.text(x + 1, y + 1, '%d' % target, fontsize=8)
    plt.scatter(pos_n[0, maskn], pos_n[1, maskn], marker='.', s=2, color='cyan')
    plt.title('Boundaries of tracked fragments')


    plt.subplot(133)
    cmap = custom_cmap()
    plt.imshow(watershed_labels, cmap=cmap, vmin=-1, vmax=mbt_p.nballs + mbt_n.nballs, interpolation='nearest', origin='lower')
    plt.title('Tracked fragments with ball-based color labeling)')
    plt.tight_layout()

    plt.savefig(PurePath(outputdir, 'frame_%02d.png'.format(i)))
    plt.close()


if __name__ == "__main__":

    datadir = PurePath('/Users/rattie/Data/SDO/HMI/Peter/dataset1/magnetograms')
    datafiles = sorted(glob.glob(str(PurePath(datadir, '*.fits'))))

    mbt_dict = {"nt":50,
                "rs":2,
                "am":0.5,
                "dp":0.3,
                "td":1,
                "intsteps":10,
                "mag_thresh":50,
                "noise_level":20,
                "track_emergence":True,
                "datafiles":datafiles}

    mbt_p, mbt_n = mblt.mballtrack_main(**mbt_dict)
    # Convert to list of positions
    list_pos_p = [mbt_p.ballpos[..., i] for i in range(mbt_dict['nt'])]
    list_pos_n = [mbt_n.ballpos[..., i] for i in range(mbt_dict['nt'])]
    #
    # # Flux extraction by markers-based watershed
    ws_list_p, markers_list_p, borders_list_p = mblt.watershed_series(mbt_dict['datafiles'], mbt_dict['nt'], mbt_dict['noise_level'], 1, mbt_p.ballpos.astype(np.int32))
    ws_list_n, markers_list_n, borders_list_n = mblt.watershed_series(mbt_dict['datafiles'], mbt_dict['nt'], mbt_dict['noise_level'], -1, mbt_n.ballpos.astype(np.int32))

    # # Merge watershed within list comprehension.
    ws_labels, borders = zip(*[mblt.merge_watershed(labels_p, borders_p, mbt_p.nballs, labels_n, borders_n)
                          for labels_p, borders_p, labels_n, borders_n in zip(ws_list_p, borders_list_p,ws_list_n, borders_list_n)])



    # inputs = zip(list_pos_p, list_pos_n, ws_labels, borders, range(mbt_dict['nt']))
    #
    # # Merge watershed labels and borders.
    #
    # ## Plot parallel
    # pool = multiprocessing.Pool(processes=4)
    # pool.map(plot_mballtrack, inputs)

