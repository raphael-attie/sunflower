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
from datetime import datetime
import pickle

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



def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02_Aimee/mtrack_20100901_120034_TAI_20100902_120034_TAI_LambertCylindrical_magnetogram.fits'

mbt_dict = {"nt":1920,
            "rs":2,
            "am":0.5,
            "dp":0.3,
            "td_":0.5,
            "ballspacing":4,
            "intsteps":20,
            "mag_thresh":100,
            "mag_thresh_sunspots":800,
            "noise_level":25,
            "track_emergence":True,
            "emergence_box":10,
            "datafiles":datafile}


### Start processing

start_time = datetime.now()

mbt_p, mbt_n = mblt.mballtrack_main(**mbt_dict)

elapsed_time1 = datetime.now() - start_time
print("Tracking time: %d s"%elapsed_time1.total_seconds())

fname = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02_Aimee/mbt_pn2.pkl'
save_object([mbt_p, mbt_n], fname)

# Flux extraction by markers-based watershed
start_time = datetime.now()

ws_list_p, markers_list_p, borders_list_p = mblt.watershed_series(mbt_dict['datafiles'], mbt_dict['nt'], mbt_dict['noise_level'], 1, mbt_p.ballpos.astype(np.int32))
ws_list_n, markers_list_n, borders_list_n = mblt.watershed_series(mbt_dict['datafiles'], mbt_dict['nt'], mbt_dict['noise_level'], -1, mbt_n.ballpos.astype(np.int32))

elapsed_time2 = datetime.now() - start_time
print("Segmentation time: %d s"%elapsed_time2.total_seconds())


print("total time: %d s"%(elapsed_time1 + elapsed_time2).total_seconds())

fname = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02_Aimee/watershed_arrays2.npz'
np.savez(fname,
         ws_list_p=ws_list_p, markers_list_p=markers_list_p, borders_list_p=borders_list_p,
         ws_list_n=ws_list_n, markers_list_n=markers_list_n, borders_list_n=borders_list_n)

# load above saved file as:
# npzfile = np.load(fname)
# List content with: npzfile.files
# Get a specific array named 'a' with:
# a = npzfile['a']

# TODO: Watershed can be marked further with the local minima, which are appended to the balls positions, and removed afterwards.