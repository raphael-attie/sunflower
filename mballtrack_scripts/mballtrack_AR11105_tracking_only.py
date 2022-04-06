from importlib import reload
import multiprocessing
import time
from functools import partial
import matplotlib
#matplotlib.use('macosx')
#matplotlib.use('qt5agg')
matplotlib.use('agg')
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
import fitsio

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


def update_fig(i):
    # Visualize tracking results at different time indices
    if i==0:
        pos_p = np.array([mbt_p.xstart, mbt_p.ystart]).astype(np.int16)
        pos_n = np.array([mbt_n.xstart, mbt_n.ystart]).astype(np.int16)
        data = fitstools.fitsread(datafile, tslice=0).astype(DTYPE)
    else:
        data = fitstools.fitsread(datafile, tslice=i-1).astype(DTYPE)
        pos_p = np.round(mbt_p.ballpos[..., i-1]).astype(np.int16)
        pos_n = np.round(mbt_n.ballpos[..., i-1]).astype(np.int16)

    maskp = pos_p[0, :] > 0
    maskn = pos_n[0, :] > 0

    im1.set_array(data)

    line1p.set_data(pos_p[0, maskp], pos_p[1, maskp])
    line1n.set_data(pos_n[0, maskn], pos_n[1, maskn])

    ax1.set_title('Tracked local extrema at frame %d'%i)

    line = [line1p, line1n]
    return line


def init():
    im1.set_data(data)
    line1p.set_data(np.ma.array(np.arange(10), mask=True), np.ma.array(np.arange(10), mask=True))
    line1n.set_data(np.ma.array(np.arange(10), mask=True), np.ma.array(np.arange(10), mask=True))
    line = [line1p, line1n]
    return line


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02_Aimee/mtrack_20100901_120034_TAI_20100902_120034_TAI_LambertCylindrical_magnetogram.fits'

mbt_dict = {"nt":20,
            "rs":2,
            "am":0.5,
            "dp":0.3,
            "td_":1,
            "ballspacing":4,
            "intsteps":10,
            "mag_thresh":100,
            "mag_thresh_sunspots":800,
            "noise_level":25,
            "track_emergence":True,
            "emergence_box":10,
            "datafiles":datafile}


### Start processing

start_time = datetime.now()

mbt_p, mbt_n = mblt.mballtrack_main(**mbt_dict)

fname = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02_Aimee/mbt_pn.pkl'
save_object([mbt_p, mbt_n], fname)

elapsed_time1 = datetime.now() - start_time
print("Tracking time: %d s"%elapsed_time1.total_seconds())


### Get a sample
data = fitstools.fitsread(datafile, tslice=0).astype(DTYPE)
range_minmax = (-200,200)

### Visualize

fig = plt.figure(figsize=(8, 8))

ax1 = plt.subplot(111)
im1 = ax1.imshow(data, vmin=range_minmax[0], vmax=range_minmax[1], cmap='gray', origin='lower', interpolation='nearest')
line1p, = plt.plot(mbt_p.xstart, mbt_p.ystart, marker='.', ms=2, color='red', ls='none')
line1n, = plt.plot(mbt_n.xstart, mbt_n.ystart, marker='.', ms=2, color='cyan', ls='none')

ax1.set_xlabel('Lambert cyl. X')
ax1.set_ylabel('Lambert cyl. Y')
ax1.set_title('Tracked local extrema at frame 0')

fig.tight_layout()

# i = 0
# update_fig(i)


interval = 100
ani = animation.FuncAnimation(fig, update_fig, interval=interval, frames=mbt_dict['nt']+1, blit=True, repeat=False, init_func=init)

fps = 10
ani.save('/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02_Aimee/movie_anim_fps%d_tracking_only_%d.mp4'%(fps, mbt_dict['nt']), fps=fps)

posp_start = np.array([mbt_p.xstart, mbt_p.ystart, mbt_p.zstart])
posn_start = np.array([mbt_n.xstart, mbt_n.ystart, mbt_n.zstart])

fitsio.write('/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02_Aimee/posp_start.fits', posp_start)


# TODO: Watershed can be marked further with the local minima, which are appended to the balls positions, and removed afterwards.