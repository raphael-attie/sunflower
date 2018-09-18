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




datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02_Aimee/mtrack_20100901_120034_TAI_20100902_120034_TAI_LambertCylindrical_magnetogram.fits'


mbt_dict = {"nt":50,
            "rs":3,
            "am":0.5,
            "dp":0.3,
            "td":2,
            "ballspacing":4,
            "intsteps":10,
            "mag_thresh":50,
            "mag_thresh_sunspots":800,
            "noise_level":25,
            "track_emergence":True,
            "datafiles":datafile}

mbtp = mblt.MBT(polarity=1, **mbt_dict)
mbtp.track_start_intermediate()

# mbtp2 = mblt.MBT(polarity=1, **mbt_dict)
# mbtp2.track_all_frames()

### Get a sample
data = fitstools.fitsread(datafile, tslice=0).astype(DTYPE)
range_minmax = (-200,200)


### Visualize
surface = mbtp.surface

ball = mblt.get_balls_at(mbtp.xstart, mbtp.ystart, 285, 211) # 129
xt, yt, zt = mbtp.xstart, mbtp.ystart, mbtp.zstart

datamax = 200
fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(111)
im1 = ax1.imshow(data, vmin=-datamax, vmax=datamax, cmap='gray', origin='lower', interpolation='nearest')
line1p, = plt.plot(xt, yt, marker='.', ms=2, color='red', ls='none')




fxp, fyp, fzp = mbtp.force_inter[0][0:3,ball]
vxp, vyp = mbtp.vel_inter[0, ball, 0], mbtp.vel_inter[1, ball, 0]

shaft_width = 0.1
plt.quiver(xt, yt, fxp, fyp, units='xy', scale=0.3, width=shaft_width, headwidth=0.5 / shaft_width,
           headlength=0.5 / shaft_width, color='blue')

plt.quiver(xt, yt, vxp, vyp, units='xy', scale=0.3, width=shaft_width, headwidth=0.5 / shaft_width,
           headlength=0.5 / shaft_width, color='green')

xt2, yt2, zt2 = mbtp.ballpos_inter[0:3, ball, 1]
plt.plot(xt2, yt2, marker='.', ms=2, color='red', ls='none')

xtf, ytf, ztf = mbtp.ballpos_inter[0:3, ball, -1]
plt.plot(xtf, ytf, marker='o', ms=4, color='red', ls='none', markerfacecolor='none')

#plt.axis([xt-10, xt+10, yt-10, yt+10])

ax1.set_xlabel('Lambert cyl. X')
ax1.set_ylabel('Lambert cyl. Y')
ax1.set_title('Tracked local extrema at frame 0')


framenb = 1
surface = mblt.prep_data(mblt.load_data(mbtp2, framenb))
xt, yt, zt = mbtp2.ballpos[:, ball, framenb]


fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(111)
im1 = ax1.imshow(surface, cmap='gray', origin='lower', interpolation='nearest')
line1p, = plt.plot(xt, yt, marker='.', ms=2, color='red', ls='none')

ax1.set_xlabel('Lambert cyl. X')
ax1.set_ylabel('Lambert cyl. Y')
ax1.set_title('Tracked local extrema at frame %d'%framenb)

