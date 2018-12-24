import os, glob
import matplotlib
matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import balltracking.mballtrack as mblt

DTYPE = np.float32


def prep_function(datajz):
    """
    Turn the current jz image into a trackable surface. Mean normalization, then shift upward so that the minimum is at
    the zero surface level, then invert and shift upward so that the max is at zero surface level.
    :param datajz:
    :return:
    """

    datajzn = (datajz - datajz.mean()) / datajz.std()
    datajzn *= 100
    datajzn2 = np.abs(datajzn)

    return datajzn2


def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


datadir = '/Users/rattie/Data/Lars/'
datafiles = sorted(glob.glob(os.path.join(datadir, '10degree*.npz')))
outputdir = os.path.join(datadir, 'mballtrack')

mbt_dict = {"nt":2,
            "rs":4,
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


# Sphere graphical data
# Make data

u = np.linspace(0, 2 * np.pi, 25)
v = np.linspace(0, np.pi, 25)
x = mbt_dict['rs'] * np.outer(np.cos(u), np.sin(v))
y = mbt_dict['rs'] * np.outer(np.sin(u), np.sin(v))
z = mbt_dict['rs'] * np.outer(np.ones(np.size(u)), np.cos(v))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x,y,z, cmap='rainbow')
plt.tight_layout()


# ### Get a sample
image = mblt.load_data(datafiles, 0)
data = prep_function(image)
# Balltrack
mbt = mblt.MBT(**mbt_dict)
mbt.track_start_intermediate()


dmax = data.max()
dmin = 0
fac = 0.25
ball = 68


x0, x1 = 225, 240
y0, y1 = 190, 205
data2 = data[y0:y1, x0:x1]

xstart = mbt.xstart[ball] - x0
ystart = mbt.ystart[ball] - y0
posx = mbt.ballpos_inter[0, ball, :] -x0
posy = mbt.ballpos_inter[1, ball, :] - y0
posz = mbt.ballpos_inter[2, ball, :]

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(111)
im1 = ax1.imshow(data2, vmin=fac*dmin, vmax=fac*dmax, cmap='gray', origin='lower', interpolation='nearest')
ax1.plot(xstart, ystart, marker='+', ms=8, color='red', ls='none')
ax1.plot(posx[::4], posy[::4], marker='.', ms=8, color='cyan', ls='none', markerfacecolor='none')

circle1 = plt.Circle((xstart, ystart), mbt_dict['rs'], color='red', fill=False)
ax1.add_artist(circle1)

plt.tight_layout()


# # 3D mesh coordinates
# XX, YY = np.meshgrid(np.arange(data2.shape[1]), np.arange(data2.shape[0]))
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(XX, YY, data2, cmap='hot', alpha=0.5)
# ax.plot_surface(x + posx[0],y + posy[0], z + posz[0], cmap='rainbow')
# ax.set_zlim3d([-5, 20])
#
# #plt.tight_layout()
#
# set_aspect_equal_3d(ax)

