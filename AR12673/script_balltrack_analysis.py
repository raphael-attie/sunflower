import glob, os
import csv
import numpy as np
import matplotlib
matplotlib.use('macosx')
#matplotlib.use('agg')
import fitstools
import fitsio
import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import matplotlib.animation as animation
import balltracking.balltrack as blt

def get_mag_cmap():
    lut_file = '/Users/rattie/Dev/sdo_tracking_framework/graphics/HMI.MagColor.IDL_256.lut.txt'
    lut_reader = csv.reader(open(lut_file, newline=''), delimiter=' ')
    lut_str = [list(filter(None, row)) for row in lut_reader]
    lut = [[float(value) for value in row] for row in lut_str]
    cmap = matplotlib.colors.ListedColormap(lut)
    return cmap


def get_avg_data(file, tslice):
    samples = fitstools.fitsread(file, tslice=tslice).astype(np.float32)
    avg_data = samples.mean(axis=2)
    return avg_data


def get_vel(frame_number):
    vx = fitsio.read(vx_files[frame_number]).astype(np.float32)
    vy = fitsio.read(vy_files[frame_number]).astype(np.float32)
    v = [vx, vy]
    return v


def get_data(frame_number):

    mag = get_avg_data(datafilem, tslices[frame_number])
    v = get_vel(frame_number)
    lanes = blt.make_lanes(*v, nsteps, maxstep)
    lanes_blue = get_lanes_rgba(lanes)

    return mag, lanes_blue, v


def get_lanes_rgba(lanes_data):
    # Create an alpha channel from the lanes data.
    lanes_norm = Normalize(0, 0.5 * lanes_data.max(), clip=True)(lanes_data)
    lanes_rgba = np.ones(lanes_norm.shape + (4,))
    lanes_rgba[..., 0] = 0
    lanes_rgba[..., 1] = 0
    lanes_rgba[..., 2] = 1
    lanes_rgba[..., 3] = lanes_norm
    return lanes_rgba

def create_plot(frame_number, ax, coords=None):

    mag, lanes_blue, _ = get_data(frame_number)

    im1 = ax.imshow(mag, vmin=-0.1 * np.max(np.abs(mag)), vmax=0.1 * np.max(np.abs(mag)), cmap='gray', origin='lower')
    ax.set_xlabel('Lambert cylindrical X')
    ax.set_ylabel('Lambert cylindrical Y')
    text = ax.text(5, 495, 'Frame %d ' % frame_number + dtimes[frame_number].strftime('%x %X'), fontsize=12,
                    bbox=dict(boxstyle="square", fc='white', alpha=0.8))
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im1, cax=cax)
    im2 = ax.imshow(lanes_blue, origin='lower')

    if coords is not None:
        #ax.plot(coords[0], coords[1], color='red', marker='.', markerfacecolor='none', ms=64)
        #ax.plot(222, 294, color='red', marker='.', markerfacecolor='none', ms=60)
        circle = plt.Circle(coords, radius=20, alpha=.6, color='red', fill=False)
        ax.add_patch(circle)

    return im1, text, im2


def create_fig(frame_number):

    fig, axs = plt.subplots(figsize=(9, 8))
    im1, text, im2 = create_plot(frame_number, axs)
    fig.tight_layout()

    return fig, axs, im1, text, im2


def create_fig_22(frame_numbers, figsize, **kwargs):

    fig, axs = plt.subplots(2,2, figsize=figsize)
    _ = create_plot(frame_numbers[0], axs[0,0], **kwargs)
    _ = create_plot(frame_numbers[1], axs[0, 1], **kwargs)
    _ = create_plot(frame_numbers[2], axs[1, 0], **kwargs)
    _ = create_plot(frame_numbers[3], axs[1, 1], **kwargs)

    fig.tight_layout()
    return fig, axs


def update_fig(n):

    mag_n = get_avg_data(datafilem, tslices[n])
    lanes_n = fitsio.read(lanes_files[n]).astype(np.float32)
    lanes_blue = get_lanes_rgba(lanes_n)

    im1.set_array(mag_n)
    im2.set_array(lanes_blue)

    text.set_text('Frame %d: '%n + dtimes[n].strftime('%x %X'))

    return []

# Get the JSOC colormap for magnetogram
cmap = get_mag_cmap()

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'
datafilem = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
tracking_dir ='/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/python_balltracking'
vx_files = glob.glob(os.path.join(tracking_dir,'vx_[0-9]*.fits'))
vy_files = glob.glob(os.path.join(tracking_dir,'vy_[0-9]*.fits'))
lanes_files = glob.glob(os.path.join(tracking_dir,'lanes_[0-9]*.fits'))

nlanes = len(lanes_files)
### Lanes parameters
nsteps = 50
maxstep = 4


sample = fitstools.fitsread(datafile, tslice=0).astype(np.float32)
header = fitstools.fitsheader(datafile)
### time windows of the flow maps
nframes = int((3600*24*2 + 18*3600)/45) # 5280 frames
tspan = 80
tstep = 40
tcenters = np.arange(0, nframes-tstep, tstep)
tranges = [[tcenters[i], tcenters[i]+ tspan] for i in range(tcenters.size)]
# Build list of slices for extracting the corresponding magnetograms
tslices = [slice(trange[0], trange[1]) for trange in tranges]

### Build a list of datetime centered on each flow map
# Middle date of first map
dtime = datetime.datetime(year=2017, month=9, day=1, hour=0, minute=30, second=0)
dstep = datetime.timedelta(minutes=30)
dtimes = [dtime + i*dstep for i in range(len(tranges))]


### Visualization
fig, axs, im1, text, im2 = create_fig(100)

figsize = (11,10)

frame_numbers = [20, 27, 31, 50]
create_fig_22(frame_numbers, figsize, coords=(225, 176))
frame_numbers = [98, 100, 101, 112]
create_fig_22(frame_numbers, figsize, coords=(222, 294))


mag, lanes, v = get_data(frame_numbers[0])
vx, vy = v
vnorm = np.sqrt(vx**2 + vy**2)
x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))

# Quiver plots

step = 10

fig, axs = plt.subplots(figsize=(10,10))
axs.quiver(x[::step, ::step], y[::step, ::step], vx[::step, ::step], vy[::step, ::step], units='xy', scale=0.005, width=0.5, headwidth=4, headlength=4)

#plt.axis([60, 150, 80, 170])
plt.gca().set_aspect('equal')
plt.tight_layout()

# Streamplot
x, y = np.arange(vx.shape[1]), np.arange(vx.shape[0])

fig, axs = plt.subplots(figsize=(10,10))
axs.streamplot(x, y, vx, vy, density=5, linewidth=1) # cmap=plt.cm.inferno
#axs.plot(start_points[:,0], start_points[:,1], marker='.', ls='none', color='black', ms=2)
axs.set_aspect('equal')
plt.tight_layout()

fig, axs = plt.subplots(figsize=(10,10))
axs.streamplot(x, y, vx, vy, density=5, linewidth=1, maxlength=10) # cmap=plt.cm.inferno
#axs.plot(start_points[:,0], start_points[:,1], marker='.', ls='none', color='black', ms=2)
axs.set_aspect('equal')
plt.tight_layout()

step_start = 10
xstart, ystart = np.meshgrid(x, y)
xstart = xstart[::step_start, ::step_start].ravel()
ystart = ystart[::step_start, ::step_start].ravel()
start_points = [[coords[0], coords[1]] for coords in zip(xstart, ystart)]
stp = np.array(start_points)


fig, axs = plt.subplots(figsize=(10,10))
for i in range(len(start_points)):
    axs.streamplot(x, y, vx, vy, start_points=[start_points[i]], linewidth=1) # cmap=plt.cm.inferno
axs.plot(stp[:,0], stp[:,1], marker='.', ls='none', color='black', ms=1)
axs.set_aspect('equal')
plt.tight_layout()



fig, axs = plt.subplots(figsize=(10,10))
axs.streamplot(x, y, vx, vy, start_points = [[245, 265], [300, 100]]) # cmap=plt.cm.inferno
axs.plot(start_points[:,0], start_points[:,1], marker='.', ls='none', color='black', ms=2)
axs.set_aspect('equal')
plt.tight_layout()


# update_fig(47)
#
# ani = animation.FuncAnimation(fig, update_fig, interval=100, frames=nlanes, blit=True, repeat=False)

# fps=10
# ani.save('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/python_balltracking/movie_anim_fps%d.mp4'%fps, fps=fps)
