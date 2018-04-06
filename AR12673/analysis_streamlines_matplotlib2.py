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
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

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
    lanes_rgba[..., 1] = 1
    lanes_rgba[..., 2] = 1
    lanes_rgba[..., 3] = lanes_norm
    return lanes_rgba


def create_plot(frame_number, ax, coords=None):

    mag, lanes_colored, _ = get_data(frame_number)

    im1 = ax.imshow(mag, vmin=vmin, vmax=vmax, cmap='gray', origin='lower')

    text = ax.text(8, 470, dtimes[frame_number].strftime('%x %X'), fontsize=10,
                    bbox=dict(boxstyle="square", fc='white', alpha=0.8))
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.

    im2 = ax.imshow(lanes_colored, origin='lower')

    ax.set_xticks(np.arange(0, mag.shape[1], 100))
    ax.set_xticks(np.arange(0, mag.shape[1], 100))
    ax.tick_params(
        axis='both',
        which='both',
        labelbottom=False,
        labelleft=False
    )


    if coords is not None:
        #ax.plot(coords[0], coords[1], color='red', marker='.', markerfacecolor='none', ms=64)
        #ax.plot(222, 294, color='red', marker='.', markerfacecolor='none', ms=60)
        circle = plt.Circle(coords, radius=20, alpha=.6, color='red', fill=False)
        ax.add_patch(circle)

    return im1, text, im2


def create_fig_31(frame_numbers, figsize, **kwargs):

    fig, axs = plt.subplots(4,1, figsize=figsize, gridspec_kw={"height_ratios":[1, 1, 1, 0.07]})

    _ = create_plot(frame_numbers[0], axs[0], **kwargs)
    _ = create_plot(frame_numbers[1], axs[1], **kwargs)
    im1,_,_ = create_plot(frame_numbers[2], axs[2], **kwargs)
    axs[2].set_xlabel('Lambert cylindrical X')
    fig.subplots_adjust(bottom=0.01, top=0.92, right=0.97, hspace=0.2)

    ip = InsetPosition(axs[0], [0, 1.05, 1, 0.05])
    axs[-1].set_axes_locator(ip)
    cb = fig.colorbar(im1, cax=axs[-1], ax=[axs[0], axs[1], axs[2]], orientation='horizontal')
    cb.set_ticks(np.arange(-800, 801, 400))
    axs[-1].xaxis.tick_top()
    axs[-1].xaxis.set_label_position('top')
    axs[-1].set_xlabel('Bz')

    return fig, axs, cb


def create_fig_41(frame_numbers, figsize, **kwargs):

    fig, axs = plt.subplots(5,1, figsize=figsize, gridspec_kw={"height_ratios":[1, 1, 1, 1, 0.07]})

    _ = create_plot(frame_numbers[0], axs[0], **kwargs)
    _ = create_plot(frame_numbers[1], axs[1], **kwargs)
    _ = create_plot(frame_numbers[2], axs[2], **kwargs)
    im1,_,_ = create_plot(frame_numbers[3], axs[3], **kwargs)
    axs[3].set_xlabel('Lambert cylindrical X')
    fig.subplots_adjust(bottom=0.01, top=0.92, right=0.97, hspace=0.2)

    ip = InsetPosition(axs[0], [0, 1.05, 1, 0.05])
    axs[-1].set_axes_locator(ip)
    cb = fig.colorbar(im1, cax=axs[-1], ax=[axs[0,1], axs[1]], orientation='horizontal')
    #cb.set_ticks(np.arange(-150, 151, 100))
    axs[-1].xaxis.tick_top()
    axs[-1].xaxis.set_label_position('top')
    axs[-1].set_xlabel('Bz')

    return fig, axs, cb


def create_fig_22(frame_numbers, figsize, **kwargs):

    fig, axs = plt.subplots(2,2, figsize=figsize)

    _ = create_plot(frame_numbers[0], axs[0, 0], **kwargs)
    _ = create_plot(frame_numbers[1], axs[0, 1], **kwargs)
    _ = create_plot(frame_numbers[2], axs[1, 0], **kwargs)
    im1,_,_ = create_plot(frame_numbers[3], axs[1, 1], **kwargs)

    axs[1, 0].set_xlabel('Lambert cylindrical X')
    axs[1, 1].set_xlabel('Lambert cylindrical X')
    axs[0, 0].set_ylabel('Lambert cylindrical Y')
    axs[1, 0].set_ylabel('Lambert cylindrical Y')


    fig.subplots_adjust(top=0.98, right=0.85, hspace = 0)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    ip = InsetPosition(axs[1,1], [1.05, 0, 0.05, 2.14])
    cbar_ax.set_axes_locator(ip)

    cb = fig.colorbar(im1, cax=cbar_ax, ax=[axs[0,0], axs[0,1], axs[1,0], axs[1,1]], orientation='vertical')
    cbar_ax.set_ylabel('Bz')

    return fig, axs, cb


def create_fig_32(frame_numbers, figsize, **kwargs):

    fig, axs = plt.subplots(3,2, figsize=figsize)

    im1,_,_ = create_plot(frame_numbers[0], axs[0, 0], **kwargs)
    _ = create_plot(frame_numbers[1], axs[0, 1], **kwargs)
    _ = create_plot(frame_numbers[2], axs[1, 0], **kwargs)
    _ = create_plot(frame_numbers[3], axs[1, 1], **kwargs)
    _ = create_plot(frame_numbers[4], axs[2, 0], **kwargs)
    _ = create_plot(frame_numbers[5], axs[2, 1], **kwargs)


    axs[0, 0].tick_params(labelleft=True)
    axs[1,0].tick_params(labelleft=True)
    axs[2, 0].tick_params(labelleft=True, labelbottom=True)
    axs[2, 1].tick_params(labelbottom=True)
    # axs[1, 0].set_yticks(np.arange(0, mag.shape[1], 100))
    # axs[2, 0].set_yticks(np.arange(0, mag.shape[1], 100))

    # axs[1, 0].set_xlabel('Lambert cylindrical X')
    # axs[1, 1].set_xlabel('Lambert cylindrical X')
    # axs[0, 0].set_ylabel('Lambert cylindrical Y')
    # axs[1, 0].set_ylabel('Lambert cylindrical Y')


    fig.subplots_adjust(top=0.98, right=0.85, hspace = 0.1)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    ip = InsetPosition(axs[2,1], [1.05, 0, 0.05, 3.2])
    cbar_ax.set_axes_locator(ip)

    cb = fig.colorbar(im1, cax=cbar_ax, ax=[axs[0,0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1]], orientation='vertical')
    cbar_ax.set_ylabel('Bz')

    return fig, axs, cb


# Get the JSOC colormap for magnetogram
cmap = get_mag_cmap()

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'
datafilem = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
tracking_dir ='/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/python_balltracking'
fig_dir = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/'
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


### Visualization - should loop over frame_numbers
#frame_numbers = [20, 27, 31, 50]
frame_numbers = [20, 27, 31]
mag, lanes_blue, v = get_data(frame_numbers[0])
vx, vy = v
vnorm = np.sqrt(vx**2 + vy**2)
x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))

vmin = -0.1 * np.max(np.abs(mag))
vmax = 0.1 * np.max(np.abs(mag))


# Plot 3 figures

figsize = (3.25, 9.125)
fig, axs, cb = create_fig_31(frame_numbers, figsize)


# plot 4 figures

frame_numbers = [20, 27, 31, 50, 75, 100]
mag, lanes_blue, v = get_data(frame_numbers[0])
vx, vy = v
vnorm = np.sqrt(vx**2 + vy**2)
x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))

vmin = -0.1 * np.max(np.abs(mag))
vmax = 0.1 * np.max(np.abs(mag))

figsize = (6.375, 5.5)
#fig, axs, cb = create_fig_41(frame_numbers, figsize)
fig, axs, cb = create_fig_22(frame_numbers, figsize)


plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/lanes_plot1_cyan.pdf', dpi=300)


####
frame_numbers = [20, 27, 31, 50, 75, 100]
mag, lanes_blue, v = get_data(frame_numbers[0])
vx, vy = v
vnorm = np.sqrt(vx**2 + vy**2)
x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))

vmin = -0.1 * np.max(np.abs(mag))
vmax = 0.1 * np.max(np.abs(mag))

figsize = (6.375, 7)
fig, axs, cb = create_fig_32(frame_numbers, figsize, coords=(225, 176))

plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/lanes_plot22_cyan.pdf', dpi=300)



### Quiver plots

# step = 8
#
#
# axs.quiver(x[::step, ::step], y[::step, ::step], vx[::step, ::step], vy[::step, ::step], vnorm[::step, ::step], units='xy', scale=0.006, width=shaft_width, headwidth=3/shaft_width, headlength=3/shaft_width, cmap='magma')
# axs.set_xlabel('Lambert cylindrical X [px]', fontsize=10)
# axs.set_ylabel('Lambert cylindrical Y [px]', fontsize=10)
# axs.tick_params(labelsize=10)
# fig.tight_layout()
#
# fname = os.path.join(fig_dir, 'lanes_plot_matplotlib.pdf')
# plt.savefig(fname, dpi=300)
