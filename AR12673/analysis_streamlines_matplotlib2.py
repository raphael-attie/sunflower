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
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import matplotlib.animation as animation
import balltracking.balltrack as blt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import skimage.morphology
from scipy.ndimage.morphology import binary_dilation, binary_opening, distance_transform_edt
from scipy.ndimage.measurements import center_of_mass
import cv2

fs = 9

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


def get_lanes_rgba(lanes_data):
    # Create an alpha channel from the lanes data.
    lanes_norm = Normalize(0, 0.5 * lanes_data.max(), clip=True)(lanes_data)
    lanes_rgba = np.ones(lanes_norm.shape + (4,))
    lanes_rgba[..., 0] = 0
    lanes_rgba[..., 1] = 0
    lanes_rgba[..., 2] = 1
    lanes_rgba[..., 3] = lanes_norm
    return lanes_rgba


def get_data(frame_number, fov = None):

    mag = get_avg_data(datafilem, tslices[frame_number])
    cont = fitstools.fitsread(datafile, tslice=int((tslices[frame_number].stop - tslices[frame_number].start)/2)).astype(np.float32)
    vx, vy = get_vel(frame_number)
    lanes = blt.make_lanes(vx, vy, nsteps, maxstep)
    lanes_rgb = get_lanes_rgba(lanes)

    if fov is not None:
        mag = mag[fov[2]:fov[3], fov[0]:fov[1]]
        vx = vx[fov[2]:fov[3], fov[0]:fov[1]]
        vy = vy[fov[2]:fov[3], fov[0]:fov[1]]
        lanes_rgb = lanes_rgb[fov[2]:fov[3], fov[0]:fov[1]]
        cont = cont[fov[2]:fov[3], fov[0]:fov[1]]

    return mag, lanes_rgb, vx, vy, cont


def get_tranges_times(nframes, tavg, tstep):
    tcenters = np.arange(0, nframes - tstep, tstep)
    tranges = [[tcenters[i], tcenters[i] + tavg] for i in range(tcenters.size)]
    # Build list of slices for extracting the corresponding magnetograms
    tslices = [slice(trange[0], trange[1]) for trange in tranges]

    ### Build a list of datetime centered on each flow map
    # Middle date of first map
    dtime = datetime.datetime(year=2017, month=9, day=1, hour=0, minute=30, second=0)
    dstep = datetime.timedelta(minutes= tstep * 45/60)
    dtimes = [dtime + i * dstep for i in range(len(tranges))]
    return tslices, dtimes


def create_plot(frame_number, ax, fov=None, coords=None, continuum=False, quiver=False):

    mag, lanes_colored, vx, vy, cont = get_data(frame_number, fov)
    vx *= ms_unit
    vy *= ms_unit

    if continuum:
        im1 = ax.imshow(cont, cmap='gray', origin='lower')
    else:
        im1 = ax.imshow(mag, vmin=vmin, vmax=vmax, cmap='gray', origin='lower')

    text = ax.text(0.02, 0.95, dtimes[frame_number].strftime('%x %X'), fontsize=fs,
                   bbox=dict(boxstyle="square", fc='white', alpha=0.8),
                   transform=ax.transAxes)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.

    im2 = ax.imshow(lanes_colored, origin='lower')

    ax.set_xticks(np.arange(0, mag.shape[1], 50))
    ax.set_yticks(np.arange(0, mag.shape[1], 50))
    ax.tick_params(
        axis='both',
        which='both',
        labelbottom=False,
        labelleft=False,
        labelsize = fs
    )

    quiv = []
    if quiver:
        vnorm = np.sqrt(vx ** 2 + vy ** 2)
        x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))
        quiv = ax.quiver(x[::step, ::step], y[::step, ::step], vx[::step, ::step], vy[::step, ::step], vnorm[::step, ::step],
                  units='xy', scale=quiver_scale, width=shaft_width, headwidth=headwidth, headlength=headlength, cmap='Oranges')

    if coords is not None:
        #ax.plot(coords[0], coords[1], color='red', marker='.', markerfacecolor='none', ms=64)
        #ax.plot(222, 294, color='red', marker='.', markerfacecolor='none', ms=60)
        # circle = plt.Circle(coords, radius=40, alpha=.6, color='red', fill=False)
        # ax.add_patch(circle)
        rectangle = patches.Rectangle(coords, 100, 100, fill=False, color='yellow')
        ax.add_patch(rectangle)

    return im1, im2, quiv


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

    axs[0, 0].tick_params(labelleft=True)
    axs[1,0].tick_params(labelleft=True, labelbottom=True)
    axs[1, 1].tick_params(labelbottom=True)

    fig.subplots_adjust(top=0.98, right=0.85, hspace = 0, wspace=0.1)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.68])
    ip = InsetPosition(axs[1,1], [1.05, 0, 0.05, 2.09])
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

    axs[0, 0].set_ylabel('Lambert cyl. Y', fontsize=fs)
    axs[1, 0].set_ylabel('Lambert cyl. Y', fontsize=fs)
    axs[2, 0].set_ylabel('Lambert cyl. Y', fontsize=fs)
    axs[2, 0].set_xlabel('Lambert cyl. X', fontsize=fs)
    axs[2, 1].set_xlabel('Lambert cyl. X', fontsize=fs)


    fig.subplots_adjust(left=0.10, bottom=0.08, top=0.98, right=0.88, hspace = 0.05, wspace=0.055)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    ip = InsetPosition(axs[2,1], [1.05, 0, 0.05, 3.15])
    #ip = InsetPosition(axs[0, 0], [0, 1.15, 2.15, 0.05])
    cbar_ax.set_axes_locator(ip)

    cb = fig.colorbar(im1, cax=cbar_ax, ax=[axs[0,0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1]], orientation='vertical')
    cb.ax.tick_params(labelsize=fs)
    cbar_ax.set_title('Bz', fontsize=fs)

    return fig, axs, cb


# Get the JSOC colormap for magnetogram
cmap = get_mag_cmap()

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'
datafilem = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
tracking_dir ='/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/python_balltracking'
fig_dir = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/'

### Velocity field parameters
fwhm = 15
tavg = 160
tstep = 80

px_meter = 0.03 * 3.1415/180 * 6.957e8
ms_unit = px_meter / 45

### Lanes parameters
nsteps = 40
maxstep = 4

vx_files = glob.glob(os.path.join(tracking_dir,'vx_fwhm%d_tavg%d_[0-9]*.fits'%(fwhm, tavg)))
vy_files = glob.glob(os.path.join(tracking_dir,'vy_fwhm%d_tavg%d_[0-9]*.fits'%(fwhm, tavg)))

# lanes_files = glob.glob(os.path.join(tracking_dir,'lanes_[0-9]*.fits'))
# nlanes = len(lanes_files)

sample = fitstools.fitsread(datafile, tslice=0).astype(np.float32)
header = fitstools.fitsheader(datafile)
### time windows of the flow maps
nframes = int((3600*24*2 + 18*3600)/45) # 5280 frames
tslices, dtimes = get_tranges_times(nframes, tavg, tstep)

### Visualization - should loop over frame_numbers

vmin = -180
vmax = abs(vmin)

## Print time series of lanes over magnetograms. Single frames.
# fig, axs = plt.subplots(1,1, figsize=(8,8))
# for i in range(len(vx_files)):
#     create_plot(i, axs)
#     plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/lanes_fwhm%d_tavg%d_nsteps%d_plot_%d.png'%(fwhm, tavg, nsteps, i))




####
## For tavg = 80
#frame_numbers = [21, 27, 31, 65, 75, 100]
## For tavg = 160
frame_numbers = [3, 12, 14, 25, 39, 46]

fov = [50, 350, 50, 350]
mag, lanes_blue, vx, vy, cont = get_data(frame_numbers[0], fov=fov)


### Overview
# Quiver plot parameters
step = 10
quiver_scale = 45
shaft_width = 1.2
headwidth = 3
headlength = 5

figsize = (6.5, 6.5)

frame_number = int((tslices[frame_numbers[0]].stop - tslices[frame_numbers[0]].start)/2)
mag, lanes_colored, vx, vy, cont = get_data(frame_numbers[0])
vnorm = np.sqrt(vx**2 + vy**2)

sigma_factor = 2
surface, mean, sigma = blt.prep_data2(cont, sigma_factor)
ymin, xmin = np.unravel_index(np.argmin(cont, axis=None), cont.shape)
sfov = [0, 200, 0, 200]
scont = cont[sfov[2]:sfov[3], sfov[0]:sfov[1]]
sfactor = 5
threshold = scont.mean() - sfactor*scont.std()
# Mask based on standard deviation
spot_mask = cont < threshold
# Dilation
se = skimage.morphology.disk(6)
spot_mask_dilated = binary_dilation(spot_mask, structure=se)
spot_mask_dilated2 = binary_dilation(spot_mask_dilated, structure=skimage.morphology.disk(25))
spot_dist = distance_transform_edt(spot_mask_dilated)
ymax2, xmax2 = np.unravel_index(np.argmax(spot_dist, axis=None), spot_dist.shape)
ycom, xcom = center_of_mass(spot_mask_dilated)
circle1 = plt.Circle((xcom, ycom), radius=spot_dist.max(), alpha=.6, color='red', fill=False)
circle2 = plt.Circle((xcom, ycom), radius=spot_dist.max()+20, alpha=.6, color='green', fill=False)

xm, ym = np.meshgrid(np.arange(cont.shape[1]), np.arange(cont.shape[0]))
xm2 = xm - xcom
ym2 = ym - ycom
r = np.sqrt(xm2**2 + ym2**2)
rm1 = r < spot_dist.max()
rm2 = r < spot_dist.max() + 20
phi = np.arctan(ym2/xm2)

#
vnorm_pol = cv2.linearPolar(vnorm, (xcom, ycom), vnorm.shape[1], cv2.INTER_LANCZOS4 + cv2.WARP_FILL_OUTLIERS)
rho, phi = cv2.cartToPolar(xm2, ym2)
phi *= 180 / np.pi

# Continuum
fig, ax = plt.subplots(1,2, figsize=figsize)
im1 = ax[0].imshow(cont, cmap='gray', origin='lower')
im2 = ax[0].imshow(lanes_colored, origin='lower')
#plt.contour(cont, levels = [cont[0:200, 0:200].mean() - 4 * cont[0:200, 0:200].std()], colors='orange')
#ax.contour(spot_mask_dilated.astype(int), 1, colors='red')
#ax.contour(spot_mask_dilated2.astype(int), 1, colors='green')
ax[0].plot(xmax2, ymax2, 'r+')
ax[0].plot(xcom, ycom, 'g.', markerfacecolor='none')
# ax.add_patch(circle1)
# ax.add_patch(circle2)
ax[0].contour(rm1, 1, colors='red')
ax[0].contour(rm2, 1, colors='green')

ax[0].set_xlabel('x [px]')
ax[0].set_ylabel('y [px]')

ax[1].imshow(vnorm_pol, cmap='Oranges',origin='lower', extent=[rho.min(), rho.max(), phi.min(), phi.max()])

fig.tight_layout()

plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/cont_OVERVIEW_frame_%d.pdf'%(fwhm, tavg, nsteps, maxstep, frame_number), dpi=300)


# Magnetogram
fig, ax = plt.subplots(1,1, figsize=figsize)
im1, im2, quiv = create_plot(frame_numbers[0], ax, quiver=True)
ax.tick_params(labelbottom=True,  labelleft=True, labelsize = fs)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(quiv, cax=cax)
cax.set_ylabel('v (m/s)')
ax.set_xlabel('x [px]')
ax.set_ylabel('y [px]')
fig.tight_layout()

plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/mag_lanes_fwhm%d_tavg%d_nsteps%d_maxstep%d_OVERVIEW_frame_%d.pdf'%(fwhm, tavg, nsteps, maxstep, frame_numbers[0]), dpi=300)





# Circle coordinates
#coords=(225 - fov[0], 176 - fov[2])
# Rectangle coordinates
coords = (235-50 - fov[0], 190-50-fov[2])

vmin = -0.1 * np.max(np.abs(mag))
vmax = 0.1 * np.max(np.abs(mag))


figsize = (6.5, 9)
fig, axs, cb = create_fig_32(frame_numbers, figsize, fov=fov, coords=coords)

plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/lanes_fwhm%d_tavg%d_nsteps%d_maxstep%d_plot32_blue_1stf_%d.pdf'%(fwhm, tavg, nsteps, maxstep, frame_numbers[0]), dpi=300)



### Quiver plots
# vnorm = np.sqrt(vx**2 + vy**2)
# x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))
#
# step = 8
# shaft_width = 0.4
#
# figsize = (6.5, 9)
# fig, axs, cb = create_fig_32(frame_numbers, figsize, fov=fov, coords=(225 - fov[0], 176 - fov[2]), quiver=True)
#
# fname = os.path.join(fig_dir, 'lanes_plot_matplotlib.pdf')
# plt.savefig(fname, dpi=300)
