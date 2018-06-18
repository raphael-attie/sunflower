import glob, os
import csv
import numpy as np
import matplotlib
#matplotlib.use('qt4agg')
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
from scipy.ndimage.morphology import binary_dilation, binary_opening, binary_closing, distance_transform_edt
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.filters import gaussian_filter
import cv2
from sunpy.coordinates.ephemeris import get_sun_L0, get_sun_B0
import astropy.units as u

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


def get_lanes_rgba(lanes_data, color=(0,1,1)):
    # Create an alpha channel from the lanes data.
    lanes_norm = Normalize(0, 0.5 * lanes_data.max(), clip=True)(lanes_data)
    lanes_rgba = np.ones(lanes_norm.shape + (4,))
    lanes_rgba[..., 0] = color[0]
    lanes_rgba[..., 1] = color[1]
    lanes_rgba[..., 2] = color[2]
    lanes_rgba[..., 3] = lanes_norm
    return lanes_rgba


def get_data(frame_number, fov = None, color=(0,1,1)):

    mag = get_avg_data(datafilem, tslices[frame_number])
    cont = fitstools.fitsread(datafile, tslice=int((tslices[frame_number].stop - tslices[frame_number].start)/2)).astype(np.float32)
    vx, vy = get_vel(frame_number)
    lanes = blt.make_lanes(vx, vy, nsteps, maxstep)
    lanes_rgb = get_lanes_rgba(lanes, color=color)

    if fov is not None:
        mag = mag[fov[2]:fov[3], fov[0]:fov[1]]
        vx = vx[fov[2]:fov[3], fov[0]:fov[1]]
        vy = vy[fov[2]:fov[3], fov[0]:fov[1]]
        lanes_rgb = lanes_rgb[fov[2]:fov[3], fov[0]:fov[1]]
        cont = cont[fov[2]:fov[3], fov[0]:fov[1]]

    return mag, lanes, lanes_rgb, vx, vy, cont


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


def detect_moat_radius(lanes_pol, lanes_max, lanes_min, dilation_length):
    #lanes_pol[lanes_pol < min_radius] = 0
    lanes_pol[:, 0:lanes_min] = 0
    mlanes_pol = lanes_pol > lanes_max
    se2 = skimage.morphology.rectangle(dilation_length,2)
    mlanes_polc = binary_dilation(mlanes_pol, se2)
    boundary_r = np.argmax(mlanes_polc, axis=1)

    return boundary_r


def azimuth_analysis(frame_number, dr, case, doprint=False):
    plt.close('all')
    mag, lanes, lanes_colored, vx, vy, cont = get_data(frame_number)
    lanes = blt.make_lanes(vx, vy, 50, maxstep)

    # convert the lambert cylindrical unit to meters and meters/second
    #L0 = get_sun_L0(time=dtimes[frame_number]).to_value(u.deg)
    #lon = 118.0 - L0
    lat = -8.5 - get_sun_B0(time=dtimes[frame_number]).to_value(u.deg)
    dlat = 1.0 / np.cos(lat * np.pi/180.0)

    vx *= ms_unit
    vy *= ms_unit
    vnorm = np.sqrt(vx ** 2 + (vy*dlat) ** 2)

    sigma_factor = 2
    surface, mean, sigma = blt.prep_data2(cont, sigma_factor)
    ymin, xmin = np.unravel_index(np.argmin(cont, axis=None), cont.shape)
    sfov = [0, 200, 0, 200]
    scont = cont[sfov[2]:sfov[3], sfov[0]:sfov[1]]
    sfactor = 5
    threshold = scont.mean() - sfactor * scont.std()
    # Mask based on standard deviation
    spot_mask = cont < threshold
    # Dilation
    dilation_radius = 6
    se = skimage.morphology.disk(dilation_radius)
    spot_mask_dilated = binary_dilation(spot_mask, structure=se)
    spot_dist = distance_transform_edt(spot_mask_dilated)
    #min_radius = round(spot_dist.max() - dilation_radius/2)+1
    min_radius = round(spot_dist.max() - dilation_radius / 2) + 1
    max_radius = min_radius + dr

    ymax2, xmax2 = np.unravel_index(np.argmax(spot_dist, axis=None), spot_dist.shape)
    ycom, xcom = center_of_mass(spot_mask_dilated)


    xm, ym = np.meshgrid(np.arange(cont.shape[1]), np.arange(cont.shape[0]))
    xm2 = xm - xcom
    ym2 = ym - ycom
    r = np.sqrt(xm2 ** 2 + ym2 ** 2)

    rm1 = r < min_radius

    # linear polar transform of velocity magnitude, lanes and magnetograms:
    # https://docs.opencv.org/3.2.0/da/d54/group__imgproc__transform.html#gaa38a6884ac8b6e0b9bed47939b5362f3

    # Velocity
    vn_pol = cv2.linearPolar(vnorm, (xcom, ycom), vnorm.shape[1], cv2.INTER_LANCZOS4 + cv2.WARP_FILL_OUTLIERS)
    Ky = vnorm.shape[0] / 360
    rho = np.arange(vnorm.shape[1])
    phi = np.arange(vnorm.shape[0]) / Ky
    # Lanes
    lanes_pol = cv2.linearPolar(lanes, (xcom, ycom), lanes.shape[1], cv2.INTER_LANCZOS4 + cv2.WARP_FILL_OUTLIERS)
    lanes_pol2 = lanes_pol.copy()
    #lanes_pol[lanes_pol < min_radius] = 0
    lanes_pol2[:, 0:int(min_radius)] = 0
    lanes_max = 23#np.max([25, min_radius+1])
    #TODO: add magnetogram polar transform here. How to define the wedges?
    # magnetogram
    mag_pol = cv2.linearPolar(mag, (xcom, ycom), lanes.shape[1], cv2.INTER_LANCZOS4 + cv2.WARP_FILL_OUTLIERS)


    dilation_length = 7
    boundary_r = detect_moat_radius(lanes_pol2, lanes_max, min_radius, dilation_length)
    med = np.median(boundary_r)
    sigma = boundary_r.std()
    # 2nd pass
    lanes_min = int(med - 2*sigma)
    lanes_pol2 = lanes_pol.copy()
    lanes_pol2[:, 0:lanes_min] = 0
    boundary_r = detect_moat_radius(lanes_pol2, lanes_max, lanes_min, dilation_length)
    med = np.median(boundary_r)
    sigma = boundary_r.std()

    ### sanity check loop
    outliers = np.abs(boundary_r - med) > 4 * sigma
    nout = outliers.sum()
    max_pass = 7
    pass_nb = 0
    while nout > 0 and pass_nb < max_pass:
        dilation_length += 2
        boundary_r = detect_moat_radius(lanes_pol2, lanes_max, min_radius, dilation_length)
        med = np.median(boundary_r)
        sigma = boundary_r.std()
        outliers = np.abs(boundary_r - med) > 4 * sigma
        nout = outliers.sum()
        pass_nb += 1

    if doprint:
        fig, axs = plt.subplots(2, 1, figsize=(4,5.5))
        imlanes = axs[0].imshow(lanes_pol2.T, extent=[phi.min(), phi.max(), rho.min(), rho.max()], cmap='gray_r', origin='lower')
        axs[0].set_ylim([0, 220])
        #axs[0].set_xlabel('Azimuth [degrees]')
        axs[0].set_ylabel('Radial distance [px]', fontsize=fs)

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("top", size="3%", pad=0.3)
        fig.colorbar(imlanes, cax=cax, ax=[axs[0], axs[1]], orientation='horizontal')
        cax.set_title('Distance [px]', fontsize=fs)

        axs[0].tick_params(labelsize=fs)

        axs[1].imshow(binary_dilation(lanes_pol2 > lanes_max, skimage.morphology.rectangle(dilation_length,2)).T, extent=[phi.min(), phi.max(), rho.min(), rho.max()], cmap='gray_r', origin='lower')
        axs[1].set_ylim([0, 220])
        axs[1].set_xlabel('Azimuth [degrees]', fontsize=fs)
        axs[1].set_ylabel('Radial distance [px]', fontsize=fs)
        axs[1].tick_params(labelsize=fs)
        #plt.tight_layout()
        #plt.show()
        fig.subplots_adjust(top=0.99, hspace=-0.15, left=0.15, bottom=0.02)
        plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/lanes_polar_frame_%d_case_%d.pdf'%(frame_number, case), dpi=300)
    # Extract within the boundary radius
    boundary_min = boundary_r - 5
    vn_mean1d = np.array([vn_pol[i, int(min_radius):boundary_min[i]].mean() for i in range(phi.size)])
    # Smooth the boundary
    boundary_r = gaussian_filter(boundary_r.astype(float), 3)
    # Smooth the velocity
    vn_mean1d = gaussian_filter(vn_mean1d, 3)



    #vn_pol2 = [vn_pol[i, int(min_radius):boundary_min[i]] for i in range(phi.size)]
    # Take mean value along rho axis


    if doprint:


        lanes_colored = get_lanes_rgba(lanes, color=(0,0,1))

        figsize = (4, 4)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        imlanes = ax.imshow(lanes, vmin=0, vmax=65, cmap='gray_r', origin='lower')
        ax.set_xlabel('x [px]', fontsize=fs)
        ax.set_ylabel('y [px]', fontsize=fs)
        ax.tick_params(labelsize=fs)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.3)
        fig.colorbar(imlanes, cax=cax, ax=[axs[0], axs[1]], orientation='horizontal')
        cax.set_title('Distance [px]', fontsize=fs)
        cax.tick_params(labelsize=fs)
        fig.tight_layout()
        plt.savefig(
            '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/lanes_fwhm%d_tavg%d_nsteps%d_overview_frame_%d_case_%d.pdf' % (
            fwhm, tavg, nsteps, frame_number, case), dpi=300)

        figsize = (6.5, 6.5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im1 = ax.imshow(cont, cmap='gray', origin='lower')
        im2 = ax.imshow(lanes_colored, origin='lower')
        x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))
        quiv = ax.quiver(x[::step, ::step], y[::step, ::step], vx[::step, ::step], vy[::step, ::step],
                         vnorm[::step, ::step],
                         units='xy', scale=quiver_scale, width=shaft_width, headwidth=headwidth, headlength=headlength,
                         cmap='Oranges')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.10)
        fig.colorbar(quiv, cax=cax)
        cax.set_ylabel('v [m/s]')

        ax.plot(xmax2, ymax2, 'r+')
        ax.plot(xcom, ycom, '.', color=(0,1,0), markerfacecolor='none', ms = 8, lw=2)

        circle1 = plt.Circle((xcom, ycom), radius=220, color='black', ls='--', fill=False)
        ax.add_patch(circle1)
        # ax.add_patch(circle2)
        ax.contour(rm1, 1, colors='red')
        #ax.contour(rm2, 1, colors='green')

        ax.set_xlabel('x [px]', fontsize=fs)
        ax.set_ylabel('y [px]', fontsize=fs)
        ax.tick_params(labelsize=fs)

        text = ax.text(0.02, 0.97, dtimes[frame_number].strftime('%x %X') + ' Frame %d'%frame_number, fontsize=fs,
                       bbox=dict(boxstyle="square", fc='white', alpha=0.8),
                       transform=ax.transAxes)

        fig.tight_layout()
        plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/azimuth_analysis_fwhm%d_tavg%d_nsteps%d_overview_frame_%d_case_%d.pdf'%(fwhm, tavg, nsteps, frame_number, case), dpi=300)

    # Velocity magnitude with Linear-polar transform
    if doprint:
        figsize = (6, 6)
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        im = ax[0].imshow(vn_pol.T, vmin=0, vmax=800, cmap='Oranges', extent=[phi.min(), phi.max(), rho.min(), rho.max()], origin='lower')
        ax[0].plot(phi, boundary_r, 'g--')
        ax[0].axhline(y=min_radius, ls='-', linewidth=1, color='red')
        #ax[0].axhline(y=max_radius, ls='-', linewidth=1, color='green')
        ax[0].set_xlim([0, 360])
        ax[0].set_ylim([0, 110])

        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("top", size="10%", pad=0.1)
        fig.colorbar(im, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.set_xlabel('v [m/s]')
        cax.xaxis.set_label_position('top')

        ax[1].plot(phi, vn_mean1d)
        ax[1].set_xlim([0, 360])
        ax[1].set_ylim([100, 800])
        ax[1].set_ylabel('v [m/s]')

        fig.tight_layout()
        plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/azimuth_analysis_polar_frame_%d_case_%d.pdf'%(frame_number, case))

    return phi, vn_mean1d, boundary_r, vn_pol, min_radius, mag_pol


def create_plot(frame_number, ax, lanes_color=(0,1,1), fov=None, coords=None, continuum=False, quiver=False):

    mag, lanes, lanes_colored, vx, vy, cont = get_data(frame_number, fov, color=lanes_color)
    vx *= ms_unit
    vy *= ms_unit

    if continuum:
        im1 = ax.imshow(cont, cmap='gray', origin='lower')
    else:
        im1 = ax.imshow(mag, vmin=vmin, vmax=vmax, cmap='gray', origin='lower')

    text = ax.text(0.02, 0.95, dtimes[frame_number].strftime('%x %X') + ' (Frame %d)'%frame_number, fontsize=fs,
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
        rectangle = patches.Rectangle(coords[0:2], coords[2], coords[3], fill=False, color='yellow')
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


def create_fig_32(frame_numbers, figsize, overlay_baseline=None, **kwargs):

    fig, axs = plt.subplots(3,2, figsize=figsize)

    im1,_,_ = create_plot(frame_numbers[0], axs[0, 0], lanes_color=(0,1,1), **kwargs)
    _ = create_plot(frame_numbers[1], axs[0, 1], lanes_color=(0,1,1), **kwargs)
    _ = create_plot(frame_numbers[2], axs[1, 0], lanes_color=(0,1,1), **kwargs)
    _ = create_plot(frame_numbers[3], axs[1, 1], lanes_color=(0,1,1), **kwargs)
    _ = create_plot(frame_numbers[4], axs[2, 0], lanes_color=(0,1,1), **kwargs)
    _ = create_plot(frame_numbers[5], axs[2, 1], lanes_color=(0,1,1), **kwargs)




    vx, vy = get_vel(frame_numbers[0])
    lanes = blt.make_lanes(vx, vy, nsteps, maxstep)
    lanes_rgb = get_lanes_rgba(lanes, (0,0,1))
    lanes_rgb = lanes_rgb[fov[2]:fov[3], fov[0]:fov[1]]

    if overlay_baseline is not None:
        # axs[0, 1].imshow(lanes_rgb, origin='lower')
        # axs[1, 0].imshow(lanes_rgb, origin='lower')
        # axs[1, 1].imshow(lanes_rgb, origin='lower')
        # axs[2, 0].imshow(lanes_rgb, origin='lower')
        axs[2, 1].imshow(lanes_rgb, origin='lower')


    axs[0, 0].tick_params(labelleft=True)
    axs[1,0].tick_params(labelleft=True)
    axs[2, 0].tick_params(labelleft=True, labelbottom=True)
    axs[2, 1].tick_params(labelbottom=True)
    # axs[1, 0].set_yticks(np.arange(0, mag.shape[1], 100))
    # axs[2, 0].set_yticks(np.arange(0, mag.shape[1], 100))

    axs[0, 0].set_ylabel('Y [px]', fontsize=fs)
    axs[1, 0].set_ylabel('Y [px]', fontsize=fs)
    axs[2, 0].set_ylabel('Y [px]', fontsize=fs)
    axs[2, 0].set_xlabel('X [px]', fontsize=fs)
    axs[2, 1].set_xlabel('X [px]', fontsize=fs)


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

### Azimuth analysis

# Continuum
# Quiver plot parameters
step = 8
quiver_scale = 45
shaft_width = 1.2
headwidth = 3
headlength = 5

doprint=False

L0_start = get_sun_L0(time='2017-09-01T01:00:00').to_value(u.deg)
L0 = get_sun_L0(time='2017-09-03T12:30:00').to_value(u.deg)

# Get the preceding quiet states


dr = 20
phi_vn_r0 = [azimuth_analysis(i, dr, 0, doprint=doprint) for i in [0,1,2]]

frame_numbers = [3, 12, 14, 25, 39, 46]

phi_vn_r1 = [azimuth_analysis(f, dr, 1, doprint=doprint) for f in frame_numbers]

frame_numbers = [44, 46, 50, 52, 56, 60]
phi_vn_r2 = [azimuth_analysis(f, dr, 2, doprint=doprint) for f in frame_numbers]

phi = phi_vn_r1[0][0]
rho = np.arange(512)

phi_mask = np.logical_and(phi >=220, phi<260)

figsize = (6, 6)
frame_numbers = [3, 12, 14, 25, 39, 46]
max_r_loc = int(np.argmax(phi_vn_r1[1][2]))
phi_max = phi[max_r_loc]


med_r = np.median(phi_vn_r1[0][2])
mean_r = phi_vn_r1[0][2].mean()
sigma_r = phi_vn_r1[0][2].std()


# max position when indexing from the end
max_r_loc_end =phi_vn_r1[1][2].size - max_r_loc -1
# Get the left and right azimuthal limits
r_mask1 = np.abs(phi_vn_r1[1][2][::-1][max_r_loc_end:] - mean_r) >  sigma_r
r_off_left = int(round(max_r_loc - np.where(~r_mask1)[0][0]))
r_mask2 = np.abs(phi_vn_r1[1][2][max_r_loc:] - mean_r) >  sigma_r
r_off_right = int(round(max_r_loc + np.where(~r_mask2)[0][0]))

phi_mask = np.ones(phi.size, dtype=np.bool)
phi_mask[r_off_left:r_off_right] = False

v_filtered = phi_vn_r1[0][1][phi_mask]
med_v = np.median(v_filtered)
mean_v = v_filtered.mean()
sigma_v = v_filtered.std()


fig, ax = plt.subplots(2, 1, figsize=figsize)
ax[0].plot(phi, phi_vn_r1[0][2], 'k--', lw=2, label='frame %d'%frame_numbers[0])
ax[0].plot(phi, phi_vn_r1[1][2], 'g-', lw=2, label='frame %d'%frame_numbers[1])
ax[0].plot(phi, phi_vn_r1[2][2], 'r-.', lw=2, label='frame %d'%frame_numbers[2])
# ax[0].plot(phi_vn_r1[3][0], phi_vn_r1[3][2], color='blue', lw=2, ls=':', label='frame %d'%frame_numbers[3])
ax[0].plot(phi_vn_r1[4][0], phi_vn_r1[4][2], 'b:', lw=2, label='frame %d'%frame_numbers[4])
#ax[0].plot(phi_vn_r1[5][0], phi_vn_r1[5][2], color='grey', ls='--', lw=1, label='frame %d'%frame_numbers[5])
ax[0].axvline(x=phi[r_off_left], ls='-', linewidth=1, color='black')
ax[0].axvline(x=phi[r_off_right], ls='-', linewidth=1, color='black')
text = ax[0].text(phi[r_off_left]+2, 105, 'Case 1 & 2', fontsize=fs)

text = ax[0].text(0.86, 0.1, r'$\overline{r}$ = %d px'%mean_r, fontsize=fs, transform=ax[0].transAxes)
text = ax[0].text(0.86, 0.04, r'$\sigma_r$ = %d px'%sigma_r, fontsize=fs, transform=ax[0].transAxes)
ax[0].axhline(y=mean_r, ls='-', linewidth=2, color='gray')
ax[0].axhline(y=mean_r + 2*sigma_r, ls='--', linewidth=1, color='gray')
ax[0].axhline(y=mean_r - 2*sigma_r, ls='--', linewidth=1, color='gray')
text = ax[0].text(5, mean_r + 2, r'$\overline{r}$', fontsize=fs)
text = ax[0].text(5, mean_r + 2*sigma_r + 2, r'$\overline{r} + 2 \sigma_r$', fontsize=fs)
text = ax[0].text(5, mean_r - 2*sigma_r - 6, r'$\overline{r} - 2 \sigma_r$', fontsize=fs)

ax[0].set_xlim([0, 360])
ax[0].set_ylim([20, 110])
ax[0].set_xlabel('Azimuth [degrees]')
ax[0].set_ylabel('r [px]')
ax[0].legend(loc='upper right')
ax[0].tick_params(labelsize=fs)

ax[1].plot(phi, phi_vn_r1[0][1], 'k--', lw=2)
ax[1].plot(phi, phi_vn_r1[1][1], 'g-', lw=2)
ax[1].plot(phi, phi_vn_r1[2][1], 'r-.', lw=2)
ax[1].plot(phi, phi_vn_r1[4][1], 'b:', lw=2)
ax[1].set_xlim([0, 360])
ax[1].set_ylim([100, 700])
ax[1].set_ylabel('v [m/s]')
ax[1].set_xlabel('Azimuth [degrees]')
ax[1].axvline(x=phi[r_off_left], ls='-', linewidth=1, color='black')
ax[1].axvline(x=phi[r_off_right], ls='-', linewidth=1, color='black')

text = ax[1].text(0.84, 0.1, r'$\overline{v}$ = %d m/s'%mean_v, fontsize=fs, transform=ax[1].transAxes)
text = ax[1].text(0.84, 0.04, r'$\sigma_v$ = %d m/s'%sigma_v, fontsize=fs, transform=ax[1].transAxes)
ax[1].axhline(y=mean_v, ls='-', linewidth=2, color='gray', label=r'$\overline{v}$')
ax[1].axhline(y=mean_v + 2*sigma_v, ls='--', linewidth=1, color='gray')
ax[1].axhline(y=mean_v - 2*sigma_v, ls='--', linewidth=1, color='gray')
text = ax[1].text(5, mean_v + 5, r'$\overline{v}$', fontsize=fs)
text = ax[1].text(5, mean_v + 2*sigma_v + 10, r'$\overline{v} + 2 \sigma_v$', fontsize=fs)
text = ax[1].text(5, mean_v - 2*sigma_v - 40, r'$\overline{v} - 2 \sigma_v$', fontsize=fs)

plt.tight_layout()
plt.show()

plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/moat_radius_velocity_1.pdf', dpi=300)


frame_numbers = [44, 46, 50, 52, 56, 60]

med_r2 = np.median(phi_vn_r2[0][2])
mean_r2 = phi_vn_r2[0][2].mean()
sigma_r2 = phi_vn_r2[0][2].std()

med_v2 = np.median(phi_vn_r2[0][1])
mean_v2 = phi_vn_r2[0][1].mean()
sigma_v2 = phi_vn_r2[0][1].std()

# case 3
vel3 = phi_vn_r2[2][2].copy()
# max position
max_r_loc = int(np.argmax(vel3))

# max position when indexing from the end
max_r_loc_end =vel3.size - max_r_loc -1
# Get the left and right azimuthal limits
r_mask1 = np.abs(vel3[::-1][max_r_loc_end:] - mean_r2) >  sigma_r2
r_off_left1 = int(round(max_r_loc - np.where(~r_mask1)[0][0]))
r_mask2 = np.abs(vel3[max_r_loc:] - mean_r2) >  sigma_r2
r_off_right1 = int(round(max_r_loc + np.where(~r_mask2)[0][0]))

phi_mask = np.ones(phi.size, dtype=np.bool)
phi_mask[r_off_left1:r_off_right1] = False

# case 4 - need to get rid of points over case 3
vel3[~phi_mask] = 0
# max position
max_r_loc2 = int(np.argmax(vel3))
# max position when indexing from the end
max_r_loc_end =vel3.size - max_r_loc2 -1
# Get the left and right azimuthal limits
r_mask1 = np.abs(vel3[::-1][max_r_loc_end:] - mean_r2) >  sigma_r2
r_off_left2 = int(round(max_r_loc2 - np.where(~r_mask1)[0][0]))
r_mask2 = np.abs(vel3[max_r_loc2:] - mean_r2) >  sigma_r2
r_off_right2 = int(round(max_r_loc2 + np.where(~r_mask2)[0][0]))
# Update the phi mask
phi_mask[r_off_left2:r_off_right2] = False


v_filtered = phi_vn_r2[0][1][phi_mask]
med_v2 = np.median(v_filtered)
mean_v2 = v_filtered.mean()
sigma_v2 = v_filtered.std()



fig, ax = plt.subplots(2, 1, figsize=figsize)
ax[0].plot(phi, phi_vn_r2[0][2], 'k--', label='frame %d'%frame_numbers[0])
ax[0].plot(phi, phi_vn_r2[1][2], 'g-', label='frame %d'%frame_numbers[1])
ax[0].plot(phi, phi_vn_r2[2][2], color='red', ls='-.', label='frame %d'%frame_numbers[2])
# ax[1].plot(phi_vn_r2[3][0], phi_vn_r2[3][2], color='blue', ls=':', label='frame %d'%frame_numbers[3])
# ax[1].plot(phi_vn_r2[4][0], phi_vn_r2[4][2], color='orange', ls='-', lw=1, label='frame %d'%frame_numbers[4])
ax[0].plot(phi_vn_r2[5][0], phi_vn_r2[5][2], color='grey', ls='--', lw=2, label='frame %d'%frame_numbers[5])

ax[0].axvline(x=phi[r_off_left1], ls='-', linewidth=1, color='black')
ax[0].axvline(x=phi[r_off_right1], ls='-', linewidth=1, color='black')
ax[0].axvline(x=phi[r_off_left2], ls='-', linewidth=1, color='black')
ax[0].axvline(x=phi[r_off_right2], ls='-', linewidth=1, color='black')

text = ax[0].text(0.86, 0.1, r'$\overline{r}$ = %d px'%mean_r2, fontsize=fs, transform=ax[0].transAxes)
text = ax[0].text(0.86, 0.04, r'$\sigma_r$ = %d px'%sigma_r2, fontsize=fs, transform=ax[0].transAxes)
ax[0].axhline(y=mean_r2, ls='-', linewidth=2, color='gray')
ax[0].axhline(y=mean_r2 + 2*sigma_r2, ls='--', linewidth=1, color='gray')
ax[0].axhline(y=mean_r2 - 2*sigma_r2, ls='--', linewidth=1, color='gray')
text = ax[0].text(1, mean_r2 + 1, r'$\overline{r}$', fontsize=fs)
text = ax[0].text(5, mean_r2 + 2*sigma_r2 + 7, r'$\overline{r} + 2 \sigma_r$', fontsize=fs)
text = ax[0].text(5, mean_r2 - 2*sigma_r2 - 6, r'$\overline{r} - 2 \sigma_r$', fontsize=fs)

text = ax[0].text(phi[r_off_left2]+2, 90, 'Case 3', fontsize=fs, rotation = 'vertical')
text = ax[0].text(phi[r_off_left1]+2, 90, 'Case 4', fontsize=fs, rotation = 'vertical')


ax[0].set_xlim([0, 360])
ax[0].set_ylim([20, 110])
ax[0].set_xlabel('Azimuth [degrees]')
ax[0].set_ylabel('Moat radius [px]')
ax[0].legend(loc='upper right')
ax[0].tick_params(labelsize=fs)

ax[1].plot(phi, phi_vn_r2[0][1], 'k--', lw=2)
ax[1].plot(phi, phi_vn_r2[1][1], 'g-', lw=2)
ax[1].plot(phi, phi_vn_r2[2][1], color='red', lw=2, ls='-.')
ax[1].plot(phi, phi_vn_r2[3][1], color='grey', ls='--', lw=2)
ax[1].set_xlim([0, 360])
ax[1].set_ylim([100, 700])
ax[1].set_ylabel('v [m/s]')
ax[1].set_xlabel('Azimuth [degrees]')
ax[1].axvline(x=phi[r_off_left1], ls='-', linewidth=1, color='black')
ax[1].axvline(x=phi[r_off_right1], ls='-', linewidth=1, color='black')
ax[1].axvline(x=phi[r_off_left2], ls='-', linewidth=1, color='black')
ax[1].axvline(x=phi[r_off_right2], ls='-', linewidth=1, color='black')

text = ax[1].text(0.84, 0.1, r'$\overline{v}$ = %d m/s'%mean_v2, fontsize=fs, transform=ax[1].transAxes)
text = ax[1].text(0.84, 0.04, r'$\sigma_v$ = %d m/s'%sigma_v2, fontsize=fs, transform=ax[1].transAxes)
ax[1].axhline(y=mean_v2, ls='-', linewidth=2, color='gray', label=r'$\overline{v}$')
ax[1].axhline(y=mean_v2 + 2*sigma_v2, ls='--', linewidth=1, color='gray')
ax[1].axhline(y=mean_v2 - 2*sigma_v2, ls='--', linewidth=1, color='gray')
text = ax[1].text(5, mean_v2 + 30, r'$\overline{v}$', fontsize=fs)
text = ax[1].text(5, mean_v2 + 2*sigma_v2 + 30, r'$\overline{v} + 2 \sigma_v$', fontsize=fs)
text = ax[1].text(5, mean_v2 - 2*sigma_v2 - 60, r'$\overline{v} - 2 \sigma_v$', fontsize=fs)

plt.tight_layout()
plt.show()
plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/moat_radius_velocity_2.pdf', dpi=300)

# Relative evolution
ref = phi_vn_r1[0][2]
vref = phi_vn_r1[0][1]

mean_dr = phi_vn_r1[0][2].mean()
sigma_dr = phi_vn_r1[0][2].std()

figsize = (6, 6)
frame_numbers = [3, 12, 14, 25, 39, 46]
fig, ax = plt.subplots(2, 1, figsize=figsize)
ax[0].axhline(y=0, ls='--', linewidth=1, color='black')
ax[0].plot(phi, (phi_vn_r1[1][2] - ref)/ref, 'g-', lw=2, label='frame %d'%frame_numbers[1])
ax[0].plot(phi, (phi_vn_r1[2][2] - ref)/ref, color='red', lw=2, ls='-.', label='frame %d'%frame_numbers[2])
ax[0].plot(phi, (phi_vn_r1[4][2] - ref)/ref, 'b:', lw=2, label='frame %d'%frame_numbers[4])
ax[0].axvline(x=phi[r_off_left], ls='-', linewidth=1, color='black')
ax[0].axvline(x=phi[r_off_right], ls='-', linewidth=1, color='black')
text = ax[0].text(phi[r_off_left], 0.45, 'Case 1 & 2', fontsize=fs)

ax[0].set_xlim([0, 360])
ax[0].set_ylim([-0.51, 0.51])
ax[0].set_xlabel('Azimuth [degrees]')
ax[0].set_ylabel(r'$\Delta$r / r')
ax[0].legend()
ax[0].tick_params(labelsize=fs)
#ax[0].set_yticks(np.arange(-0.5, 0.6, 0.1))

ax[1].axhline(y=0, ls='--', linewidth=1, color='black')
ax[1].plot(phi, (phi_vn_r1[1][1] - vref)/vref, 'g-', lw=2)
ax[1].plot(phi, (phi_vn_r1[2][1] - vref)/vref, 'r-.', lw=2)
ax[1].plot(phi, (phi_vn_r1[4][1] - vref)/vref, 'b:', lw=2)
ax[1].set_xlim([0, 360])
ax[1].set_ylim([-1.05, 1.05])
ax[1].set_ylabel(r'$\Delta$v / v')
ax[1].set_xlabel('Azimuth [degrees]')
ax[1].axvline(x=phi[r_off_left], ls='-', linewidth=1, color='black')
ax[1].axvline(x=phi[r_off_right], ls='-', linewidth=1, color='black')
ax[1].set_yticks(np.arange(-1, 1.05, 0.25))

plt.tight_layout()
plt.show()
plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/moat_radius_velocity_relative_1.pdf', dpi=300)

#TODO: bin the data over wedges. Integrate magnetogram over wedges
# see https://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
bins = np.arange(0,361,24)

drrs = [(phi_vn_r1[i][2] - ref)/ref for i in range(len(phi_vn_r1))]
bin_width = 34
conv_drrs = [np.convolve(drr, np.ones((bin_width,))/bin_width, mode='valid') for drr in drrs]
bdrr = [conv_drr[::bin_width] for conv_drr in conv_drrs]
bphi = phi[::bin_width]
#sigma_boundary = phi_vn_r1[0][2].std()
wedge_rmin = int(med_r)
wedge_rmax = int(1.5*med_r)
mag_pols = [phi_vn_r1[i][5] for i in range(len(phi_vn_r1))]
# Integrated magnetograms over wedges
wedged_mag_pol1 = [mag_pol[:, wedge_rmin:wedge_rmax].sum(axis=1) for mag_pol in mag_pols]
wedged_mag_pol2 = [np.convolve(wmag_pol, np.ones((bin_width,))/bin_width, mode='valid') for wmag_pol in wedged_mag_pol1]
bmag_pol2 = [wmag_pol2[::bin_width] for wmag_pol2 in wedged_mag_pol2]
#TODO: bin the data over all frames, not just the subset of frame_numbers = [3, 12, 14, 25, 39, 46]


frame_numbers = [44, 46, 50, 52, 56, 60]
ref = phi_vn_r2[0][2]
vref = phi_vn_r2[0][1]

fig, ax = plt.subplots(2, 1, figsize=figsize)
ax[0].axhline(y=0, ls='--', linewidth=1, color='black')
ax[0].plot(phi_vn_r2[1][0], (phi_vn_r2[1][2] - ref)/ref, 'g-', label='frame %d'%frame_numbers[1])
ax[0].plot(phi_vn_r2[2][0], (phi_vn_r2[2][2] - ref)/ref, color='red', ls='-.', label='frame %d'%frame_numbers[2])
# ax[0].plot(phi_vn_r2[3][0], phi_vn_r2[3][2], color='blue', ls=':', label='frame %d'%frame_numbers[3])
# ax[0].plot(phi_vn_r2[4][0], phi_vn_r2[4][2], color='orange', ls='-', lw=1, label='frame %d'%frame_numbers[4])
ax[0].plot(phi_vn_r2[5][0], (phi_vn_r2[5][2] - ref)/ref, color='grey', ls='--', lw=2, label='frame %d'%frame_numbers[5])

ax[0].axvline(x=phi[r_off_left1], ls='-', linewidth=1, color='black')
ax[0].axvline(x=phi[r_off_right1], ls='-', linewidth=1, color='black')
ax[0].axvline(x=phi[r_off_left2], ls='-', linewidth=1, color='black')
ax[0].axvline(x=phi[r_off_right2], ls='-', linewidth=1, color='black')
text = ax[0].text(phi[r_off_left2]+1, 0.45, 'Case 3', fontsize=fs, rotation = 'vertical')
text = ax[0].text(phi[r_off_left1]+1, 0.45, 'Case 4', fontsize=fs, rotation = 'vertical')


ax[0].set_xlim([0, 360])
ax[0].set_ylim([-0.51, 0.51])
ax[0].set_xlabel('Azimuth [degrees]')
ax[0].set_ylabel(r'$\Delta$r / r')
ax[0].legend(loc='upper right')
ax[0].tick_params(labelsize=fs)

ax[1].axhline(y=0, ls='--', linewidth=1, color='black')
ax[1].plot(phi, (phi_vn_r2[1][1] - vref)/vref, 'g-', lw=2)
ax[1].plot(phi, (phi_vn_r2[2][1] - vref)/vref, color='red', lw=2, ls='-.')
ax[1].plot(phi, (phi_vn_r2[5][1] - vref)/vref, color='grey', ls='--', lw=2)
ax[1].set_xlim([0, 360])
ax[1].set_ylim([-1.05, 1.05])
ax[1].set_ylabel(r'$\Delta$v / v')
ax[1].set_xlabel('Azimuth [degrees]')

ax[1].axvline(x=phi[r_off_left1], ls='-', linewidth=1, color='black')
ax[1].axvline(x=phi[r_off_right1], ls='-', linewidth=1, color='black')
ax[1].axvline(x=phi[r_off_left2], ls='-', linewidth=1, color='black')
ax[1].axvline(x=phi[r_off_right2], ls='-', linewidth=1, color='black')

ax[1].set_yticks(np.arange(-1, 1.05, 0.25))
plt.tight_layout()
plt.show()
plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/moat_radius_velocity_relative_2.pdf', dpi=300)


# TODO: consider doing polar plots:
# https://matplotlib.org/users/annotations.html
# Velocity / azimuth
frame_numbers = [3, 12, 14, 25, 39, 46]
figsize = (5, 7)
fig, ax = plt.subplots(4, 1, figsize=figsize)
for i in range(3):
    im = ax[i].imshow(phi_vn_r1[i][3].T, vmin=0, vmax=800, cmap='Oranges', extent=[phi.min(), phi.max(), rho.min(), rho.max()], origin='lower')
    ax[i].plot(phi, phi_vn_r1[i][2], 'g--')
    ax[i].axhline(y=phi_vn_r1[i][4], ls='-', linewidth=1, color='red')
    ax[i].set_xlim([0, 360])
    ax[i].set_ylim([0, 110])
    ax[i].set_yticks(np.arange(0, 101, 25))
    ax[i].set_ylabel('Moat radius [px]')
    ax[i].tick_params(labelsize=fs)

ax[3].plot(phi, phi_vn_r1[0][1], 'k--', lw=2, label='frame %d'%frame_numbers[0])
ax[3].plot(phi, phi_vn_r1[1][1], 'g-', lw=2, label='frame %d'%frame_numbers[1])
ax[3].plot(phi, phi_vn_r1[2][1], color='red', lw=2, ls='-.', label='frame %d'%frame_numbers[2])
ax[3].set_xlim([0, 360])
ax[3].set_ylim([0, 700])
ax[3].set_ylabel('v [m/s]')
ax[3].set_xlabel('Azimuth [degrees]')
ax[3].legend(loc='lower left', fontsize=fs, framealpha=0.5)

fig.subplots_adjust(left=0.15, bottom=0.08, top=0.9, right=0.88, hspace = 0.2, wspace=0.055)
cbar_ax = fig.add_axes([0.15, 1.35, 0.7, 0.07])
ip = InsetPosition(ax[0], [0, 1.35, 1, 0.07])
cbar_ax.set_axes_locator(ip)
cb = fig.colorbar(im, cax=cbar_ax, ax=[ax[0], ax[1], ax[2]], orientation='horizontal')
cb.ax.tick_params(labelsize=fs)
cbar_ax.set_title('v [m/s]', fontsize=fs)

plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/azimuth_velocity_1.pdf', dpi=300)


frame_numbers = [44, 46, 50, 52, 56, 60]
fig, ax = plt.subplots(4, 1, figsize=figsize)
for i in range(3):
    im = ax[i].imshow(phi_vn_r2[i][3].T, vmin=0, vmax=800, cmap='Oranges', extent=[phi.min(), phi.max(), rho.min(), rho.max()], origin='lower')
    ax[i].plot(phi, phi_vn_r2[i][2], 'g--')
    ax[i].axhline(y=phi_vn_r2[i][4], ls='-', linewidth=1, color='red')
    ax[i].set_xlim([0, 360])
    ax[i].set_ylim([0, 110])
    ax[i].set_yticks(np.arange(0, 101, 25))
    ax[i].set_ylabel('Moat radius [px]')
    ax[i].tick_params(labelsize=fs)

ax[3].plot(phi, phi_vn_r2[0][1], 'k--', lw=2, label='frame %d'%frame_numbers[0])
ax[3].plot(phi, phi_vn_r2[1][1], 'g-', lw=2, label='frame %d'%frame_numbers[1])
ax[3].plot(phi, phi_vn_r2[2][1], color='red', lw=2, ls='-.', label='frame %d'%frame_numbers[2])
ax[3].set_xlim([0, 360])
ax[3].set_ylim([0, 700])
ax[3].set_ylabel('v [m/s]')
ax[3].set_xlabel('Azimuth [degrees]')
ax[3].legend(loc='lower left', fontsize=fs, framealpha=0.5)

fig.subplots_adjust(left=0.15, bottom=0.08, top=0.9, right=0.88, hspace = 0.2, wspace=0.055)
cbar_ax = fig.add_axes([0.15, 1.35, 0.7, 0.07])
ip = InsetPosition(ax[0], [0, 1.35, 1, 0.07])
cbar_ax.set_axes_locator(ip)
cb = fig.colorbar(im, cax=cbar_ax, ax=[ax[0], ax[1], ax[2]], orientation='horizontal')
cb.ax.tick_params(labelsize=fs)
cbar_ax.set_title('v [m/s]', fontsize=fs)

plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/azimuth_velocity_2.pdf', dpi=300)


# Magnetogram
step = 8
figsize = (6.5, 6.5)

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

plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/mag_lanes_fwhm%d_tavg%d_nsteps%d_maxstep%d_OVERVIEW_frame_%d.pdf'%(fwhm, tavg, nsteps, maxstep, frame_numbers[0]), dpi=300)



# Circle coordinates
# Rectangle coordinates
case = 1
frame_numbers = [3, 12, 14, 25, 39, 46]

#fov = [50, 350, 50, 350]
fov = [100, 400, 100, 400]
# Coordinates of the rectangle: (left, bottom, width, height)
coords = (220-50 - fov[0], 165-50-fov[2], 125, 125)


vmin = -200
vmax = 200

figsize = (6.5, 9)

fig, axs, cb = create_fig_32(frame_numbers, figsize, fov=fov, coords=coords)
marker_color = 'red'
marker_coords = (125, 82)

axs[0,0].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[0,1].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[1,0].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[1,1].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[2,0].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[2,1].plot(*marker_coords, marker='+', color=marker_color, ms=8)

plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/lanes_fwhm%d_tavg%d_nsteps%d_maxstep%d_plot32_case_%d.pdf'%(fwhm, tavg, nsteps, maxstep, case), dpi=300)



## 2nd case of emergence
case = 2
frame_numbers = [44, 46, 50, 52, 56, 60]
coords = (200-50 - fov[0], 285-50-fov[2], 125, 125)

vmin = -200
vmax = 200

figsize = (6.5, 9)

fig, axs, cb = create_fig_32(frame_numbers, figsize, overlay_baseline=True, fov=fov, coords=coords)
marker_coords = (110, 150)
axs[0,0].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[0,1].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[1,0].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[1,1].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[2,0].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[2,1].plot(*marker_coords, marker='+', color=marker_color, ms=8)

marker_coords = (130, 185)
axs[0,0].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[0,1].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[1,0].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[1,1].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[2,0].plot(*marker_coords, marker='+', color=marker_color, ms=8)
axs[2,1].plot(*marker_coords, marker='+', color=marker_color, ms=8)



plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/paper/lanes_fwhm%d_tavg%d_nsteps%d_maxstep%d_plot32_case_%d.pdf'%(fwhm, tavg, nsteps, maxstep, case), dpi=300)


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
