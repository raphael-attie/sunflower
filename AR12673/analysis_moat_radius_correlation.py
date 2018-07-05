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
from scipy.signal import correlate
import cv2
from sunpy.coordinates.ephemeris import get_sun_L0, get_sun_B0
import astropy.units as u
from figures import lineplots
import matplotlib.lines as mlines
from skimage.measure import block_reduce
from importlib import reload

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
    min_radius = round(spot_dist.max() - dilation_radius / 2) + 1
    ycom, xcom = center_of_mass(spot_mask_dilated)

    xm, ym = np.meshgrid(np.arange(cont.shape[1]), np.arange(cont.shape[0]))
    xm2 = xm - xcom
    ym2 = ym - ycom
    r = np.sqrt(xm2 ** 2 + ym2 ** 2)

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

    # Extract within the boundary radius
    boundary_min = boundary_r - 5
    vn_mean1d = np.array([vn_pol[i, int(min_radius):boundary_min[i]].mean() for i in range(phi.size)])
    # Smooth the boundary
    boundary_r = gaussian_filter(boundary_r.astype(float), 3)
    # Smooth the velocity
    vn_mean1d = gaussian_filter(vn_mean1d, 3)

    return phi, vn_mean1d, boundary_r, vn_pol, min_radius, mag_pol, mag, (xcom, ycom), lanes



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

### Azimuth analysis
doprint=False

L0_start = get_sun_L0(time='2017-09-01T01:00:00').to_value(u.deg)
L0 = get_sun_L0(time='2017-09-03T12:30:00').to_value(u.deg)


dr = 20
frame_numbers = np.arange(61)
phi_vn_r_all = [azimuth_analysis(f, dr, 1, doprint=doprint) for f in frame_numbers]
min_radii = np.array([samples[4] for samples in phi_vn_r_all])

# Case 1 & 2
case_frames1 = [3, 12, 14, 25, 39, 46]
phi = phi_vn_r_all[0][0]
rho = np.arange(512)
phi_mask = np.logical_and(phi >=220, phi<260)

# Get the maximum radius in case 1 (at frame 3)
max_r_loc = int(np.argmax(phi_vn_r_all[case_frames1[1]][2]))
phi_max = phi[max_r_loc]

med_r = np.median(phi_vn_r_all[case_frames1[0]][2])
mean_r = phi_vn_r_all[case_frames1[0]][2].mean()
sigma_r = phi_vn_r_all[case_frames1[0]][2].std()

# max position when indexing from the end
max_r_loc_end =phi_vn_r_all[case_frames1[1]][2].size - max_r_loc -1
# Get the left and right azimuthal limits
r_mask1 = np.abs(phi_vn_r_all[case_frames1[1]][2][::-1][max_r_loc_end:] - mean_r) >  sigma_r
r_off_left = int(round(max_r_loc - np.where(~r_mask1)[0][0]))
r_mask2 = np.abs(phi_vn_r_all[case_frames1[1]][2][max_r_loc:] - mean_r) >  sigma_r
r_off_right = int(round(max_r_loc + np.where(~r_mask2)[0][0]))

phi_mask = np.ones(phi.size, dtype=np.bool)
phi_mask[r_off_left:r_off_right] = False

v_filtered = phi_vn_r_all[case_frames1[0]][1][phi_mask]
med_v = np.median(v_filtered)
mean_v = v_filtered.mean()
sigma_v = v_filtered.std()

# Case 3 & 4
case_frames2 = [44, 46, 50, 52, 56, 60]
med_r2 = np.median(phi_vn_r_all[case_frames2[0]][2])
mean_r2 = phi_vn_r_all[case_frames2[0]][2].mean()
sigma_r2 = phi_vn_r_all[case_frames2[0]][2].std()

med_v2 = np.median(phi_vn_r_all[case_frames2[0]][1])
mean_v2 = phi_vn_r_all[case_frames2[0]][1].mean()
sigma_v2 = phi_vn_r_all[case_frames2[0]][1].std()

# case 3
vel3 = phi_vn_r_all[case_frames2[2]][2].copy()
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

# case 3 and 4 - need to get rid of points over case 3
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

v_filtered = phi_vn_r_all[case_frames2[0]][1][phi_mask]
med_v2 = np.median(v_filtered)
mean_v2 = v_filtered.mean()
sigma_v2 = v_filtered.std()


# Relative evolution Case 1 & 2
ref = phi_vn_r_all[case_frames1[0]][2]
vref = phi_vn_r_all[case_frames1[0]][1]

mean_dr = phi_vn_r_all[case_frames1[0]][2].mean()
sigma_dr = phi_vn_r_all[case_frames1[0]][2].std()

#TODO: bin the data over wedges. Integrate magnetogram over wedges
#TODO: BUG: the reference must be adaptated. It wasn't the same from case 1 & 2 to 3 & 4, use median?

# see https://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy

# Define angular bin size
bin_width = 32#int(np.round((phi_bins[2] - phi_bins[1]) / (phi[2] - phi[1])))
phi_bins = np.linspace(0,336, int(512/bin_width))

# Bin the relative change of the moat radius
ref =  phi_vn_r_all[0][2] # med_r
rsamples = np.array([samples[2] for samples in phi_vn_r_all])
rsamples_binned = block_reduce(rsamples, block_size=(1, bin_width), func = np.max)
rsamples_binned2 = block_reduce(rsamples, block_size=(1, bin_width), func = np.mean)
# diff_r = (rsamples[2:] - rsamples[0:-2])/rsamples[0]

drrs = [(samples[2] - ref)/ref for samples in phi_vn_r_all]

conv_drrs = [np.convolve(drr, np.ones((bin_width,))/bin_width, mode='same') for drr in drrs]
#bdrrs = [conv_drr[::bin_width] for conv_drr in conv_drrs]
#bdrrs = (rsamples_binned - rsamples_binned[0, :])/ rsamples_binned[0, :]
#bdrrs = (rsamples_binned - np.median(rsamples_binned, axis=0))/ np.median(rsamples_binned, axis=0)
#bdrrs = (bdrrs - bdrrs.mean(axis=0))/bdrrs.std(axis=0)
r_range = rsamples_binned.max(axis=0) - rsamples_binned.min(axis=0)

gaps = (5, 6, 7, 29, 30, 31, 53, 54, 55)

rmask = rsamples_binned - rsamples_binned.mean(axis=0) > rsamples_binned.std(axis=0)
rmask[gaps, :] = True

rmasked_array = np.ma.array(rsamples_binned, mask=rmask)
rmasked_array2 = np.ma.array(rsamples_binned2, mask=rmask)
mr_range = rmasked_array.max(axis=0) - rmasked_array.min(axis=0)
mr_range2 = rmasked_array2.max(axis=0) - rmasked_array2.min(axis=0)


#bdrrs = (rsamples_binned - rsamples_binned[0, :])/rmasked_array.std(axis=0)
#bdrrs = (rsamples_binned - rsamples_binned[3, :])/mr_range
#bdrrs = (rsamples_binned - rsamples_binned[3, :])/rsamples_binned[3, :]
#bdrrs = (rsamples_binned - rsamples_binned[3, :])/rsamples_binned[3, :]
#bdrrs = (rsamples_binned - rmasked_array.mean(axis=0))/rmasked_array.mean(axis=0)
#bdrrs = (rsamples_binned - rmasked_array.mean(axis=0))/mr_range
#bdrrs = (rsamples_binned - rmasked_array2.mean(axis=0))/rmasked_array2.mean(axis=0)
#bdrrs = (rsamples_binned2 - rsamples_binned2[3, :])/rsamples_binned2[3, :]
#bdrrs = (rsamples_binned2 - rsamples_binned2[3, :])/rmasked_array2.mean(axis=0)
#bdrrs = (rsamples_binned2 - rsamples_binned2[3, :])/rmasked_array2.mean(axis=1)[:, np.newaxis]
#bdrrs = (rsamples_binned - rsamples_binned[3, :])/rmasked_array2.mean(axis=0)
#bdrrs = (rsamples_binned - rmasked_array2.mean(axis=0))/rmasked_array2.mean(axis=0)
#bdrrs = (rsamples_binned - rmasked_array2.mean(axis=0)[np.newaxis, :])/rmasked_array2.mean(axis=0)[np.newaxis, :]
#bdrrs = (rsamples_binned - rmasked_array2[3,:].mean())/rmasked_array2[3,:].mean()
#bdrrs = (rsamples_binned - rsamples_binned[3, :])/rsamples_binned[3, :]
bdrrs = (rsamples_binned - rsamples_binned[3, :])/rsamples_binned2[3, :]


bphi = phi[::bin_width]
#sigma_boundary = phi_vn_r_all[0][2].std()
wedge_rmin = int(med_r)
wedge_rmax = int(2*med_r)
mag_pols = [phi_vn_r[5] for phi_vn_r in phi_vn_r_all]
mags = [phi_vn_r[6] for phi_vn_r in phi_vn_r_all]
spot_centers = [phi_vn_r[7] for phi_vn_r in phi_vn_r_all]
lanes = [phi_vn_r[8] for phi_vn_r in phi_vn_r_all]

# Integrated magnetograms over wedges
sel = np.ones((bin_width,)) # structuring element (rectangular window)
wedged_mag_pol1 = [np.abs(mag_pol[:, wedge_rmin:wedge_rmax]).sum(axis=1)/(wedge_rmax - wedge_rmin+1) for mag_pol in mag_pols]
wedged_mag_pol2 = [np.convolve(wmag_pol, sel/sel.sum(), mode='same') for wmag_pol in wedged_mag_pol1]
bmag_pol2 = np.array([wmag_pol2[::bin_width] for wmag_pol2 in wedged_mag_pol2])
bdrrs2 = np.array(bdrrs)


for i in range(60):

    frame_nb = i

    lanes_colored = get_lanes_rgba(lanes[frame_nb], color=(0,1,1))
    magmin = -200
    magmax = 200

    xyls = [lineplots.line_polar(spot_centers[frame_nb], angle, wedge_rmin, wedge_rmax) for angle in phi_bins]
    lcolors = 'yellow'

    plt.figure(0, figsize=(18, 9))
    plt.subplot(121)
    plt.imshow(mags[frame_nb], vmin=magmin, vmax=magmax, origin='lower', cmap='gray')
    plt.imshow(lanes_colored, origin='lower')
    circle1 = plt.Circle(spot_centers[frame_nb], radius=wedge_rmin, color=lcolors, ls='-', lw=1, fill=False)
    circle2 = plt.Circle(spot_centers[frame_nb], radius=wedge_rmax, color=lcolors, ls='-', lw=1, fill=False)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')

    lps = [mlines.Line2D(xyl[0], xyl[1], color=lcolors, linestyle='--', lw=1) for xyl in xyls]
    for lp in lps:
        plt.gca().add_line(lp)

    text = plt.gca().text(0.02, 0.97, dtimes[frame_nb].strftime('%x %X') + ' Frame %d'%frame_nb, fontsize=fs,
                           bbox=dict(boxstyle="square", fc='white', alpha=0.8),
                           transform=plt.gca().transAxes)

    plt.subplot(122)
    plt.imshow(mag_pols[frame_nb].T, vmin=magmin, vmax=magmax, origin='lower', extent=[phi.min(), phi.max(), 0, 512], cmap='gray')
    plt.axis([0, 360, 0, 220])
    plt.xlabel('Azimuth [degrees]')
    plt.ylabel('Radial distance [px]')
    plt.gca().axhline(y=wedge_rmin, ls='-', linewidth=1, color=lcolors)
    plt.gca().axhline(y=wedge_rmax, ls='-', linewidth=1, color=lcolors)
    for angle in phi_bins:
        plt.gca().axvline(x=angle, ymin=wedge_rmin/plt.gca().get_ybound()[1], ymax=wedge_rmax/plt.gca().get_ybound()[1], ls='--', linewidth=1, color=lcolors)

    plt.tight_layout()

    plt.savefig(os.path.join('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/wedges_plots/', 'wedgeplot_%d.png'%i))
    plt.close()


    fig, ax = plt.subplots(1,1, num=1, figsize=(6, 5))
    im = plt.imshow(mags[frame_nb], vmin=magmin, vmax=magmax, origin='lower', cmap='gray')
    plt.imshow(lanes_colored, origin='lower')
    circle1 = plt.Circle(spot_centers[frame_nb], radius=wedge_rmin, color=lcolors, ls='-', lw=1, fill=False)
    circle2 = plt.Circle(spot_centers[frame_nb], radius=wedge_rmax, color=lcolors, ls='-', lw=1, fill=False)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')

    lps = [mlines.Line2D(xyl[0], xyl[1], color=lcolors, linestyle='--', lw=1) for xyl in xyls]
    for lp in lps:
        plt.gca().add_line(lp)

    text = plt.gca().text(0.02, 0.97, dtimes[frame_nb].strftime('%x %X') + ' Frame %d'%frame_nb, fontsize=fs,
                           bbox=dict(boxstyle="square", fc='white', alpha=0.8),
                           transform=plt.gca().transAxes)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)
    cax.set_ylabel('Bz [G]')

    plt.tight_layout()

    plt.savefig(os.path.join('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/wedges_plots/','single_wedgeplot_%d.png' % i), dpi=300)
    plt.close()


# Interpolate the gaps linearly

fillg = ((4,8), (4,8), (4,8), (28, 32), (28, 32), (28, 32), (52, 56), (52, 56), (52, 56))
for i in range(len(gaps)):
    a = abs(gaps[i]-fillg[i][1])/4
    #bdrrs2[gaps[i], :] = a * bdrrs2[fillg[i][0], :] + (1-a) * bdrrs2[fillg[i][1], :]
    bmag_pol2[gaps[i], :] = a * bmag_pol2[fillg[i][0], :] + (1-a) * bmag_pol2[fillg[i][1], :]


delta_bmag = np.insert(bmag_pol2[1:,:] - bmag_pol2[0:-1,:], 0, np.zeros(bmag_pol2[1,:].shape), axis=0)

delta_bmag2 = np.insert(bmag_pol2[1:,:] - bmag_pol2[0,:], 0, np.zeros(bmag_pol2[1,:].shape), axis=0)
ncorrs = [correlate(bdrrs2[:,i], delta_bmag[:,i], mode='full', method='direct')/np.sqrt((bdrrs2[:,i]**2).sum() * (delta_bmag[:,i]**2).sum()) for i in range(len(bphi))]
#ncorrs = [correlate(bdrrs2[:,i], delta_bmag2[:,i], mode='full', method='direct')/np.sqrt((bdrrs2[:,i]**2).sum() * (delta_bmag2[:,i]**2).sum()) for i in range(len(bphi))]

ncorrs2 = np.array([np.flipud(ncorr[0:len(mags)]) for ncorr in ncorrs]).T


gaplinestyle = '--'
gapcolor = 'orange'

FS = 10
aspect = 20

fig, (ax1, ax2) = plt.subplots(1,2, num=0, figsize=(5,7))
fig.subplots_adjust(left=0.10, bottom=0.08, top=0.9, right=0.88, hspace = 0.05, wspace=0.2)
ax1.imshow(rsamples, vmin=min_radii, vmax=100, origin='lower', cmap='viridis')
ax1.set_aspect(aspect)
ax2.imshow(rsamples_binned, vmin=min_radii, vmax=100, origin='lower', cmap='viridis')
ax2.set_aspect(0.5)
plt.tight_layout()



start_frame = 3
plt.close(1)

fig, (ax1, ax2, ax3) = plt.subplots(1,3, num=1, figsize=(8,7))
fig.subplots_adjust(left=0.10, bottom=0.08, top=0.9, right=0.88, hspace = 0.05, wspace=0.2)

im1 = ax1.imshow(rsamples_binned[start_frame:,:], vmin=min_radii, vmax=100, origin='lower', cmap='viridis', extent=[phi.min(), phi.max(), 0, bdrrs2.shape[0] - start_frame])
ax1.set_aspect(aspect)
ax1.tick_params(labelsize=FS)
ax1.set_xlabel('Azimuth [degrees]')
ax1.set_ylabel('Elapsed time [hr]')

for gap in fillg:
    gap_rect = patches.Rectangle((0, gap[0]+1 - start_frame), 360, 3, linewidth=1, edgecolor='black', facecolor='white')
    ax1.add_patch(gap_rect)

rect11 = patches.Rectangle((220,8), 55, 5,linewidth=2,edgecolor='black',facecolor='none')
ax1.add_patch(rect11)

rect12 = patches.Rectangle((195, 35), 60, 3,linewidth=2,edgecolor='black',facecolor='none')
ax1.add_patch(rect12)

rect34 = patches.Rectangle((105, 47), 60, 3,linewidth=2,edgecolor='black',facecolor='none')
ax1.add_patch(rect34)

text1 = ax1.text(220, 6, '1', fontsize=fs+1)  # , bbox=dict(boxstyle="square", fc='white', alpha=0.6)
text2 = ax1.text(195, 33, '2', fontsize=fs+1)  # , bbox=dict(boxstyle="square", fc='white', alpha=0.6)
text3 = ax1.text(100, 45, '3 & 4', fontsize=fs+1)  # , bbox=dict(boxstyle="square", fc='white', alpha=0.6)

pos = ax1.get_position()
cbar_ax = fig.add_axes([pos.x0, pos.y0+pos.height+0.04, pos.width, 0.02*pos.height])
cb = plt.colorbar(im1, cax = cbar_ax, orientation='horizontal')
cb.ax.set_title(r'moat radius [px]')


#im1 = ax1.imshow(bdrrs2, cmap='seismic', vmin=-0.5, vmax=0.5, origin='lower', extent=[phi.min(), phi.max(), 0, bdrrs2.shape[0]])
#im1 = ax1.imshow(bdrrs2, cmap='seismic', vmin=-np.abs(bdrrs2).max(), vmax=np.abs(bdrrs2).max(), origin='lower', extent=[phi.min(), phi.max(), 0, bdrrs2.shape[0]])
#im1 = ax1.imshow(bdrrs, cmap='seismic', vmin=-np.abs(bdrrs2).max(), vmax=np.abs(bdrrs2).max(), origin='lower', extent=[phi.min(), phi.max(), 0, bdrrs2.shape[0]])
#im2 = ax2.imshow(bdrrs[start_frame:,:], vmin=-0.6, vmax=0.6, origin='lower', cmap='seismic', extent=[phi.min(), phi.max(), 0, bdrrs2.shape[0] - start_frame])

im2 = ax2.imshow(bdrrs[start_frame:,:], vmin=-0.5, vmax=0.5, origin='lower', cmap='RdGy_r', extent=[phi.min(), phi.max(), 0, bdrrs2.shape[0] - start_frame])
#im2 = ax2.imshow(bdrrs[start_frame:,:], vmin=-0.7, vmax=0.7, origin='lower', cmap='RdGy_r', extent=[phi.min(), phi.max(), 0, bdrrs2.shape[0] - start_frame])


ax2.set_aspect(aspect)
ax2.tick_params(labelsize=FS)
ax2.set_xlabel('Azimuth [degrees]')
#ax2.set_ylabel('Elapsed time [hr]')

for gap in fillg:
    gap_rect = patches.Rectangle((0, gap[0]+1 - start_frame), 360, 3, linewidth=1, edgecolor='black', facecolor='white')
    ax2.add_patch(gap_rect)

rect11 = patches.Rectangle((220,8), 55, 5,linewidth=2,edgecolor='black',facecolor='none')
ax2.add_patch(rect11)

rect12 = patches.Rectangle((195, 35), 60, 3,linewidth=2,edgecolor='black',facecolor='none')
ax2.add_patch(rect12)

rect34 = patches.Rectangle((105, 47), 60, 3,linewidth=2,edgecolor='black',facecolor='none')
ax2.add_patch(rect34)

text1 = ax2.text(220, 6, '1', fontsize=fs+1)  # , bbox=dict(boxstyle="square", fc='white', alpha=0.6)
text2 = ax2.text(195, 33, '2', fontsize=fs+1)  # , bbox=dict(boxstyle="square", fc='white', alpha=0.6)
text3 = ax2.text(100, 45, '3 & 4', fontsize=fs+1)  # , bbox=dict(boxstyle="square", fc='white', alpha=0.6)



pos = ax2.get_position()
cbar_ax = fig.add_axes([pos.x0, pos.y0+pos.height+0.04, pos.width, 0.02*pos.height])
cb = plt.colorbar(im2, cax = cbar_ax, orientation='horizontal')
cb.ax.set_title(r'binned $\Delta r / r$')


im3 = ax3.imshow(bmag_pol2[start_frame:,:], cmap='gray_r', vmin=0, vmax=200, origin='lower', extent=[phi.min(), phi.max(), 0, bmag_pol2.shape[0] - start_frame])
ax3.set_aspect(aspect)
ax3.tick_params(labelsize=FS)
ax3.set_xlabel('Azimuth [degrees]')

for gap in fillg:
    gap_rect = patches.Rectangle((0, gap[0] + 1 - start_frame), 360, 3, linewidth=1, edgecolor='black', facecolor='white')
    ax3.add_patch(gap_rect)


rect3 = patches.Rectangle((225,35),70,15,linewidth=2,edgecolor='red',facecolor='none')
ax3.add_patch(rect3)

rect4 = patches.Rectangle((105, 54), 60, 4,linewidth=2,edgecolor='red',facecolor='none')
ax3.add_patch(rect4)

pos = ax3.get_position()
cbar_ax = fig.add_axes([pos.x0, pos.y0+pos.height+0.04, pos.width, 0.02*pos.height])
cb = plt.colorbar(im3, cax = cbar_ax, orientation='horizontal')
cb.set_ticks(np.arange(0,201,50))
cb.ax.set_title('binned unsigned Bz [G]')


# im3 = ax3.imshow(delta_bmag, cmap='gray_r', vmin=0, vmax=34, origin='lower', extent=[phi.min(), phi.max(), 0, bmag_pol2.shape[0]])
# ax3.set_aspect(aspect)
# ax3.tick_params(labelsize=FS)
# ax3.set_xlabel('Azimuth [degrees]')
#
# for gap in fillg:
#     ax3.axhline(y=gap[0]+1, ls=gaplinestyle, linewidth=1, color=gapcolor)
#     ax3.axhline(y=gap[1], ls=gaplinestyle, linewidth=1, color=gapcolor)
#
# rect3 = patches.Rectangle((230,37),30,5,linewidth=2,edgecolor='red',facecolor='none')
# ax3.add_patch(rect3)
#
# pos3 = ax3.get_position()
# cbar3_ax = fig.add_axes([pos3.x0, pos3.y0+pos3.height+0.04, pos3.width, 0.02*pos3.height])
# cb3 = plt.colorbar(im3, cax = cbar3_ax, orientation='horizontal')
# cb3.ax.set_title('Binned emergence rate')
#
#
# im4 = ax4.imshow(ncorrs2, vmin=-0.5, vmax=0.5, origin='lower', extent=[bphi[0], bphi[-1], 0, ncorrs2.shape[0]], cmap='Spectral_r')
# #im4 = ax4.imshow(ncorrs2, extent=[bphi[0], bphi[-1], ncorrs2.shape[0], 0], cmap='Spectral_r')
# ax4.set_aspect(aspect)
# ax4.set_xlabel('Azimuth [degrees]')
# ax4.yaxis.tick_right()
# ax4.set_ylabel('Time shift [frame #]')
# ax4.yaxis.set_label_position("right")
#
# pos4 = ax4.get_position()
# cbar4_ax = fig.add_axes([pos4.x0, pos4.y0+pos4.height+0.04, pos4.width, 0.02*pos4.height])
# cb4 = plt.colorbar(im4, cax = cbar4_ax, orientation='horizontal')
# cb4.ax.set_title('Norm. Cross-correlation')
#
# maxind = np.unravel_index(np.argmax(ncorrs2, axis=None), ncorrs2.shape)
# dphi = bphi[1] - bphi[0]
# # circle_max = plt.Circle((maxind[1]*dphi, maxind[0]), radius=6, color='black', ls='-', lw=1, fill=False)
# # ax4.add_patch(circle_max)
#
# rect4 = patches.Rectangle(((maxind[1]-0.75)*dphi,maxind[0]-1),1.25*dphi,4,linewidth=2,edgecolor='black',facecolor='none')
# ax4.add_patch(rect4)


plt.savefig(os.path.join('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/wedges_plots/', 'binned_plot.png'), dpi=300)

# Normalized cross correlation:
# see https://anomaly.io/understand-auto-cross-correlation-normalized-shift/#/normalized_cross_correlation
# and https://stackoverflow.com/questions/4688715/find-time-shift-between-two-similar-waveforms


ncorrs = [correlate(bdrrs2[:,i], bmag_pol2[:,i], mode='full', method='direct')/np.sqrt((bdrrs2[:,i]**2).sum() * (bmag_pol2[:,i]**2).sum()) for i in range(len(bphi))]
ncorrs2 = np.array([np.flipud(ncorr[0:len(mags)]) for ncorr in ncorrs]).T

plt.figure(2, figsize=(6, 10))
plt.imshow(ncorrs2, vmin=-0.8, vmax=0.8, extent=[bphi[0], bphi[-1], ncorrs2.shape[0], 0], cmap='Spectral_r')
plt.gca().set_aspect(20)
plt.colorbar()
plt.tight_layout()


ncorrs = [correlate(bdrrs2[:,i], delta_bmag[:,i], mode='full', method='direct')/np.sqrt((bdrrs2[:,i]**2).sum() * (delta_bmag[:,i]**2).sum()) for i in range(len(bphi))]
ncorrs2 = np.array([np.flipud(ncorr[0:len(mags)]) for ncorr in ncorrs]).T

plt.figure(3, figsize=(4, 8))
plt.imshow(ncorrs2, vmin=-0.5, vmax=0.5, extent=[bphi[0], bphi[-1], ncorrs2.shape[0], 0], cmap='Spectral_r')
plt.gca().set_aspect(20)
plt.ylabel('Time shift [frame #]')
axc = plt.colorbar()
axc.ax.set_ylabel('Normalized cross-correlation', fontsize=FS)
plt.tight_layout()
plt.savefig(os.path.join('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/wedges_plots/', 'cross_correlation.png'))


plt.figure(4)
plt.scatter(bdrrs2[0:(-15-25),:], bmag_pol2[15:-25,:], marker='o', color='black',
            label=dtimes[0].strftime('%x %X')[0:-3] + ' - ' + dtimes[20].strftime('%x %X')[0:-3]
            ) # 'Frames [0:20] vs. [15:35]'
plt.scatter(bdrrs2[(-15-25):(-15),:], bmag_pol2[(-15-25):(-15),:], marker='o', color='red', alpha=0.5,
            label=dtimes[21].strftime('%x %X')[0:-3] + ' - ' + dtimes[46].strftime('%x %X')[0:-3]) # Frames [21:46] vs. [36:45]
plt.xlabel(r'binned $\Delta r / r$')
plt.ylabel('Binned unsigned Bz')
plt.title(r'$\Delta r / r$ vs. binned unsigned Bz (+15 hr time-shift)')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig(os.path.join('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/wedges_plots/', 'scatter_plot.png'))


frame_numbers = [44, 46, 50, 52, 56, 60]
ref = phi_vn_r_all[0][2]
vref = phi_vn_r_all[0][1]


