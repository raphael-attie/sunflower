"""
Script testing the dowhill-clumping algorithm.
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use('macosx')
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import seaborn as sns
import fitstools
from image_processing import segmentation
import skimage.measure
import skimage.morphology
from scipy.ndimage import gaussian_filter
import mahotas
from skimage.feature import peak_local_max
from scipy.ndimage.morphology import distance_transform_edt
import balltracking.mballtrack as mblt
from operator import add

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * sig**2))


def custom_cmap(ncolors):

    # ## Custom colors => This must add a unique color for the background
    colors = plt.cm.Set1_r(np.linspace(0, 1, 9))
    # colors = plt.cm.Dark2(np.linspace(0, 1, 8))

    cmap = matplotlib.colors.ListedColormap(colors, name='mycmap', N=ncolors)
    colors2 = np.array([cmap(i) for i in range(ncolors)])
    # Add unique background color
    # light gray color
    gray = np.array([[220, 220, 220, 255]]) / 255
    colors2 = np.concatenate((gray, colors2), axis=0)
    cmap2 = matplotlib.colors.ListedColormap(colors2, name='mycmap2')

    return cmap2


def custom_cmap2(ncolors):
    np.random.seed(seed=1)
    colors1 = plt.cm.Set1_r(np.linspace(0, 1, int(round(ncolors/3))))
    colors1 = np.random.permutation(colors1)
    colors2 = plt.cm.Dark2(np.linspace(0, 1, int(round(ncolors/3))))
    colors2 = np.random.permutation(colors2)
    colors3 = plt.cm.tab10(np.linspace(0, 1, int(round(ncolors/3))))
    colors3 = np.random.permutation(colors3)
    colorsc = np.concatenate((colors1, colors2, colors3), axis=0)
    colorsc = np.random.permutation(colorsc)
    # Add unique background color
    # light gray color
    gray = np.array([[220, 220, 220, 255]]) / 255
    colors4 = np.concatenate((gray, colorsc), axis=0)
    cmap = matplotlib.colors.ListedColormap(colors4, name='mycmap')

    return cmap


def make2DGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def detect2(magnetogram, threshold=50):
    magnetogram = np.ma.masked_invalid(magnetogram.astype(np.float64)).filled(0)
    m_pos = np.ma.masked_less(magnetogram, 0).filled(0)
    m_neg = np.ma.masked_less(-magnetogram, 0).filled(0)

    labels_pos = segmentation.detect_polarity(m_pos, float(threshold))
    # labels_neg = segmentation.detect_polarity(m_neg, float(threshold))
    # The segmentation above only assign the integer values to the grown regions,
    # but two different integer values can still be assigned to connected pixels
    # Thus proper connected component labeling must be applied.
    #labels_pos = np.ma.masked_less(skimage.measure.label(labels_pos, neighbors=4, background=0), 0).filled(0)
    # labels_neg = np.ma.masked_less(skimage.measure.label(labels_neg, neighbors=4, background=0), 0).filled(0)
    # skimage.morphology.remove_small_objects(labels_pos, 6, in_place=True)
    # skimage.morphology.remove_small_objects(labels_neg, 6, in_place=True)

    return labels_pos


def find_extrema(data, threshold):

    segmented_dh = detect2(data, threshold)
    labels_dh = skimage.measure.label(segmented_dh, neighbors=8, background=0)
    labels_dhR = skimage.morphology.remove_small_objects(labels_dh.copy(), min_size=6)
    labels_dhR_relabelled = skimage.measure.label(labels_dhR, neighbors=8, background=0)
    ymax, xmax = np.transpose( peak_local_max(data, indices=True, labels=labels_dhR_relabelled, num_peaks_per_label=1))

    return xmax, ymax

def find_extrema2(data, threshold, selem=None):
    # selem only used for closing
    segmented_dh = detect2(data, threshold)
    labels_dh = skimage.measure.label(segmented_dh, neighbors=8, background=0)
    labels_dhR = skimage.morphology.remove_small_objects(labels_dh.copy(), min_size=6)
    labels_dhR_relabelled = skimage.measure.label(labels_dhR, neighbors=8, background=0)
    # Region props
    rprops = skimage.measure.regionprops(labels_dhR_relabelled, intensity_image=data.astype(np.int32))
    # Find the maxima in each region
    if selem is not None:
        ymax, xmax = zip(*[
            map(add, np.unravel_index(np.argmax(mahotas.close(prop.intensity_image, Bc=selem)), prop.intensity_image.shape),
                prop.bbox[0:2]) for prop in rprops])
    else:
        ymax, xmax = zip(*[
            map(add,
                np.unravel_index(np.argmax(prop.intensity_image), prop.intensity_image.shape),
                prop.bbox[0:2]) for prop in rprops])

    return xmax, ymax


def find_extrema3(data, threshold, selem):
    segmented_dh = detect2(data, threshold)
    labels_dh = skimage.measure.label(segmented_dh, neighbors=8, background=0)
    labels_dhR = skimage.morphology.remove_small_objects(labels_dh.copy(), min_size=6)
    labels_dhR_relabelled = skimage.measure.label(labels_dhR, neighbors=8, background=0)
    # Region props
    rprops = skimage.measure.regionprops(labels_dhR_relabelled, intensity_image=data.astype(np.int32))
    # Find the maxima in each region
    ymax, xmax = zip(*[
        map(add, np.unravel_index(np.argmax( prop.filled_image), prop.intensity_image.shape),
            prop.bbox[0:2]) for prop in rprops])

    return xmax, ymax

# gpos = make2DGaussian(100, 20) * 200
# gneg = -make2DGaussian(50, 10) * 200

gpos1 = make2DGaussian(100, 20) * 200
gpos2 = make2DGaussian(50, 10) * 200

# # Load an HMI magnetogram

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'

tslice = 4
data = fitstools.fitsread(datafile, tslice=tslice).astype(np.float32)
datasm = gaussian_filter(data, sigma=1.2)
datamax = np.abs(data).max()
# datamask = np.abs(data) < 200
# datasm[datamask] = data[datamask]
# data = np.zeros([300, 300])
# # data[10:110, 10:110] = gpos
# # data[120:170, 120:170] = gneg
# data[10:110, 10:110] = gpos1
# data[120:170, 120:170] = gpos2

# magnetic threshold for the downhill algorithm
threshold = 25

segmented_dh = detect2(datasm, threshold)
labels_dh = skimage.measure.label(segmented_dh, neighbors=8, background=0)
labels_dhR = skimage.morphology.remove_small_objects(labels_dh.copy(), min_size=9)
labels_dhR_relabelled = skimage.measure.label(labels_dhR, neighbors=8, background=0)
borders = skimage.segmentation.find_boundaries(labels_dhR_relabelled)
border_lines = np.where(borders)

# Local maxima in downhill region-grown labelled data
ypmax2, xpmax2 = np.transpose( peak_local_max(data, indices=True, labels=labels_dh, num_peaks_per_label=1))
ypmax2R, xpmax2R = np.transpose( peak_local_max(data, indices=True, labels=labels_dhR_relabelled, num_peaks_per_label=1))

# Region props
rprops = skimage.measure.regionprops(labels_dhR_relabelled, intensity_image=data.astype(np.int32))
# Get centroid of each region
yc, xc = zip(*[prop.centroid for prop in rprops])
# Get maximum of each region
yrpmax, xrpmax = zip(*[ map(add, np.unravel_index(np.argmax(prop.intensity_image), prop.intensity_image.shape), prop.bbox[0:2])  for prop in rprops ])
# Get maximum of each filled region
se1 = np.ones([3,3], dtype=np.bool)#skimage.morphology.disk(2)
# 32.9 ms +/- 1 ms
yrpmax_filled, xrpmax_filled = zip(*[ map(add, np.unravel_index(np.argmax( skimage.morphology.closing(prop.intensity_image, se1) ), prop.intensity_image.shape), prop.bbox[0:2])  for prop in rprops ])
# 10.1 ms +/- 459 us , 3 x faster than with skimage closing method. About 3 times slower than with no closing at all (~ 3 ms )
yrpmax_filled2, xrpmax_filled2 = zip(*[ map(add, np.unravel_index(np.argmax( mahotas.close(prop.intensity_image, Bc=se1) ), prop.intensity_image.shape), prop.bbox[0:2])  for prop in rprops ])


def close_region(rprop):
    region = data[rprop.bbox[0]:rprop.bbox[2], rprop.bbox[1]:rprop.bbox[3]].copy()
    region[~rprop.filled_image] = 0
    return region


def argmax_from_close_region(rprop):
    region = data[rprop.bbox[0]:rprop.bbox[2], rprop.bbox[1]:rprop.bbox[3]].copy()
    region[~rprop.filled_image] = 0
    ymax, xmax = map(add, np.unravel_index(np.argmax(region), region.shape), rprop.bbox[0:2])
    return xmax, ymax

def argmax_from_filtered_region(rprop):
    region = data[rprop.bbox[0]:rprop.bbox[2], rprop.bbox[1]:rprop.bbox[3]].copy()
    region[~rprop.filled_image] = 0
    max_coords_local =  np.unravel_index(np.argmax(region), region.shape)
    ymax, xmax = map(add, max_coords_local, rprop.bbox[0:2])
    # Reject those too close to the border of the region
    # if (rprop.equivalent_diameter > 8) and ( (xmax == rprop.bbox[1]) or (xmax == rprop.bbox[3]-1) or (ymax == rprop.bbox[0]) or (ymax == rprop.bbox[2]-1) ):
    #     xmax, ymax = np.nan, np.nan
    # else:
    #     dist_borders = distance_transform_edt(region)
    #     if dist_borders[max_coords_local[0], max_coords_local[1]] < 2:
    #         xmax, ymax = np.nan, np.nan

    return xmax, ymax

xrpmax4, yrpmax4 = zip(*list(map(argmax_from_close_region, rprops)))
xrpmax5, yrpmax5 = zip(*list(map(argmax_from_filtered_region, rprops)))

label = 103
sample = close_region(rprops[label])
dist_sample = distance_transform_edt(sample)

ym1, xm1 = np.unravel_index(np.argmax( rprops[label].intensity_image ), rprops[label].intensity_image.shape)
ym2, xm2 = np.unravel_index(np.argmax( sample ), sample.shape)

fig, axs = plt.subplots(1,6, figsize=(14,6))

axs[0].imshow( rprops[label].image )
axs[1].imshow( rprops[label].intensity_image)
axs[1].plot(xm1, ym1, color='black', marker='.', ls = 'none', ms =2)
axs[2].imshow( rprops[label].filled_image )
axs[3].imshow( sample )
axs[3].plot(xm1, ym1, color='black', marker='.', ls = 'none', ms =2)
axs[3].plot(xm2, ym2, color='blue', marker='.', ls = 'none', ms =10, markerfacecolor='none')
axs[4].imshow(dist_sample, cmap='gray', interpolation='nearest')
axs[5].imshow(dist_sample, cmap='gray', interpolation='bilinear')
plt.tight_layout()


fig, axs = plt.subplots(1,2, sharex=True,sharey=True, figsize=(18,10))
axs[0].imshow(data, vmin= -0.1*datamax, vmax=0.1*datamax)
axs[0].plot(border_lines[1], border_lines[0], color='black', marker='.', ls = 'none', ms=1, lw=0.5)
axs[1].imshow(labels_dhR_relabelled, cmap=custom_cmap2(labels_dhR_relabelled.max()), vmin=0, vmax=labels_dhR_relabelled.max())
#axs[1].plot(xpmax2R, ypmax2R, color='black', marker='.', ls = 'none', ms =2, label='%d local max (Downhill curated)'%xpmax2R.max())
axs[1].plot(xrpmax, yrpmax, color='red', marker='.', ls = 'none', ms =8, markerfacecolor='none', label='argmax(rprops.intensity_image)')
axs[1].plot(xrpmax4, yrpmax4, color='blue', marker='.', ls = 'none', ms =12, markerfacecolor='none', label='argmax from closed region (rprops.filled_image)')
axs[1].plot(xrpmax5, yrpmax5, color='cyan', marker='.', ls = 'none', ms =16, markerfacecolor='none', label='argmax from closed region (rprops.filled_image)')

plt.legend()
plt.title('Downhill curated with remove_small_object()')

[axs[i].set_adjustable('box-forced') for i in range(2)]

plt.tight_layout()


labels_pos_labR2bool = skimage.morphology.remove_small_holes(labels_pos_labR.copy(), min_size=6)
labels_pos_labR2lab = skimage.measure.label(labels_pos_labR2bool, neighbors=8, background=0)



zoom = [150, 500, 100, 450]

fig, axs = plt.subplots(1,4, sharex=True,sharey=True, figsize=(18,6))

axs[0].imshow(labels_pos_lab, cmap=custom_cmap(labels_pos_lab.max()), vmin=0, vmax=labels_pos_lab.max())
axs[0].set_title('Downhill region-grow')

axs[1].imshow(labels_pos_labR, cmap=custom_cmap(labels_pos_lab.max()), vmin=0, vmax=labels_pos_lab.max())
axs[1].set_title('remove_small_objects()')

axs[2].imshow(labels_pos_labR2bool, cmap='gray', vmin=0, vmax=1)
axs[2].set_title('remove_small_holes() [USELESS - loss of segmentation]')

axs[3].imshow(labels_pos_labR2lab, cmap=custom_cmap(labels_pos_lab.max()), vmin=0, vmax=labels_pos_lab.max())
axs[3].set_title('labelled [USELESS - loss of segmentation]')

[axs[i].set_adjustable('box-forced') for i in range(4)]
axs[0].axis(zoom)

plt.tight_layout()

# axs[0].autoscale(enable=False, tight=True)
# axs[0].set_aspect('equal', 'datalim')
# axs[0].axis(zoom)
# axs[0].set_aspect('equal', 'datalim')

#hmax = skimage.morphology.h_maxima(data, 25)
# a = np.array([[0, 0, 0, 0, 0],
#               [0, 0, 2, 0, 0],
#               [0, 0, 2, 0, 0],
#               [0, 0, 3, 0, 0],
#               [0, 0, 3, 0, 0],
#               [0, 0, 0, 0, 0]])
#
# rmaxa = mahotas.regmax(a)

#TODO compare downhill region-grow & marker-watershed

data2 = data.copy()
data2[data < threshold] = 0

min_distance = 8
se = skimage.morphology.disk(min_distance/2)
#se = np.ones([min_distance, min_distance])

# Regional max (139 ms +/- 5.85 ms per loop)
rmax = mahotas.regmax(data2, se)
yrmax, xrmax = np.where(rmax)
# Local max (68.7 ms +/- 1.81 ms per loop)
lmax = skimage.morphology.local_maxima(data2, se)
ylmax, xlmax = np.where(lmax)
# peak local max (34.5 ms +/- 012 us ms per loop)
ypmax, xpmax = np.transpose(peak_local_max(data2, indices=True, footprint=se))

# Marker-based watershed
wlabels, wmarkers, borders = mblt.marker_watershed(data, xpmax, ypmax, threshold, 1)
wlabels +=1
wlabelsR = skimage.morphology.remove_small_objects(wlabels.copy(), min_size=6)
ypmaxR, xpmaxR = np.array( peak_local_max(data, indices=True,labels=wlabelsR.copy(), num_peaks_per_label=1)).T

fig2, axs2 = plt.subplots(2,3, sharex=True,sharey=True, figsize=(18,10))

axs2[0,0].imshow(data, cmap='gray', vmin=-datamax, vmax=datamax)
axs2[0,0].plot(xrmax, yrmax, color='red', marker='.', ls = 'none', ms =4, label='regional max (mahotas)')
axs2[0,0].plot(xlmax, ylmax, color='cyan', marker='.', ls = 'none', ms =8, markerfacecolor='none', label='local max (skimage)')
axs2[0,0].plot(xpmax, ypmax, color='blue', marker='.', ls = 'none', ms =12, markerfacecolor='none', label='peak max (skimage)')
axs2[0,0].legend()
axs2[0,0].set_title('Frame #%d compare local maxima'%tslice)

axs2[0,1].imshow(labels_pos_lab, cmap=custom_cmap2(labels_pos_lab.max()), vmin=0, vmax=labels_pos_lab.max())
axs2[0,1].plot(xpmax2, ypmax2, color='purple', marker='.', ls = 'none', ms =2, label='local max of %d regions'%xpmax2.size)
axs2[0,1].legend()
axs2[0,1].set_title('Downhill region-grow')

axs2[0,2].imshow(labels_pos_labR_relabelled, cmap=custom_cmap2(labels_pos_labR_relabelled.max()), vmin=0, vmax=labels_pos_labR_relabelled.max())
axs2[0,2].plot(xpmax2R, ypmax2R, color='purple', marker='.', ls = 'none', ms =2, label='%d local max (Downhill curated)'%xpmax2R.max())
axs2[0,2].legend()
axs2[0,2].set_title('Downhill curated with remove_small_object()')

axs2[1,0].imshow(data, cmap='gray', vmin=-datamax, vmax=datamax)
axs2[1,0].plot(xpmax, ypmax, color='blue', marker='.', ls = 'none', ms =12, markerfacecolor='none', label='%d peak local max from threshold (skimage)'%xpmax.size)
axs2[1,0].plot(xpmax2R, ypmax2R, color='purple', marker='.', ls = 'none', ms =2, label='%d local max in Downhill curated'%xpmax2R.size)
axs2[1,0].legend()
axs2[1,0].set_title('Compare peak local max & Downhill curated')

axs2[1,1].imshow(wlabels, cmap=custom_cmap2(wlabels.max()), vmin=0, vmax=wlabels.max())
axs2[1,1].plot(xpmax, ypmax, color='blue', marker='.', ls = 'none', ms =2, label='%d peak local max from threshold (skimage)'%xpmax.size)
axs2[1,1].legend()
axs2[1,1].set_title('Watershed marked with %d peak local maxima'%xpmax.size)

axs2[1,2].imshow(wlabelsR, cmap=custom_cmap2(wlabels.max()), vmin=0, vmax=wlabels.max())
axs2[1,2].plot(xpmaxR, ypmaxR, color='blue', marker='.', ls = 'none', ms =2, label='%d local max in Watershed curated'%xpmaxR.size)
axs2[1,2].legend()
axs2[1,2].set_title('Watershed curated, %d regions'%xpmaxR.size)

[axs2[i,j].set_adjustable('box-forced') for i in range(2) for j in range(3)]

plt.tight_layout()


plt.subplot(245)
#plt.imshow(rmax, cmap='gray', vmin=0, vmax=1)
plt.imshow(data, cmap='gray', vmin=-0.1*datamax, vmax=0.1*datamax)
plt.plot(xrmax, yrmax, color='red', marker='.', ls = 'none', ms =4, label='regional max (mahotas)')
plt.plot(xlmax, ylmax, color='cyan', marker='.', ls = 'none', ms =8, markerfacecolor='none', label='local max (skimage)')
plt.plot(xpmax, ypmax, color='blue', marker='.', ls = 'none', ms =12, markerfacecolor='none', label='peak max (skimage)')
plt.legend()

plt.subplot(246)
#plt.imshow(labels_rmaxR, cmap=custom_cmap(labels_pos_lab.max()), vmin=0, vmax=labels_pos_lab.max())
plt.imshow(data, cmap='Spectral', vmin=-0.1*datamax, vmax=0.1*datamax)
plt.plot(xlmax, ylmax, color='red', marker='.', ls = 'none', ms =2)
plt.title('Local max (skimage)')

plt.subplot(247)
plt.imshow(labels_pos_labR, cmap=custom_cmap(labels_pos_lab.max()), vmin=0, vmax=labels_pos_lab.max())

plt.subplot(248)
plt.imshow(labels_pos_labR2, cmap=custom_cmap(labels_pos_lab.max()), vmin=0, vmax=labels_pos_lab.max())
plt.plot(xpmax2, ypmax2, color='blue', marker='.', ls = 'none', ms =4, markerfacecolor='none', label='peak max (skimage)')
plt.title('Downhill region with local max')

plt.tight_layout()


# TODO: investigate shape_index to reject some local maxima not suitable for tracking
# TODO: or see usage of removing the small area, filling the hole, and relabel? Filling hole does not work as expected...
# TODO: Use peak_local_max using labelled image and num_peaks_per_label. Does not work as expected (the local max are more than one per label)