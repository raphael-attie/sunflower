# import glob
# import os
# import time
# import drms
from astropy.io import fits
import sunpy.map
from sunpy.sun import heliographic_solar_center
from astropy import units as u
import matplotlib.pyplot as plt
from sunpy.coordinates.frames import HeliographicCarrington


def hg_overlay(overlay, wrap180=True, spacing=10, color='white', **kwargs):
    """
    Create a heliographic overlay using wcsaxes.
    Also draw a grid and label the top axes.
    Parameters
    ----------
    overlay : `wcsaxes.coordinates_map.CoordinatesMap` object.
    wrap180 : Whether the angles should be [-180; 180] or [0-360]
    spacing: Angle [degrees] between each longitude and latitude grid lines
    color : color of the grid and ticks
    """
    #overlay = axes.get_coords_overlay('heliographic_stonyhurst')
    #overlay = axes.get_coords_overlay('heliographic_carrington')

    lon = overlay[0]
    lat = overlay[1]

    if wrap180:
        lon.coord_wrap = 180
    lon.set_major_formatter('dd')

    lon.set_ticks(spacing=spacing * u.deg, color=color)
    lat.set_ticks(spacing=spacing * u.deg, color=color)

    overlay.grid(color=color, alpha=0.5, **kwargs)

from astropy.io import fits
import sunpy.map
from sunpy.sun import heliographic_solar_center
from astropy import units as u

url = 'http://jsoc.stanford.edu/SUM93/D918158747/S00000/hmi.m_45s.20101127_150000_TAI.2.magnetogram.fits'
hdu = fits.open(url)
hdu.verify('silentfix')
header = hdu[1].header
data = hdu[1].data
smap = sunpy.map.Map(data, header).rotate(angle=180.0* u.deg, order=1)
print(smap.dimensions)
# Print dimensions of the raw image data
print(data.shape)


# Define a region of interest
length = 250 * u.arcsec
x0 = -320 * u.arcsec
y0 = 210 * u.arcsec
# Define a region to highlight with a box
bottom_left = u.Quantity([x0 - length/2, y0 - length/2])

# Extract submap from smap
submap = smap.submap(u.Quantity([x0 - length/2, x0 + length/2]),
                     u.Quantity([y0 - length/2, y0 + length/2]))

# Draw a box on the image


# Figure size (inches)
w = 16.0
h = 11.0
# margin at bottom of figure (inches)
blmx, blmy = 1, 0.5
# axes size and positions of the map
L0 = 10 # inches
p0 = [blmx/w, blmy/h, L0/w, L0/h]
# Space between axes (inches)
s = 1.3
# axes square size for the submaps(inches)
L = (L0 - 2*s)/3


# Horizontal space between full map and column of submaps (normalized)
dh = (L0 + s)/w
# axes positions of the three submaps (x,y,w,h) (normalized), from top to bottom
p1 = [p0[0] + dh, (blmy + 2*s + 2*L)/h, L/w, L/h]
p2 = [p0[0] + dh, (blmy + s + L)/h, L/w, L/h]
p3 = [p0[0] + dh, blmy/h, L/w, L/h]


fig1 = plt.figure(1, figsize=(w,h))
ax0 = fig1.add_axes(p0, projection=smap)
ax1 = fig1.add_axes(p1, projection=submap)
ax2 = fig1.add_axes(p2, projection=submap)
ax3 = fig1.add_axes(p3, projection=submap)

# Full Sun
smap.plot(axes=ax0, vmin=-100, vmax=100)
smap.draw_rectangle(bottom_left, length, length)
# Overlay heliographic grid
overlay = ax0.get_coords_overlay('heliographic_stonyhurst')
hg_overlay(overlay, color='yellow', linewidth=1)
overlay[0].set_axislabel('')
overlay[1].set_axislabel('Latitude')

# Submaps (1 Stonyhurst, 1 Carrington [-180; 180], 1 Carrington [0; 360]


submap.plot(axes=ax1, vmin=-100, vmax=100, title=False)
# Stonyhurst frame - wrap 180 degrees
overlay_Stonyhurst = ax1.get_coords_overlay('heliographic_stonyhurst')
hg_overlay(overlay_Stonyhurst, spacing=5, color='yellow', linewidth=2)
overlay_Stonyhurst[0].set_axislabel('Longitude')
overlay_Stonyhurst[1].set_axislabel('Latitude')
# Text annotations
textpos = (0.1, 0.93)
fontsize = 12
plt.annotate('Stonyhurst', textpos, xycoords='axes fraction', color='yellow', fontsize=fontsize)

# Carrington frame - wrap 180 degrees [-180; 180]
L0 = heliographic_solar_center(header['DATE-OBS'])[0].value

submap.plot(axes=ax2, vmin=-100, vmax=100, title=False)
overlay_Carrington1 = ax2.get_coords_overlay(HeliographicCarrington(dateobs=header['DATE-OBS']))
hg_overlay(overlay_Carrington1, wrap180=True, spacing=5,  color='yellow', linewidth=2)
overlay_Carrington1[0].set_axislabel('Longitude')
overlay_Carrington1[1].set_axislabel('Latitude')
plt.annotate('Carrington (L0=%0.2f)' %L0, textpos, xycoords='axes fraction', color='yellow', fontsize=fontsize)
plt.annotate('"Wrapped" ', (0.1, 0.85), xycoords='axes fraction', color='yellow', fontsize=fontsize)

## Carrington frame - No wrap 180 degrees [0; 360]
submap.plot(axes=ax3, vmin=-100, vmax=100, title=False)
overlay_Carrington2 = ax3.get_coords_overlay(HeliographicCarrington(dateobs=header['DATE-OBS']))
hg_overlay(overlay_Carrington2, wrap180=False, spacing=5, color='yellow', linewidth=2)
overlay_Carrington1[0].set_axislabel('Longitude')
overlay_Carrington1[1].set_axislabel('Latitude')
plt.annotate('Carrington (L0=%0.2f)' %L0, textpos, xycoords='axes fraction', color='yellow', fontsize=fontsize)

plt.savefig('/Users/rattie/Desktop/figure.png')

#plt.text(-0.5, -0.25, 'Brackmard minimum')

#'Carrington



# TODO     Check usage of SkyCoord
# hg_c = SkyCoord(uset_longitudes[i] * u.deg, uset_latitudes[i] * u.deg, frame = 'heliographic_carrington', dateobs = time_series[i])

# def load_data(file):
#
#     hdu = fits.open(file)
#     hdu.verify('silentfix')
#     data = hdu[1].data
#     header = hdu[1].header
#     return data, header
#
# def load_map(data, header):
#
#     rmap = sunpy.map.Map(data, header).rotate(angle=header['CROTA2'] * u.deg)
#     return rmap
#
#
# def fetch_pair(files, ind1, ind2):
#
#     data1, header1 = load_data(files[ind1])
#     data2, header2 = load_data(files[ind2])
#
#     return data1, header1, data2, header2
#
# def rpeek(rmap, **matplot_args):
#     # TODO me: Try to get this in Stonyhurst or Carrington coordinates...
#     axes = wcsaxes_compat.gca_wcs(rmap.wcs)
#     rmap.plot(axes, **matplot_args)
#
#
#
# data_dir = '/Users/rattie/Data/SDO/HMI/Nov_28_2011'
# # Query string
# qs      = 'hmi.M_45s[2010.11.27_TAI/3d@3h]{magnetogram}'
#
#
# c = drms.Client(email='attie.raphael@gmail.com', verbose=True)
# r = c.export(qs, method='url', protocol='fits')
# urls = r.urls.url
#
#
# rmaps = []
# for url in urls[::2]:
#     data, header = load_data(url)
#     rmaps.append(load_map(data, header))
#
#
#
# plt.figure(0)
# plt.subplot(121)
# rpeek(rmaps[2], vmin=-100, vmax=100)
# plt.subplot(122)
# rpeek(rmaps[5], vmin=-100, vmax=100)
#
#
# # SHARP
# # url="http://jsoc.stanford.edu/doc/data/hmi/harp/harp_definitive/2012/03/06/harp.2012.03.06_23:00:00_TAI.png"
#
# # Try wcsaxes for overlaying more custom coordinate system grids
# data, header = load_data(urls[0])
# smap = rmaps[0]
# wcs = WCS(header)
#
# fig0 = plt.figure(0)
# ax = fig0.add_axes([0.15, 0.1, 0.8, 0.8], projection=wcs)
# ax.set_xlim(-0.5, data.shape[1] - 0.5)
# ax.set_ylim(-0.5, data.shape[0] - 0.5)
# ax.imshow(data, vmin=-100, vmax=100, cmap='gray')


# from astropy.io import fits
# import sunpy.map
# from astropy import units as u
# import matplotlib.pyplot as plt




print("done")