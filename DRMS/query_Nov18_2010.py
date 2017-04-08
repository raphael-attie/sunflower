import glob
import os
import time
import drms

import matplotlib
# Use a backend that can suppress screen display
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import sunpy.map
from sunpy.sun import heliographic_solar_center
from astropy import units as u
from sunpy.coordinates.frames import HeliographicCarrington


#plt.ioff()

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

    lon = overlay[0]
    lat = overlay[1]

    if wrap180:
        lon.coord_wrap = 180
    lon.set_major_formatter('dd')

    lon.set_ticks(spacing=spacing * u.deg, color=color)
    lat.set_ticks(spacing=spacing * u.deg, color=color)

    overlay.grid(color=color, alpha=0.5, **kwargs)

# Directory where the files will be downloaded
data_dir = '/Users/rattie/Data/SDO/HMI/Nov_28_2010_3hr'

# # Instantiate a drms client
# c = drms.Client(email='raphael.attie@nasa.gov', verbose=True)
#
# # Query string for the data export request. Magnetograms from 2010/11/27 , during 3 days, every 3 hours.
# qs      = 'hmi.M_45s[2010.11.27_TAI/3d@3h]{magnetogram}'
#
# # Export request - fits files with headers
# r = c.export(qs, method='url', protocol='fits')
#
# # Download the data
# r.download(data_dir)

# Get the data from the local data directory
files   = glob.glob(os.path.join(data_dir, '*.fits'))


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

# Use the file #5 to determine a good location of the AR at that time and get these coordinates in Carrington to have tracked location
file = files[5]

hdu = fits.open(file)
hdu.verify('silentfix')
data = hdu[1].data
header = hdu[1].header

# Set the helioprojective coordinates of a region of interest at a given time
length = 250 * u.arcsec
x0 = -320 * u.arcsec
y0 = 210 * u.arcsec
center_hpc = SkyCoord(x0, y0, frame='helioprojective', dateobs=header['DATE-OBS'])
# Convert to Carrington
center_HGC = center_hpc.heliographic_carrington


fig1 = plt.figure(1, figsize=(w,h))

#for file in files[0]:
file = files[5]

hdu = fits.open(file)
hdu.verify('silentfix')
data = hdu[1].data
header = hdu[1].header

smap = sunpy.map.Map(data, header).rotate(angle=header['CROTA2'] * u.deg)
smap.

# Set the Carrington coordinates from above, but at the new time.
center_HGC2 = SkyCoord(center_HGC.lon, center_HGC.lat, frame='heliographic_carrington', dateobs=header['DATE-OBS'])
# Convert back to helioprojective. The Skycoord class should take that new time into account when converting to helioprojective
center_hpc2 = center_HGC2.helioprojective
# Define the helioprojective coordinates of a region of interest at a given time
x0 = center_hpc2.Tx
y0 = center_hpc2.Ty
# Box to show the ROI
bottom_left = u.Quantity([x0 - length/2, y0 - length/2])
# Extract submap from smap
submap = smap.submap(u.Quantity([x0 - length/2, x0 + length/2]),
                     u.Quantity([y0 - length/2, y0 + length/2]))



fig1.clf()

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

fname = os.path.splitext(file)[0] + '.png'
#plt.savefig(fname)


print("done")