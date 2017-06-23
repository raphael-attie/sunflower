import copy
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.data.sample import AIA_171_IMAGE
import numpy as np
from astropy import units as u
from reproject import reproject_interp
# reproject module from https://github.com/astrofrog/reproject
from astropy.coordinates import SkyCoord
from sunpy import wcs
import sunpy.sun
from sunpy.coordinates import HeliographicCarrington

# Test Skycoord coordinate transforms
# sc_car1 = SkyCoord(0 * u.deg, 0 * u.deg, frame = 'heliographic_stonyhurst', dateobs = '2011-03-19T10:54:00')
# sc_car1.transform_to(frame=HeliographicCarrington(dateobs='2011-03-22T10:54:00'))


#
sc_car2 = SkyCoord(0 * u.deg, 0 * u.deg, frame = 'heliographic_carrington', dateobs = '2011-03-19T10:54:00')
sc_car2.transform_to(frame='heliographic_stonyhurst')

sc_car3 = SkyCoord(0 * u.deg, 0 * u.deg, frame = 'heliographic_carrington', dateobs = '2011-03-21T10:54:00')
sc_car3.transform_to(frame='heliographic_stonyhurst')


aia = sunpy.map.Map(AIA_171_IMAGE)
shape_out = [360,720]
wcs_out = aia.wcs.deepcopy()
# CAR = CARRE!!!!
#wcs_out.wcs.ctype = ['HGLN-CAR', 'HGLT-CAR']
wcs_out.wcs.ctype = ['CRLN-CAR', 'CRLT-CAR']

wcs_out.wcs.crval = [0, 0]
wcs_out.wcs.crpix = np.array(shape_out)[::-1]/2.
wcs_out.wcs.cunit = ['deg', 'deg']
wcs_out.wcs.cdelt = [0.5,0.5]
wcs_out.wcs.pc = np.identity(2)

#wcs_out.wcs.dateobs='2011-03-19T10:54:00.340000'

output, footprint = reproject_interp((aia.data, aia.wcs), wcs_out, shape_out)
out_header = copy.deepcopy(aia.meta)
out_header.update(wcs_out.to_header())
outmap = sunpy.map.Map((output, out_header))

fig = plt.figure()
ax = plt.subplot(211, projection=outmap)
outmap.plot()
lon, lat = ax.coords
lon.set_ticks(spacing=30.*u.deg)
lon.set_major_formatter("d.d")
lat.set_major_formatter("d.d")

#wcs_out.wcs.dateobs='2011-03-22T10:54:00.340000'
# Give CRVAL the value of L0
wcs_out.wcs.crval = [sunpy.sun.heliographic_solar_center('2011-03-22T10:54:00')[0].value, 0]

output, footprint = reproject_interp((aia.data, aia.wcs), wcs_out, shape_out)
out_header = copy.deepcopy(aia.meta)
out_header.update(wcs_out.to_header())
outmap = sunpy.map.Map((output, out_header))


ax = plt.subplot(212, projection=outmap)
ax.set_title('')
outmap.plot()
lon, lat = ax.coords
lon.set_ticks(spacing=30.*u.deg)
lon.set_major_formatter("d.d")
lat.set_major_formatter("d.d")

plt.show()


