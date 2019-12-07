from astropy.coordinates import SkyCoord
import sunpy.map
from sunpy.sun import heliographic_solar_center
from astropy import units as u
from sunpy.coordinates.frames import HeliographicCarrington
from sunpy.coordinates import frames

from sunpy.coordinates.ephemeris import get_sun_L0, get_sun_B0

# def get_harvey_lon(date, radians=False):
#     """
#     Need to update the rotation period to Carrington (~27-ish days) and the Carrington reference date, which is currently is the number:
#     2415023.5 in julian day. The date needs to be of type  astropy.time.core.Time
#     :param date:
#     :param radians:
#     :return:
#     """
#     # 2415023.5 JD = Jan 4, 1900 => 1st Harvey Rotation
#     # 1 Harvey Rotation => 360 degrees in 33 days
#
#
#
#     if not isinstance(date, astropy.time.core.Time):
#         raise ValueError('Input needs to be an astropy time object.')
#
#     if radians:
#         return Longitude([math.radians(((360. / 33.) * (date.jd - 2415023.5)) - (np.floor(((360. / 33.) * (date.jd - 2415023.5)) / 360.) * 360.))] * u.rad)
#     else:
#         return Longitude([((360. / 33.) * (date.jd - 2415023.5)) - (np.floor(((360. / 33.) * (date.jd - 2415023.5)) / 360.) * 360.)] * u.deg)



# Set the helioprojective coordinates of a region of interest at a given time
x0 = 0 * u.arcsec
y0 = 0 * u.arcsec
center_hpc = SkyCoord(x0, y0, frame=frames.Helioprojective, obstime="2011-06-27T20:00:00")
# Convert to Carrington
center_hgc = center_hpc.transform_to(frames.HeliographicCarrington)
print(center_hgc)

L0 = get_sun_L0(time='2011-06-27T22:00:00').to_value(u.deg)


c = SkyCoord(-7*u.deg, 60*u.deg, frame=frames.HeliographicStonyhurst, obstime="2011-06-27T20:00:00")
print(c.transform_to(frames.HeliographicCarrington))

c2 = SkyCoord(60*u.deg, -7*u.deg, frame=frames.HeliographicStonyhurst, obstime="2011-06-27T20:00:00")
print(c2.transform_to(frames.HeliographicCarrington))