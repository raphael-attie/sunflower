# import glob
# import os
# import time
import drms
from astropy.io import fits
import sunpy.map
import sunpy.visualization.wcsaxes_compat as wcsaxes_compat
from astropy import units as u
import matplotlib.pyplot as plt

def load_data(file):

    hdu = fits.open(file)
    hdu.verify('silentfix')
    data = hdu[1].data
    header = hdu[1].header
    return data, header

def load_map(data, header):

    rmap = sunpy.map.Map(data, header).rotate(angle=header['CROTA2'] * u.deg)
    return rmap


def fetch_pair(files, ind1, ind2):

    data1, header1 = load_data(files[ind1])
    data2, header2 = load_data(files[ind2])

    return data1, header1, data2, header2

def rpeek(rmap, **matplot_args):
    # TODO me: Try to get this in Stonyhurst or Carrington coordinates...
    axes = wcsaxes_compat.gca_wcs(rmap.wcs)
    rmap.plot(axes, **matplot_args)



data_dir = '/Users/rattie/Data/SDO/HMI/Nov_28_2011'
# Query string
qs      = 'hmi.M_45s[2010.11.27_TAI/3d@3h]{magnetogram}'


c = drms.Client(email='attie.raphael@gmail.com', verbose=True)
r = c.export(qs, method='url', protocol='fits')
urls = r.urls.url


rmaps = []
for url in urls[::2]:
    data, header = load_data(url)
    rmaps.append(load_map(data, header))



plt.figure(0)
plt.subplot(121)
rpeek(rmaps[2], vmin=-100, vmax=100)
plt.subplot(122)
rpeek(rmaps[5], vmin=-100, vmax=100)

fig = plt.figure(0)


# SHARP
# url="http://jsoc.stanford.edu/doc/data/hmi/harp/harp_definitive/2012/03/06/harp.2012.03.06_23:00:00_TAI.png"

#url         = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=hmi.M_720s[2012.03.06_23:29:06_TAI]&op=rs_list&seg=magnetogram&key=T_REC,CROTA2,CDELT1,CDELT2,CRPIX1,CRPIX2,CRVAL1,CRVAL2"
url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=hmi.M_720s[2012.03.06_23:29:06_TAI]&op=rs_list&seg=magnetogram"

#url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=hmi.M_720s[2012.03.06_23:29:06_TAI]&op=rs_list&seg=magnetogram"
# response    = urllib.request.urlopen(url)
# data        = json.loads(response.read())
# filename    = data['segments'][0]['values'][0]
# url         = "http://jsoc.stanford.edu"+filename
# magnetogram = fits.open(url)   # download the data

# Use a fits files that have all keywords
f = '/Users/rattie/Data/SDO/HMI/magnetograms/JSOC_20170321_666/hmi.m_45s.20160401_001500_TAI.2.magnetogram.fits'
map = sunpy.map.Map(f)              # peek at the data

angle = 180 * u.deg
map2 = map.rotate(angle=angle)

print("done")