import glob
import os
import time
import drms
from astropy.io import fits
from sunpy.io import read_file

# c = drms.Client(email='attie.raphael@gmail.com', verbose=True)
# r = c.export('hmi.M_45s[2016.04.01_TAI/1d@900s]{magnetogram}')
# hdu0 = fits.open(r.urls.url[0])
# data0 = hdu0[1].data
#
# drms_export = c.export('hmi.M_45s[2016.04.01_TAI/1h@900s]{magnetogram}', method='url', protocol='fits')
# drms_export.wait()
# urls    = drms_export.urls.url
# hdu1    = fits.open(urls[0])
# hdu1.verify('silentfix')
# data1   = hdu1[1].data


data_dir    = '/Users/rattie/Data/SDO/HMI/magnetograms/JSOC_20170321_666'
file_list   = glob.glob(os.path.join(data_dir, '*.fits'))

stime1 = time.time()
for file in file_list:
    hdu1 = fits.open(file)
elapsed_time1 = time.time() - stime1

stime2 = time.time()
for file in file_list:
    hdu2 = fits.open(file)
    hdu2.verify('silentfix')
elapsed_time2 = time.time() - stime2

stime3 = time.time()
for file in file_list:
    hdu3 = read_file(file)
elapsed_time3 = time.time() - stime3

print("elapsed_time1 (astropy) = %s" %elapsed_time1)
print("elapsed_time2 (astropy fixed) = %s" %elapsed_time2)
print("elapsed_time3 (sunpy) = %s" %elapsed_time3)