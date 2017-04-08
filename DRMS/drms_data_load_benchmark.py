"""
Benchmark of loading time of drms records from remote location at JSOC using urls
"""

import time
import urllib
import drms
from astropy.io import fits
import os

def r_qurls(drms_client, query):
    jsoc_url = 'http://jsoc.stanford.edu'
    segment = 'None'
    if query.lower()[4] == 'm':
        segment = 'magnetogram'
    elif query.lower()[4:6] == 'ic':
        segment = 'continuum'

    jsoc_file_path = drms_client.query(query, seg=segment)
    qurls = jsoc_url + jsoc_file_path[segment]
    return qurls


c = drms.Client(email='attie.raphael@gmail.com', verbose=True)

## Measure time to load data from remote location (JSOC)

# Direct download, manually constructing the URL
# urls = r_qurls(c, 'hmi.M_45s[2016.04.01_TAI/1d@900s]')
#
# start_time = time.time()
# ii = 0
# for url in urls:
#     ii += 1
#     urllib.request.urlretrieve(url, "/Users/rattie/SDO/HMI/magnetograms/"+ "magnetogram_" + '{:05d}'.format(ii) + ".fits")
#     #hdu = fits.open(url)
#
# elapsed_time = time.time() - start_time
# print('done')
# print('elapsed time (s):')
# print(elapsed_time)

# With the export request as-is, it is the same as above, but we do not need to build the URL.
r = c.export('hmi.M_45s[2016.04.01_TAI/1d@900s]{magnetogram}')
r.urls.url
hdu = fits.open(r.urls.url[0])

"""
0     http://jsoc.stanford.edu/SUM68/D803708322/S000...
1     http://jsoc.stanford.edu/SUM68/D803708322/S000...
2     http://jsoc.stanford.edu/SUM51/D803711882/S000...
3     http://jsoc.stanford.edu/SUM57/D803715500/S000...
4     http://jsoc.stanford.edu/SUM49/D803708349/S000...
5     http://jsoc.stanford.edu/SUM49/D803708349/S000...
6     http://jsoc.stanford.edu/SUM50/D803711846/S000...
...
90    http://jsoc.stanford.edu/SUM67/D803746271/S000...
91    http://jsoc.stanford.edu/SUM62/D803753457/S000...
92    http://jsoc.stanford.edu/SUM70/D803743094/S000...
93    http://jsoc.stanford.edu/SUM70/D803743094/S000...
94    http://jsoc.stanford.edu/SUM53/D803746569/S000...
95    http://jsoc.stanford.edu/SUM68/D803753620/S000...
"""

# Use url-tar export method
out_dir = "/Users/rattie/Data/SDO/HMI/magnetograms/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

c = drms.Client(email='raphael.attie@nasa.gov', verbose=True)

ds='hmi.M_45s[2010.11.27_TAI/5d@3h]{magnetogram}&sizeratio=0.015625&process=n=0|im_patch,t_start=2010.11.27_00:00:00_TAI,' \
   't_stop=2010.12.01_21:00:00_TAI,t=0,r=0,c=0,cadence=3h,locunits=carrlong,boxunits=pixels,t_ref=2010.11.27_00:00:00_TAI,' \
   'car_rot=2104,x=328.5,y=12.6,width=512,height=512'
r = c.export('hmi.M_45s[2016.04.01_TAI/1h@900s]{magnetogram}', method='url-tar', protocol='fits')
r.wait()

r_tar.request_url

start_time2 = time.time()

r_tar.download(out_dir)

elapsed_time2 = time.time() - start_time2

print('done')
print('export method url-tar, elapsed time (s):')
print(elapsed_time2)


