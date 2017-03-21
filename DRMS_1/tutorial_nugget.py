"""
Experiment on the drms module nugget at http://hmi.stanford.edu/hminuggets/?p=1757
Testing some custom function that parses the drms query string to write more intuitive queries.

"""
import time
import drms
from astropy.io import fits
import matplotlib.pyplot as plt


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


c = drms.Client()

# Use a regular expression to filter the result.
# Here, list HMI series that start with the string "Ic_"
c.series('hmi.Ic_')
# output gives series name, e.g:
"""
['hmi.Ic_45s',
 'hmi.Ic_720s',
 'hmi.Ic_noLimbDark_720s',
 'hmi.ic_nolimbdark_720s_nrt']
"""

# Get the primekeys associated with a given series name
c.pkeys('hmi.Ic_45s')
"""
['T_REC', 'CAMERA']
"""
# Can also use "keys()" method for more regular "fits-like" keywords
c.keys('hmi.Ic_45s')
"""
['cparms_sg000',
 'continuum_bzero',
 'continuum_bscale',
 'DATE',
 'DATE__OBS',
 'TELESCOP',
 'INSTRUME',
 'WAVELNTH',
 'CAMERA',
 'BUNIT',
  ...
"""

# Use a given series name to make a query, followed by square-bracketed primekeys.
k = c.query('hmi.Ic_45s[2016.04.01_TAI/1d@6h]', key='T_REC')
"""
                     T_REC
0  2016.04.01_00:00:00_TAI
1  2016.04.01_06:00:00_TAI
2  2016.04.01_12:00:00_TAI
3  2016.04.01_18:00:00_TAI
"""

# More detailed information with "info()" method.
# Example with intensity continuum (Ic)
ic_info = c.info('hmi.Ic_45s')
# Get the segment names out of that, which is needed to get the path of the FITS files
ic_info.segments
"""
            type units protocol       dims                 note
name
continuum  short  DN/s     fits  4096x4096  continuum intensity

"""

# With the segment name, we finally can query the path to the records (i.e, the images)

# Example with continuum images
q_ic = c.query('hmi.Ic_45s[2016.04.01_TAI/1d@6h]', seg='continuum')
"""
                                 continuum
0  /SUM30/D803708325/S00008/continuum.fits
1  /SUM41/D803708365/S00008/continuum.fits
2  /SUM52/D803720863/S00008/continuum.fits
3  /SUM32/D803730123/S00008/continuum.fits

"""

# Example with magnetograms
m_info = c.info('hmi.M_45s')
m_info.segments
"""
            type  units protocol       dims         note
name
magnetogram  int  Gauss     fits  4096x4096  magnetogram

"""

# Another way to query is with 2 output: keys and segment path
key_m, path_m = c.query('hmi.M_45s[2016.04.01_TAI/1d@6h]', key='T_REC', seg='magnetogram')

key_m
"""
                     T_REC
0  2016.04.01_00:00:00_TAI
1  2016.04.01_06:00:00_TAI
2  2016.04.01_12:00:00_TAI
3  2016.04.01_18:00:00_TAI
"""

path_m
"""
                                 magnetogram
0  /SUM68/D803708322/S00008/magnetogram.fits
1  /SUM60/D803708362/S00008/magnetogram.fits
2  /SUM32/D803720860/S00008/magnetogram.fits
3  /SUM43/D803730120/S00008/magnetogram.fits

"""
m_url = 'http://jsoc.stanford.edu' + path_m.magnetogram[0]

# Yet the query string hosts is enough to get the segment name for HMI continuum and magnetogram
# So by parsing it within the custom function "r_qurls", we do it in one line:
urls = r_qurls(c, 'hmi.M_45s[2016.04.01_TAI/1d@6h]')

# Test loading some data
hdu = fits.open(urls[0])
image = hdu[1].data
urls = r_qurls(c, 'hmi.M_45s[2016.04.01_TAI/1d@900s]')
# Measure time to load data from remote location (JSOC)
start_time = time.time()

for url in urls:
    hdu = fits.open(url)

elapsed_time = time.time() - start_time
print('done')
print('elapsed time (s):')
print(elapsed_time)


# # Plot a continuum image and a magnetogram
#
# fig = plt.figure(1)
# fig.clear()
# # Plot image
# plt.imshow(image, vmin=-100, vmax=100, cmap='gray', origin='lower')
#
# ax = plt.gca()
# plt.xlabel('X [px]')
# plt.ylabel('Y [px]')
