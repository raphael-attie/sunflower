import fitstools
import fitsio
from astropy.io import fits
import numpy as np
import os

file = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'

directory = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/series_continuum'

fitsfile = fitsio.FITS(file)
header   = fitsfile[1].read_header()
nslices  = header['ZNAXIS3']

for i in range(nslices):
    image = np.squeeze(fitsfile[1][i, 0:210, 0:210])
    basename = '%s_%05d_C.fits' % (os.path.basename(os.path.splitext(file)[0]), i)
    fname = os.path.join(directory, basename)

    chdu = fits.CompImageHDU(data=image, compression_type='RICE_1')
    chdu.writeto(fname, overwrite=True)

fitsfile.close()

# Extract a smaller FOV for calibbration
directory = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/series_continuum_calibration'

for i in range(nslices):
    image = np.squeeze(fitsfile[1][i, 0:210, 0:210])
    basename = '%s_%05d_C.fits' % (os.path.basename(os.path.splitext(file)[0]), i)
    fname = os.path.join(directory, basename)

    chdu = fits.CompImageHDU(data=image, compression_type='RICE_1')
    chdu.writeto(fname, overwrite=True)


#fitstools.write_cube_to_series(file, directory, True)
