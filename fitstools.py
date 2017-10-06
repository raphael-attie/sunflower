import os
import numpy as np
import fitsio
from astropy.io import fits


def fitsread(file, n):

    with fitsio.FITS(file) as fitsfile:
        # NO. 0 is also valid for indexing into the slices.
        if fitsfile[0].has_data():
            data = np.squeeze(fitsfile[0][n, :, :])
        else:
            data = np.squeeze(fitsfile[1][n, :, :])
    return data

def fitsheader(file):

    with fitsio.FITS(file) as fitsfile:
        if fitsfile[0].has_data():
            header = fitsfile[0].read_header()
        else:
            header = fitsfile[1].read_header()

    return header


def write_cube_to_series(file, directory, compressed=False):

    h = fitsheader(file)
    nfiles = h['ZNAXIS3']

    for i in range(nfiles):
        data = fitsread(file, n=i)
        basename = '%s_%05d.fits' %(os.path.basename(os.path.splitext(file)[0]), i)
        fname = os.path.join(directory, basename)
        writefits(data, fname, compressed=compressed)

    return

def writefits(image, fname, compressed=False):

    if not compressed:
        try:
            fits.writeto(fname, image, output_verify='exception', overwrite=True)
        except TypeError:
            fits.writeto(fname, image, output_verify='exception', checksum=True, clobber=True)
    else:
        chdu = fits.CompImageHDU(data=image, compression_type='RICE_1')
        chdu.writeto(fname, overwrite=True)

    return