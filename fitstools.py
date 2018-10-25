from importlib import reload
import os
import numpy as np
import fitsio
from astropy.io import fits


def fitsread(file, xslice=slice(None), yslice=slice(None), tslice=slice(None)):

    if isinstance(file, str):
        with fitsio.FITS(file) as fitsfile:
            if fitsfile[0].has_data():
                data = np.squeeze( np.moveaxis(fitsfile[0][tslice, yslice, xslice], 0, 2) )
            else:
                data = np.squeeze( np.moveaxis(fitsfile[1][tslice, yslice, xslice], 0, 2))
    else: # Assume and read list of files
        # Load sample to get dimensions
        sample = fitsio.read(file[0])
        # Initialize list of empty arrays of expected size, as many as we have files.
        data = np.empty([*sample.shape, len(file)], np.float32)
        for i, datafile in enumerate(file):
              data[:,:,i] = fitsio.read(datafile)

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
        data = fitsread(file, tslice=slice(i,i+1))
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