import os
import numpy as np
from astropy.io import fits
from astropy.io.fits import getdata
import importlib
if importlib.util.find_spec("fitstio") is not None:
    import fitsio


def fitsread(files, xslice=slice(None), yslice=slice(None), tslice=slice(None), cube=True, header=False):

    if isinstance(files, str):
        if cube:
            # for now, in Windows, with python fitsio package not compiling, this won't work.
            with fitsio.FITS(files) as fitsfile:
                if fitsfile[0].has_data():
                    data = np.squeeze(fitsfile[0][tslice, yslice, xslice])
                else:
                    data = np.squeeze(fitsfile[1][tslice, yslice, xslice])
        else:
            # Load as single file and single image
            if header:
                data, hdr = getdata(files, header=True)
                return data, hdr
            else:
                data = getdata(files)
            
    else:
        if isinstance(tslice, int):
            # There's only 1 file to read.
            data = getdata(files[tslice])
        else: # Assume and read list of files
            # Load sample to get dimensions
            fitsfiles = files[tslice]
            data = np.array([getdata(f) for f in fitsfiles])

    return data


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
