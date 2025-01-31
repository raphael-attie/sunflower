import numpy as np
from astropy.io import fits
from astropy.io.fits import getdata
# The fitsio python package installation often does not work in Windows or in linux with latest Python version
import importlib.util
fitsio_found = importlib.util.find_spec("fitsio")
if fitsio_found is not None:
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
            # Read at given index
            data = getdata(files[tslice])
        else: # Read list of files
            data = np.array([getdata(f) for f in files[tslice]])

    return data


def writefits(image, fname, header=None, compressed=False):

    if not compressed:
        try:
            fits.writeto(fname, image, header=header, output_verify='silentfix', overwrite=True)
        except TypeError:
            fits.writeto(fname, image, header=header, output_verify='silentfix', checksum=True, overwrite=True)
    else:
        chdu = fits.CompImageHDU(data=image, compression_type='RICE_1')
        chdu.writeto(fname, overwrite=True)

    return
