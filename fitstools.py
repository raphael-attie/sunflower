from importlib import reload
import os
import numpy as np
from astropy.io import fits
from astropy.io.fits import getdata
import importlib
# Check fitsio is installed and import if it does
if importlib.util.find_spec("fitsio") is not None:
    import fitsio

def fitsread(files, xslice=slice(None), yslice=slice(None), tslice=slice(None), cube=True, astropy=False, header=False):

    if isinstance(files, str):
        if cube:
            with fitsio.FITS(files) as fitsfile:
                if fitsfile[0].has_data():
                    data = np.squeeze(np.moveaxis(fitsfile[0][tslice, yslice, xslice], 0, 2))
                else:
                    data = np.squeeze(np.moveaxis(fitsfile[1][tslice, yslice, xslice], 0, 2))
        else:
            #Load as single file and single image
            if astropy:
                if header:
                    data, hdr = getdata(files, header=True)
                    return data, hdr
                else:
                    data = getdata(files)
                    
            else:
                data = fitsio.read(files)
            
    else:
        if isinstance(tslice, int):
            # There's only 1 file to read.
            if astropy:
                data = getdata(files[tslice])
            else:
                data = fitsio.read(files[tslice])
        else: # Assume and read list of files
            # Load sample to get dimensions
            fitsfiles = files[tslice]
            sample = fitsio.read(fitsfiles[0])
            data = np.empty([*sample.shape, len(fitsfiles)], np.float32)
            for i, datafile in enumerate(fitsfiles):
                data[:, :, i] = fitsio.read(datafile)
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