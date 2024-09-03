import numpy as np
from astropy.io import fits
# The fitsio python package installation often does not work in Windows or in linux with latest Python version
import importlib.util
from pathlib import Path

def fitsread(filepaths, xslice=slice(None), yslice=slice(None), tslice=slice(None), cube=True, header=False):

    if isinstance(filepaths, str) or isinstance(filepaths, Path):
        if cube:
            # for now, in Windows, with python fitsio package not compiling, this won't work.
            with fits.open(filepaths) as hdul:
                    data = np.squeeze(hdul[-1].section[tslice, yslice, xslice])
        else:
            # Load as single file and single image
            if header:
                data, hdr = fits.getdata(filepaths, header=True)
                return data, hdr
            else:
                data = fits.getdata(filepaths)

  
            
    elif isinstance(tslice, int):
            # There's only 1 file to read.
            data = fits.getdata(filepaths[tslice])
    else:
        # Assume and read list of files
        subset_files = filepaths[tslice]
        data = np.array([fits.getdata(f) for f in subset_files])

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
