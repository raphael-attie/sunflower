import numpy as np
from astropy.io import fits
from astropy.io.fits import getdata


from pathlib import Path

def fitsread(files, xslice=slice(None), yslice=slice(None), tslice=slice(None), cube=True, header=False):

    if isinstance(files, (str, Path)):
        if cube:
            with fits.open(files, memmap=True) as hdul:
                if hdul[0].data is not None:
                    data = np.squeeze(hdul[0].data[tslice, yslice, xslice]).copy()
                else:
                    data = np.squeeze(hdul[1].data[tslice, yslice, xslice]).copy()
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
