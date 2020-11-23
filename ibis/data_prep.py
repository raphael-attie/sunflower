import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import fitstools
import balltracking.balltrack as blt
import filters
from scipy.signal import convolve2d
from skimage.exposure import rescale_intensity
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale
from scipy import ndimage
import numpy as np
import fitsio
import matplotlib.pyplot as plt

from pathlib import PurePath, Path
import glob
import multiprocessing
import gc
import time

def shift_upscale(i):
    print(i)
    upfactor = 10.0
    dx = shifts1[i]
    dy = shifts0[i]
    # samplef = fitstools.fitsread(str(ibisf2), tslice=i+1)
    samplef = ndimage.zoom(ibisdata[i+1], upfactor)
    # samplef = filters.translate_by_phase_shift(samplef, dx * upfactor, dy * upfactor)
    samplef = ndimage.shift(samplef, (dy * upfactor, dx * upfactor))
    # downsample
    samplef = rescale(samplef, 1 / upfactor, anti_aliasing=True)
    return samplef


if __name__ == '__main__':

    ibisdir = Path(os.environ['DATA'], 'Ben', 'IBIS', 'white_light')
    ibisf = ibisdir.joinpath('ibis.wl.speckle.hmi_destr.seq1.sonicfilter.fits')
    time_trim = 4
    trim = 40
    ibisdata = fitsio.read(str(ibisf))[time_trim:-time_trim]
    radius = 10
    fseries = np.array([blt.filter_image(im[trim:-trim, trim:-trim], pixel_radius=radius) for im in ibisdata])
    ibisf2 = ibisdir.joinpath('ibis.wl.speckle.refiltered.fits')
    fitsio.write(ibisf2, fseries)

    shifts0 = []
    shifts1 = []
    for i in range(len(fseries) - 1):
        shift, error, diffphase = phase_cross_correlation(fseries[i], fseries[i + 1], upsample_factor=100)
        shifts0.append(shift[0])
        shifts1.append(shift[1])

    with multiprocessing.Pool(processes=20) as pool:
        shifted_images = pool.map(shift_upscale, range(len(ibisdata)-1))

    shifted_images.insert(0, ibisdata[0])
    shifted_images_np = np.array(shifted_images)[:, trim:-trim, trim:-trim]

    fitsio.write(ibisdir.joinpath('ibis.wl.speckle.jitter_corrected.fits'), shifted_images_np)

