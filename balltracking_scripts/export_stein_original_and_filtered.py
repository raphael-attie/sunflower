import cv2
import os, glob
from pathlib import PurePath
import fitsio
import numpy as np
from skimage.exposure import rescale_intensity
from balltracking.balltrack import blt_prep_data2
# skimage.exposure.rescale_intensity(image, in_range='image', out_range='dtype')

outputdir = PurePath(os.environ['DATA'], 'Ben/SteinSDO/converted_intensity_images')
fitsfiles = sorted(glob.glob(PurePath(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits').as_posix()))

sigma_factor = 2
fourier_radius =2
fov = np.s_[10:-10, 10:-10]


for f in fitsfiles:
    fp = PurePath(f)
    data = fitsio.read(f).astype(np.float32)
    prep_data, _, _ = blt_prep_data2(data, sigma_factor=sigma_factor, pixel_radius=fourier_radius)
    vmin = np.percentile(prep_data[fov], 0.1)
    vmax = np.percentile(prep_data[fov], 99.5)
    #data2 = np.pad(data, (0, 1), mode='constant')
    data2 = np.uint8(rescale_intensity(data, out_range=(0,255)))
    prep_data2 = np.uint8(rescale_intensity(prep_data, in_range=(vmin, vmax), out_range=(0, 255)))
    prep_jpegfile = PurePath(outputdir, fp.stem + '_prep.jpg').as_posix()
    prep_fitsfile = PurePath(outputdir, fp.stem + '_prep.fits').as_posix()
    jpegfile = PurePath(outputdir, fp.stem + '.jpg').as_posix()

    fitsio.write(prep_fitsfile, prep_data.astype(np.float32))
    cv2.imwrite(prep_jpegfile, prep_data2)
    cv2.imwrite(jpegfile, data2)



