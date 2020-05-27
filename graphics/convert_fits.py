import cv2
import os, glob
from pathlib import PurePath
import fitsio
import numpy as np
from skimage.exposure import rescale_intensity
# skimage.exposure.rescale_intensity(image, in_range='image', out_range='dtype')

outputdir = PurePath(os.environ['DATA'], 'Ben/SteinSDO/converted_intensity_images')
fitsfiles = sorted(glob.glob(PurePath(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits').as_posix()))

for f in fitsfiles:
    fp = PurePath(f)
    data = fitsio.read(f)
    #data2 = np.pad(data, (0, 1), mode='constant')
    data2 = np.uint8(rescale_intensity(data, out_range=(0,255)))
    jpegfile = PurePath(outputdir, fp.stem + '.jpg').as_posix()
    cv2.imwrite(jpegfile, data2)

