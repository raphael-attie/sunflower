"""
Create series of shifted images for sanity-checking FLCT

Translate an image by different decimal amount from ~ 0.1 to 2 px. The different image has an increasing time-gap with
respect to the reference image (from 1 to 9 frames). There is one series of space-shifted images per value of time gap.
"""

import os, glob
import numpy as np
import filters
import fitsio
import fitstools
from pathlib import Path

DATADIR = os.environ['DATA']

def shift_series(images, outputdir):
    # Velocity offset. Will be applied to both x and y axis
    dv = 0.04
    vx_rates = np.arange(-0.2, 0.21, dv)
    vx_rates[int(len(vx_rates)/2)] = 0
    for i, rate in enumerate(vx_rates):
        outputdir_dt = os.path.join(outputdir, 'drift_{:02d}'.format(i))
        if not os.path.exists(outputdir_dt):
            os.makedirs(outputdir_dt)
        for n, image in enumerate(images):
            shifted_image = filters.translate_by_phase_shift(image, n*rate, 0)
            filename = os.path.join(outputdir_dt, 'im_shifted_{:04d}.fits'.format(n))
            fitsio.write(filename, shifted_image)


# Create series of translated simulated HMI-like continuum images
# outputdir = os.path.join(DATADIR, 'sanity_check/stein_series/')
# files = sorted(glob.glob(os.path.join(DATADIR, 'Ben/SteinSDO/SDO_int*.fits')))
# images_sim = np.array([fitstools.fitsread(f, cube=False) for f in files])
# shift_series(images_sim, outputdir)

# Create series of translated real HMI continuum images
filepath = os.path.join(DATADIR, 'SDO/HMI/polar_study/mtrack_20110627_200034_TAI_20110628_000033_TAI_Postel_060.4_00.0_continuum.fits')
outputdir = os.path.join(DATADIR, 'sanity_check/hmi_series/')
images_hmi = fitstools.fitsread(filepath)
images_hmi = np.moveaxis(images_hmi, -1, 0)
shift_series(images_hmi, outputdir)

#TODO: check for discrepancies between first cropping by integer amount and shifting by remaining fractional amount
