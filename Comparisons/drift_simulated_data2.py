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

def shift_series(images, outputdir):
    # Velocity offset. Will be applied to both x and y axis
    nframes = 80
    dv = 0.04
    vx_rates = np.arange(-0.2, 0.21, dv)
    vx_rates[int(len(vx_rates)/2)] = 0
    for i, rate in enumerate(vx_rates):
        outputdir_dt = os.path.join(outputdir, 'drift_{:02d}'.format(i))
        if not os.path.exists(outputdir_dt):
            os.makedirs(outputdir_dt)
        for n in range(nframes):
            image = images[...,n]
            shifted_image = filters.translate_by_phase_shift(image, n*rate, 0)
            filename = os.path.join(outputdir_dt, 'im_shifted_{:02d}.fits'.format(n))
            fitsio.write(filename, shifted_image)


# Create series of translated simulated HMI-like continuum images
outputdir = '/Users/rattie/Data/sanity_check/simulation_series/'
files = sorted(glob.glob('/Users/rattie/Data/Ben/SteinSDO/SDO_int*.fits'))
images_sim = np.moveaxis(np.array([fitstools.fitsread(f, cube=False) for f in files]), 0, -1)
shift_series(images_sim, outputdir)

# Create series of translated real HMI continuum images
filepath = '/Users/rattie/Data/SDO/HMI/continuum/Lat_0/mtrack_20110627_200034_TAI_20110628_000033_TAI_Postel_060.4_00.0_continuum.fits'
outputdir = '/Users/rattie/Data/sanity_check/hmi_series/'
images_hmi = fitstools.fitsread(filepath)
shift_series(images_hmi, outputdir)

#TODO: check for discrepancies between first cropping by integer amount and shifting by remaining fractional amount
