import glob, os
import matplotlib
matplotlib.use('agg')
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import fitstools
import fitsio
### https://matplotlib.org/gallery/images_contours_and_fields/image_transparency_blend.html

def load_rgb_data(index, tstep, tavg):

    mag_slice=slice(int((index+1)*tstep -tavg/2), int((index+1)*tstep+tavg/2))
    mag = fitstools.fitsread(mag_file, tslice=mag_slice)
    lanes = fitsio.read(lanesf[index])

    lanes_n = Normalize(0, 8, clip=True)(lanes)

    mag_pos = mag.copy()
    mag_pos[mag_pos < 0] = 0
    mag_neg = mag.copy()
    mag_neg[mag_neg > 0] = 0
    mag_neg = np.abs(mag_neg)
    mag_pos = np.mean(mag_pos, axis=2)
    mag_neg = np.mean(mag_neg, axis=2)

    mag_pos_n = Normalize(0, 400, clip=True)(mag_pos)
    mag_neg_n = Normalize(0, 400, clip=True)(mag_neg)

    # Create the RGB image
    rgb = np.moveaxis(np.array([mag_pos_n, mag_neg_n, lanes_n]), 0, -1)
    return rgb


lanes_dir = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/python_balltracking/'
lanesf = sorted(glob.glob(os.path.join(lanes_dir, 'lanes_fwhm15_tavg80_nsteps20_*.fits')))
mag_file = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/mtrack_20100901_120034_TAI_20100902_120034_TAI_LambertCylindrical_magnetogram.fits'

tavg = 80
tstep = 40
nsteps = 20

for i in range(len(lanesf)):
#for i in range(3):
    rgb = load_rgb_data(i, tstep, tavg)
    plt.figure(0, figsize=(11,11))
    plt.imshow(rgb, origin='lower')
    plt.title('Frame {:d} [Lanes at 80-min average, sliding at 30-min cadence]'.format(i))
    plt.tight_layout()
    fname = os.path.join(lanes_dir, 'matlab_lanes_rgb_fwhm15_tavg{:d}_nsteps{:d}_{:03d}.png'.format(tavg, nsteps, i))
    plt.savefig(fname)

