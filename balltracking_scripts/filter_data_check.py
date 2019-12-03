import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import numpy as np
import balltracking.balltrack as blt
import os
import fitstools

# Path to fits file (fits cube)
file = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/mtrack_20100901_120034_TAI_20100902_120034_TAI_LambertCylindrical_continuum.fits'
datadir = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/python_balltracking/'
ballpos_top = np.load(os.path.join(datadir,'ballpos_top.npy'))
#surface = np.zeros([30, 60])
nt = 40
rs = 2
dp = 0.2
ballspacing = 4 * rs
sigma_factor = 1

means1, means2 = [], []
sigmas1, sigmas2 = [], []
surfaces = []

for n in range(nt):
    image = fitstools.fitsread(file, tslice=n).astype(np.float32)
    surface, mean, sigma = blt.prep_data2(image, sigma_factor=sigma_factor)
    surfaces.append(surface)
    means1.append(surface.mean())
    sigmas1.append(surface.std())
    means2.append(surface[0:100, 0:100].mean())
    sigmas2.append(surface[0:100, 0:100].std())


fig = plt.figure(0, figsize=(10, 10))
plt.imshow(surfaces[0], origin='lower', cmap='gray', vmin=-4, vmax=4, interpolation='lanczos')
plt.show()


plt.close('all')