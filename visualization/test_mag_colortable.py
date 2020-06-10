import csv
import fitstools
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import numpy as np

lut_file = '/Users/rattie/Dev/sdo_tracking_framework/graphics/HMI.MagColor.IDL_256.lut.txt'
lut_reader = csv.reader( open(lut_file, newline=''), delimiter=' ')
lut_str = [list(filter(None, row)) for row in lut_reader]
lut = [[float(value) for value in row] for row in lut_str]
cmap = matplotlib.colors.ListedColormap(lut)

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
tslice = 0
data = fitstools.fitsread(datafile, tslice=tslice).astype(np.float32)
datamax = np.max(np.abs(data))

fig, axs = plt.subplots(1,3,figsize=(18,10))
axs[0].imshow(data, vmin=-datamax, vmax=datamax, cmap=cmap, origin='lower')
axs[1].imshow(data, vmin=-datamax, vmax=datamax, cmap='RdBu', origin='lower')
axs[2].imshow(data, vmin=-datamax, vmax=datamax, cmap='coolwarm', origin='lower')
plt.tight_layout()
