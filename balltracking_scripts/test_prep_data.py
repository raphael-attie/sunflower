from importlib import reload
import numpy as np
import numpy.ma as ma
import balltracking.balltrack as blt
import fitstools
import fitsio
from timeit import timeit

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


file = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/series_continuum/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum_00000.fits'
# Get the 1st image
image   = fitsio.read(file).astype(np.float32)

### Ball parameters
# Nb of intermediate steps
nt = 15
# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2
# Get a BT instance with the above parameters
bt = blt.BT(image.shape, nt, rs, dp, sigma_factor=sigma_factor)
bt.initialize(image)

mywrap = wrapper(blt.prep_data, image, bt.mean, bt.sigma)
mywrap2 = wrapper(blt.prep_data2, image)

print(timeit(mywrap, number = 100))
print(timeit(mywrap2, number = 100))