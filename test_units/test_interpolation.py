from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import fitsio
import fitstools
import cython_modules.interp as cinterp
import balltracking.balltrack as blt
from timeit import timeit

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

file = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/series_continuum/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum_00000.fits'
#file = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/series_continuum_calibration/calibration/drift0.20/drift_0001.fits'
# Get the header
h       = fitstools.fitsheader(file)
# Get the 1st image
image   = fitsio.read(file)
imagef  = image.astype(np.float32)
imaged  = image.astype(np.double)


###### 1-element tests #########
xd = np.array([10.3], dtype=np.double)
yd = np.array([10.1], dtype=np.double)
xf = np.array([10.3], dtype=np.float32)
yf = np.array([10.1], dtype=np.float32)
zf = np.array([0], dtype=np.float32)
# Python
testf  = blt.bilin_interp_f(imagef, xf, yf)
# Cython
testcf = cinterp.bilin_interp1f(imagef, xf, yf)
# Cython C
zf = cinterp.cbilin_interp1(imagef, xf, yf)


##### multi-balls test #########
nballs = 16129
npts = 25

#dims = [nballs, npts]
dims = [npts, nballs]

xf = np.full(dims, 10.3, dtype=np.float32)
yf = np.full(dims, 10.1, dtype=np.float32)
zf = np.zeros(dims, dtype=np.float32)
# Python
testf = blt.bilin_interp_f(imagef, xf, yf)
# Cython with 2 dimensions
testcf = cinterp.bilin_interp2f(imagef, xf, yf)
# Cython C with 2 dimensions
zf = cinterp.cbilin_interp2(imagef, xf, yf)

# Wrap and profile the above
mywrap_f = wrapper(blt.bilin_interp_f, imagef, xf, yf)
mywrap_cf = wrapper(cinterp.bilin_interp2f, imagef, xf, yf)
mywrap_cf3 = wrapper(cinterp.bilin_interp3f, imagef, xf, yf)

zf = np.full(dims, 0, dtype=np.float32)
mywrap_cf4 = wrapper(cinterp.cbilin_interp2, imagef, xf, yf)

timeit(mywrap_f, number=100) # 3s
timeit(mywrap_cf, number=100) # 0.25s
timeit(mywrap_cf3, number=100) # 0.37s
timeit(mywrap_cf4, number=100) # 0.24s


