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

nt = 10
rs = 2
dp = 0.2
bt = blt.BT(image.shape, nt, rs, dp)
# Initialize ball positions with height
bt.initialize_ballpos(imagef)

# x = np.arange(1000, dtype=np.float32)
# y = np.arange(1000, dtype=np.float32)

xstart = np.full([16129], 20.3, dtype=np.float32)
ystart = np.full([16129], 30.1, dtype=np.float32)
zstart = blt.put_balls_on_surface(imagef, xstart, ystart, rs, dp)

pos, vel = blt.initialize_ball_vector(xstart, ystart, zstart)

xcoarse, ycoarse = blt.replace_bad_balls1(pos, bt)


mywrap1 = wrapper(blt.replace_bad_balls1, pos, bt)
mywrap2 = wrapper(blt.replace_bad_balls1, pos, bt)

s1 = timeit(mywrap1, number=10000)
s2 = timeit(mywrap2, number=10000)

print(s1)
print(s2)


# Testing the unique extraction based on another array

pos     = np.array([3, 3, 7, 7, 9, 9, 9, 10, 10])
weights = np.array([2, 10, 20, 8, 5, 7, 15, 7, 2])
# Get the number of occurences of the elements in pos but throw away the unique array, it's not the one I want.
_, ucounts = np.unique(pos, return_counts=True)
# Initialize the output array.
unique_pos_idx = np.zeros([ucounts.size], dtype=np.uint32)

last = 0
for i in range(ucounts.size):
    maxpos = np.argmax( weights[last:last+ucounts[i]] )
    unique_pos_idx[i] = last + maxpos
    last += ucounts[i]

#
sidx = np.lexsort([weights,pos])
# Get indices where the position array changes. This maps to the last argmax of the lexsort result
# So we use it to extract an array of unique positions sorted by the weights (balls age).
pos_changes = np.flatnonzero(pos[1:] != pos[:-1])