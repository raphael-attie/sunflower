import numpy as np
import balltracking.balltrack as blt
import fitstools
from datetime import datetime
import multiprocessing

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/mtrack_20100901_120034_TAI_20100902_120034_TAI_LambertCylindrical_continuum.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/python_balltracking/'
nframes = 1920 # 5280 frames
#nframes = int(80 * 5)

# Load the series
image = fitstools.fitsread(datafile, tslice=0).astype(np.float32)
### Ball parameters

# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2

if __name__ == '__main__':

    # the multiprocessing start method can only bet set once.
    multiprocessing.set_start_method('spawn')

    # Parallel pools does not work because BT instances created within the children cannot be pickled.
    # The alternative to "hide" the creation of the 4 BT instances is to return pickable objects,e.g numpy arrays ballpos.
    # See https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647

    startTime = datetime.now()
    ballpos_top, ballpos_bottom = blt.balltrack_all(nframes, rs, dp, sigma_factor, outputdir, datafiles=datafile, ncores=4)

    print(" Time elapsed: %s " % (datetime.now() - startTime))