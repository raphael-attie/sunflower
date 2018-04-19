import os
from importlib import reload
import matplotlib
matplotlib.use('macosx')
import numpy as np
import balltracking.balltrack as blt
import fitstools
from datetime import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt

datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/python_balltracking'
# Go from 1st frame at Sep 1st 00:00:00 till ~ Sep 3 18:00:00
nframes = int((3600*24*2 + 18*3600)/45) # 5280 frames
#nframes = int(80 * 5)

# Load the series
image = fitstools.fitsread(datafile, tslice=0).astype(np.float32)
dims = image.shape
### Ball parameters

# Ball radius
rs = 2
# depth factor
dp = 0.2
# Multiplier to the standard deviation.
sigma_factor = 2

# Get a BT instance with the above parameters
bt_tf = blt.BT(dims, nframes, rs, dp, sigma_factor=sigma_factor, mode='top', direction='forward', datafiles=datafile)
bt_tb = blt.BT(dims, nframes, rs, dp, sigma_factor=sigma_factor, mode='top', direction='backward', datafiles=datafile)
bt_bf = blt.BT(dims, nframes, rs, dp, sigma_factor=sigma_factor, mode='bottom', direction='forward', datafiles=datafile)
bt_bb = blt.BT(dims, nframes, rs, dp, sigma_factor=sigma_factor, mode='bottom', direction='backward', datafiles=datafile)



if __name__ == '__main__':

    startTime = datetime.now()

    # Parallel pools does not work because of unable to pickle the bt objects. Try using them as global resources.
    # See https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647
    # with Pool(processes=4) as pool:
    #     bt_tf, bt_tb, bt_bf, bt_bb = pool.map(blt.track_all_frames, [bt_tf, bt_tb, bt_bf, bt_bb])
    bt_tf, bt_tb, bt_bf, bt_bb = list(map(blt.track_all_frames, [bt_tf, bt_tb, bt_bf, bt_bb]))

    ballpos_top = np.concatenate((bt_tf.ballpos, bt_tb.ballpos), axis=1)
    ballpos_bottom = np.concatenate((bt_bf.ballpos, bt_bb.ballpos), axis=1)

    print(" Time elapsed: %s " % (datetime.now() - startTime))

    np.save(os.path.join(outputdir,'ballpos_top.npy'), ballpos_top)
    np.save(os.path.join(outputdir, 'ballpos_bottom.npy'), ballpos_bottom)
