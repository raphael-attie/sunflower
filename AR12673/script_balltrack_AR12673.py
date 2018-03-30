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
# Smoothing for Euler maps
fwhm = 15
# Calibration factors
a_top = 1.41
a_bottom = 1.30

### time windows for the euler map
tspan = 80
tstep = 40
tcenters = np.arange(0, nframes-tstep, tstep)
tranges = [[tcenters[i], tcenters[i]+ tspan] for i in range(tcenters.size)]

### Lanes parameters
nsteps = 30
maxstep = 4

# Get a BT instance with the above parameters
bt_tf = blt.BT(dims, nframes, rs, dp, sigma_factor=sigma_factor, mode='top', direction='forward', datafiles=datafile)
bt_tb = blt.BT(dims, nframes, rs, dp, sigma_factor=sigma_factor, mode='top', direction='backward', datafiles=datafile)
bt_bf = blt.BT(dims, nframes, rs, dp, sigma_factor=sigma_factor, mode='bottom', direction='forward', datafiles=datafile)
bt_bb = blt.BT(dims, nframes, rs, dp, sigma_factor=sigma_factor, mode='bottom', direction='backward', datafiles=datafile)



if __name__ == '__main__':

    startTime = datetime.now()

    # with Pool(processes=4) as pool:
    #     bt_tf, bt_tb, bt_bf, bt_bb = pool.map(blt.track_all_frames, [bt_tf, bt_tb, bt_bf, bt_bb])
    bt_tf, bt_tb, bt_bf, bt_bb = list(map(blt.track_all_frames, [bt_tf, bt_tb, bt_bf, bt_bb]))

    ballpos_top = np.concatenate((bt_tf.ballpos, bt_tb.ballpos), axis=1)
    ballpos_bottom = np.concatenate((bt_bf.ballpos, bt_bb.ballpos), axis=1)

    print(" Time elapsed: %s " % (datetime.now() - startTime))

    np.save(os.path.join(outputdir,'ballpos_top.npy'), ballpos_top)
    np.save(os.path.join(outputdir, 'ballpos_bottom.npy'), ballpos_top)

    startTime = datetime.now()

    for i in range(len(tranges)):

        vx_top, vy_top, wplane_top = blt.make_velocity_from_tracks(ballpos_top, dims, tranges[i], fwhm)
        vx_bottom, vy_bottom, wplane_top = blt.make_velocity_from_tracks(ballpos_bottom, dims, tranges[i], fwhm)

        vx_top *= a_top
        vy_top *= a_top
        vx_bottom *= a_bottom
        vy_bottom *= a_bottom

        vx = 0.5*(vx_top + vx_bottom)
        vy = 0.5*(vy_top + vy_bottom)

        # Write fits file
        fitstools.writefits(vx, os.path.join(outputdir, 'vx_%03d.fits'%i))
        fitstools.writefits(vy, os.path.join(outputdir, 'vy_%03d.fits'%i))

        lanes_top = blt.make_lanes(vx_top, vy_top, nsteps, maxstep)
        lanes_bottom = blt.make_lanes(vx_bottom, vy_bottom, nsteps, maxstep)
        lanes = blt.make_lanes(vx, vy, nsteps, maxstep)

        fitstools.writefits(lanes_top, os.path.join(outputdir, 'lanes_top_%03d.fits'%i))
        fitstools.writefits(lanes_bottom, os.path.join(outputdir, 'lanes_bottom_%03d.fits' % i))
        fitstools.writefits(lanes, os.path.join(outputdir, 'lanes_%03d.fits' % i))

    total_time = datetime.now() - startTime
    avg_time = total_time.total_seconds()/len(tranges)
    print("avg_time: %0.1f seconds" %avg_time)




