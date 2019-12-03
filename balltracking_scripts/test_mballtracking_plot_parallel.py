"""
Test magnetic balltracking with figure printing in parallel
"""
from importlib import reload
import multiprocessing
import time
from functools import partial
import matplotlib
# matplotlib.use('macosx')
matplotlib.use('agg')
import numpy as np
import fitstools
import matplotlib.pyplot as plt
import balltracking.balltrack as blt
import balltracking.mballtrack as mblt
from skimage.feature import peak_local_max
DTYPE = np.float32



def plot_mballtrack(args):

    pos_p, pos_n, i = args
    datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
    data = fitstools.fitsread(datafile, tslice=i).astype(np.int32).astype(DTYPE)
    #surface = mblt.prep_data(data)
    # mask_maxi = data >= mbt.mag_thresh
    # ymaxi, xmaxi = np.array(peak_local_max(data, indices=True, footprint=np.ones((mbt.ballspacing, mbt.ballspacing)), labels=mask_maxi)).T
    # ymaxi3, xmaxi3 = np.array(peak_local_max(data, indices=True, footprint=np.ones((3, 3)), labels=mask_maxi)).T
    #print('Plotting figure %d'%i)

    plt.figure(i, figsize=(13, 10))
    plt.imshow(data, cmap='gray', vmin=-100, vmax=100)
    plt.colorbar()
    # plt.scatter(xmaxi3, ymaxi3, s=12, facecolors='none', edgecolors='blue')
    # plt.scatter(xmaxi, ymaxi, s=12, facecolors='none', edgecolors='green')
    # plt.scatter(mbt_p.ballpos[0, mbt_p.unique_valid_balls, i], mbt_p.ballpos[1, mbt_p.unique_valid_balls, i],
    #             marker='.', s=12, color='red')
    # # plt.text(x + 1, y + 1, '%d' % target, fontsize=8)
    # plt.scatter(mbt_n.ballpos[0, mbt_n.unique_valid_balls, i], mbt_n.ballpos[1, mbt_n.unique_valid_balls, i],
    #             marker='.', s=12, color='cyan')

    maskp = pos_p[0, :] > 0
    maskn = pos_n[0, :] > 0
    plt.scatter(pos_p[0, maskp], pos_p[1, maskp],
                marker='.', s=12, color='red')
    # plt.text(x + 1, y + 1, '%d' % target, fontsize=8)
    plt.scatter(pos_n[0, maskn], pos_n[1, maskn],
                marker='.', s=12, color='cyan')

    #plt.axis([0, data.shape[1], 0, data.shape[0]])
    plt.axis([0, 512, 0, 512])
    plt.xlabel('Lambert cyl. X')
    plt.ylabel('Lambert cyl. Y')
    plt.tight_layout()

    plt.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mballtrack/frame_%02d.png' % i)
    plt.close()

if __name__ == "__main__":
    datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
    data = fitstools.fitsread(datafile, tslice=0).astype(np.int32).astype(DTYPE)
    surface = mblt.prep_data(data)
    mag_thresh=50
    intsteps = 10
    nt = 100

    # Get a BT instance with default parameters
    mbt_p = mblt.MBT(nt=nt, rs=2, am=0.5, dp=0.3, td=1,
                     intsteps=intsteps, mag_thresh=mag_thresh, polarity=1, track_emergence=True, datafiles=datafile)
    #mbt.track_all_frames_debug()
    mbt_p.track_all_frames()

    mbt_n = mblt.MBT(nt=nt, rs=2, am=0.5, dp=0.3, td=1,
                     intsteps=intsteps, mag_thresh=mag_thresh, polarity=-1, track_emergence=True, datafiles=datafile)
    #mbt.track_all_frames_debug()
    mbt_n.track_all_frames()

    list_pos_p = [mbt_p.ballpos[...,i] for i in range(nt)]
    list_pos_n = [mbt_n.ballpos[...,i] for i in range(nt)]
    inputs = zip(list_pos_p, list_pos_n, range(nt))

    ## Plot parallel
    pool = multiprocessing.Pool(processes=4)

    pool.map(plot_mballtrack, inputs)
    #Parallel(n_jobs=4)(delayed(plot_mballtrack)(mbt_p, mbt_n, i) for i in range(nt))


