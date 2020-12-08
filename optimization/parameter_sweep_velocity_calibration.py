import os, glob, sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
import fitsio
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter


def load_vel_mean(v_files, trange):
    "Load the velocity files and average over a time range"
    vxs = []
    vys = []
    vx_files_subset = v_files[0][trange[0]:trange[1]]
    vy_files_subset = v_files[1][trange[0]:trange[1]]
    for vxf, vyf in zip(vx_files_subset, vy_files_subset):
        vxs.append(fitsio.read(vxf))
        vys.append(fitsio.read(vyf))
    # Get the mean of the velocity components
    vx = np.array(vxs).mean(axis=0)
    vy = np.array(vys).mean(axis=0)
    return vx, vy


def smooth_vel(vxs, vys, fwhm, kernel='boxcar'):
    """ Smooth the velocity with a smoothing kernel that can either be:
     - boxcar: width set to fwhm
     - gaussian: parametrized by fwhm.

     Returns the smoothed velocity components
    """

    if kernel == 'boxcar':
        box = np.ones([fwhm, fwhm]) / fwhm**2
        vxs2 = convolve2d(vxs, box, mode='same')
        vys2 = convolve2d(vys, box, mode='same')
    elif kernel == 'gaussian':
        sigma = fwhm / 2.35
        vxs2 = gaussian_filter(vxs, sigma=sigma, order=0)
        vys2 = gaussian_filter(vys, sigma=sigma, order=0)
    else:
        sys.exit('invalid kernel')

    return vxs2, vys2


def calc_c_pearson(vx1, vx2, vy1, vy2, fov=None):
    vx1f, vx2f, vy1f, vy2f = vx1[fov], vx2[fov], vy1[fov], vy2[fov]
    c_pearson = np.sum(vx1f*vx2f + vy1f*vy2f) / np.sqrt(np.sum(vx1f**2 + vy1f**2)*np.sum(vx2f**2 + vy2f**2))
    return c_pearson


# output directory for the drifting images
datadir = os.path.join(os.environ['DATA'], 'sanity_check/stein_series/calibration2')
datadir_stein = os.path.join(os.environ['DATA'], 'Ben/SteinSDO/')
# number of parameter sets
# These files don't really need to be ordered. This will be taken care for by sorting per index key.
filelist = sorted(glob.glob(os.path.join(datadir, 'param_sweep_*.csv')))
# Concatenate all csv file content into one dataframe
df_list = [pd.read_csv(f) for f in filelist]
df = pd.concat(df_list, axis=0, ignore_index=True)
df.set_index('index', inplace=True)
df.sort_index(inplace=True)

# unit in m/s for Stein simulation for 1 px / frame interval
u = 368000 / 60

trange = [0, 60]
fwhm = 7
trim = 10 # Same as Benoit
fov = np.s_[trim:-trim:fwhm, trim:-trim:fwhm]
# Get Stein velocity
svx_files = sorted(glob.glob(os.path.join(datadir_stein,'SDO_vx*.fits')))
svy_files = sorted(glob.glob(os.path.join(datadir_stein,'SDO_vy*.fits')))
vx_stein, vy_stein = load_vel_mean((svx_files, svy_files), trange)
# smooth the Stein velocities
vx_stein_sm, vy_stein_sm = smooth_vel(vx_stein, vy_stein, fwhm, kernel='boxcar')


# List of balltracked velocity flows
filelist = sorted(glob.glob(os.path.join(datadir, 'mean_velocity*.npz')))

# Create new columns in the dataframes
df.insert(8, 'corr_uncal', -1)
df.insert(9, 'corr', -1)
df.insert(10, 'corr_top', -1)
df.insert(11, 'corr_bot', -1)
df.insert(12, 'MAE_uncal_vx', -1)
df.insert(13, 'MAE_uncal_vy', -1)
df.insert(14, 'MAE_cal_vx', -1)
df.insert(15, 'MAE_cal_vy', -1)
df.insert(16, 'RMSE_uncal_vx', -1)
df.insert(17, 'RMSE_uncal_vy', -1)
df.insert(18, 'RMSE_cal_vx', -1)
df.insert(19, 'RMSE_cal_vy', -1)


for f in filelist:
    # Parse the index from that filename
    regex = re.compile(r'\d+')
    idx = int(regex.findall(Path(f).stem)[0])

    with np.load(f) as vel:
        print(f'file: {f} - idx = {idx}')
        vx_top_cal = vel['vx_top'] * df['p_top_0'].loc[idx] * u
        vy_top_cal = vel['vy_top'] * df['p_top_0'].loc[idx] * u
        vx_bot_cal = vel['vx_bot'] * df['p_bot_0'].loc[idx] * u
        vy_bot_cal = vel['vy_bot'] * df['p_bot_0'].loc[idx] * u
        # Calibrate velocity
        vx_ball_cal = 0.5 * (vx_top_cal + vx_bot_cal)
        vy_ball_cal = 0.5 * (vy_top_cal + vy_bot_cal)
        vx_ball_uncal = 0.5 * (vel['vx_top'] + vel['vx_bot']) * u
        vy_ball_uncal = 0.5 * (vel['vy_top'] + vel['vy_bot']) * u

    error_vx_cal_top = (vx_stein_sm[fov].ravel() - vx_top_cal[fov].ravel())
    error_vx_cal_bot = (vx_stein_sm[fov].ravel() - vx_bot_cal[fov].ravel())
    df.loc[idx, 'MAE_cal_vx_top'] = np.mean(np.abs(error_vx_cal_top))
    df.loc[idx, 'MAE_cal_vx_bot'] = np.mean(np.abs(error_vx_cal_bot))
    df.loc[idx, 'RMSE_cal_vx_top'] = np.sqrt(np.mean(error_vx_cal_top ** 2))
    df.loc[idx, 'RMSE_cal_vx_bot'] = np.sqrt(np.mean(error_vx_cal_bot ** 2))

    error_uncal_vx = (vx_stein_sm[fov].ravel() - vx_ball_uncal[fov].ravel())
    df.loc[idx, 'RMSE_uncal_vx'] = np.sqrt(np.mean(error_uncal_vx ** 2))
    df.loc[idx, 'MAE_uncal_vx'] = np.mean(np.abs(error_uncal_vx))

    error_uncal_vy = (vy_stein_sm[fov].ravel() - vy_ball_uncal[fov].ravel())
    df.loc[idx, 'RMSE_uncal_vy'] = np.sqrt(np.mean(error_uncal_vy ** 2))
    df.loc[idx, 'MAE_uncal_vy'] = np.mean(np.abs(error_uncal_vy))

    error_cal_vx = (vx_stein_sm[fov].ravel() - vx_ball_cal[fov].ravel())
    df.loc[idx, 'RMSE_cal_vx'] = np.sqrt(np.mean(error_cal_vx ** 2))
    df.loc[idx, 'MAE_cal_vx'] = np.mean(np.abs(error_cal_vx))

    error_cal_vy = (vy_stein_sm[fov].ravel() - vy_ball_cal[fov].ravel())
    df.loc[idx, 'RMSE_cal_vy'] = np.sqrt(np.mean(error_cal_vy ** 2))
    df.loc[idx, 'MAE_cal_vy'] = np.mean(np.abs(error_cal_vy))


    # Calculate correlation with Stein simulation
    df.loc[idx, 'corr_uncal'] = calc_c_pearson(vx_stein_sm, vx_ball_uncal, vy_stein_sm, vy_ball_uncal, fov=fov)
    df.loc[idx, 'corr'] = calc_c_pearson(vx_stein_sm, vx_ball_cal, vy_stein_sm, vy_ball_cal, fov=fov)
    df.loc[idx, 'corr_top'] = calc_c_pearson(vx_stein_sm, vx_top_cal, vy_stein_sm, vy_top_cal, fov=fov)
    df.loc[idx, 'corr_bot'] = calc_c_pearson(vx_stein_sm, vx_bot_cal, vy_stein_sm, vy_bot_cal, fov=fov)


df.to_csv(os.path.join(os.environ['DATA'], 'sanity_check/stein_series/correlation_dataframe2.csv'), index=False)

