import os, glob, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import pandas as pd
import numpy as np
from astropy.io.fits import getdata
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from optimization import inputs


def load_vel_mean(v_files, trange):
    """ Load the velocity files and average over a time range """
    vx_files_subset = v_files[0][trange[0]:trange[1]]
    vy_files_subset = v_files[1][trange[0]:trange[1]]
    # Get the mean of the velocity components
    vx = np.array([getdata(f) for f in vx_files_subset]).mean(axis=0)
    vy = np.array([getdata(f) for f in vy_files_subset]).mean(axis=0)
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



# number of parameter sets
# These files don't really need to be ordered. This will be taken care for by sorting per index key.
filelist_csv = sorted(Path(inputs.outputdir, 'param_sweep_files').glob('param_sweep_*.csv'))
# Concatenate all csv file content into one dataframe
df = pd.concat([pd.read_csv(f) for f in filelist_csv], axis=0, ignore_index=True)
df.drop(df.columns[0], axis=1, inplace=True)
df.set_index(['index', 'kernel'], inplace=True)
df.sort_index(inplace=True)
# Create new columns in the dataframes
df.insert(8, 'corr_uncal', -1)
df.insert(9, 'corr', -1)
df.insert(10, 'corr_top', -1)
df.insert(11, 'corr_bot', -1)

df.insert(12, 'MAE_uncal_vx', -1)
df.insert(13, 'MAE_uncal_vy', -1)
df.insert(14, 'MAE_cal_vx', -1)
df.insert(15, 'MAE_cal_vy', -1)
df.insert(16, 'MAE_cal_vx_top', -1)
df.insert(17, 'MAE_cal_vx_bot', -1)

df.insert(18, 'RMSE_uncal_vx', -1)
df.insert(19, 'RMSE_uncal_vy', -1)
df.insert(20, 'RMSE_cal_vx', -1)
df.insert(21, 'RMSE_cal_vy', -1)
df.insert(22, 'RMSE_cal_vx_top', -1)
df.insert(23, 'RMSE_cal_vx_bot', -1)
df.insert(24, 'MAE_discrep', -1)
df.insert(25, 'MAPE', -1)
df.insert(26, 'MAPD', -1)

# unit in m/s for sim simulation for 1 px / frame interval
u = inputs.v_scale


trange = inputs.bt_params['trange']
fwhm = inputs.maps_params['fwhm']
trim = 10 # Same as Benoit
fov = np.s_[trim:-trim:fwhm, trim:-trim:fwhm]
# Get the ground truth velocity (simulation)
svx_files = sorted(Path(inputs.inputdir).glob('vy_out*.fits'))
svy_files = sorted(Path(inputs.inputdir).glob('vz_out*.fits'))
vx_sim, vy_sim = load_vel_mean((svx_files, svy_files), trange)
# smooth the simulation velocities
kernel = inputs.maps_params['kernel']
vx_sim_sm, vy_sim_sm = smooth_vel(vx_sim, vy_sim, fwhm, kernel=kernel)
# magnitude
v_sim = np.sqrt(vx_sim_sm ** 2 + vy_sim_sm ** 2)

# List of balltracked velocity flows
filelist_npz = sorted(glob.glob(os.path.join(inputs.outputdir, 'mean_velocity_files', f'mean_velocity_{kernel}*.npz')))
print(f'len(filelist_npz) = {len(filelist_npz)}')

for f in filelist_npz:
    with np.load(f) as vel:
            idx = int(vel['index'])
            print(f'file: {f} - idx = {idx}')
            # Query string to get the dataframe rows at the right kernel
            ker = f'kernel=="{kernel}"'

            vx_top_cal = vel['vx_top'] * df.query(ker).loc[idx, 'p_top_0'].values[0] * u
            vy_top_cal = vel['vy_top'] * df.query(ker).loc[idx, 'p_top_0'].values[0] * u
            vx_bot_cal = vel['vx_bot'] * df.query(ker).loc[idx, 'p_bot_0'].values[0] * u
            vy_bot_cal = vel['vy_bot'] * df.query(ker).loc[idx, 'p_bot_0'].values[0] * u
            # Calibrate velocity
            vx_ball_cal = 0.5 * (vx_top_cal + vx_bot_cal)
            vy_ball_cal = 0.5 * (vy_top_cal + vy_bot_cal)
            vx_ball_uncal = 0.5 * (vel['vx_top'] + vel['vx_bot']) * u
            vy_ball_uncal = 0.5 * (vel['vy_top'] + vel['vy_bot']) * u
            # magnitudes - top, bottom and both
            v_ball_top_cal = np.sqrt(vx_top_cal**2 + vy_top_cal ** 2)
            v_ball_bot_cal = np.sqrt(vx_bot_cal**2 + vy_bot_cal ** 2)
            v_ball_cal = np.sqrt(vx_ball_cal ** 2 + vy_ball_cal ** 2)

            error_vx_cal_top = (vx_sim_sm[fov].ravel() - vx_top_cal[fov].ravel())
            error_vx_cal_bot = (vx_sim_sm[fov].ravel() - vx_bot_cal[fov].ravel())
            df.loc[(idx, kernel), 'MAE_cal_vx_top'] = np.mean(np.abs(error_vx_cal_top))
            df.loc[(idx, kernel), 'MAE_cal_vx_bot'] = np.mean(np.abs(error_vx_cal_bot))
            df.loc[(idx, kernel), 'RMSE_cal_vx_top'] = np.sqrt(np.mean(error_vx_cal_top ** 2))
            df.loc[(idx, kernel), 'RMSE_cal_vx_bot'] = np.sqrt(np.mean(error_vx_cal_bot ** 2))

            error_uncal_vx = (vx_sim_sm[fov].ravel() - vx_ball_uncal[fov].ravel())
            df.loc[(idx, kernel), 'RMSE_uncal_vx'] = np.sqrt(np.mean(error_uncal_vx ** 2))
            df.loc[(idx, kernel), 'MAE_uncal_vx'] = np.mean(np.abs(error_uncal_vx))

            error_uncal_vy = (vy_sim_sm[fov].ravel() - vy_ball_uncal[fov].ravel())
            df.loc[(idx, kernel), 'RMSE_uncal_vy'] = np.sqrt(np.mean(error_uncal_vy ** 2))
            df.loc[(idx, kernel), 'MAE_uncal_vy'] = np.mean(np.abs(error_uncal_vy))

            error_cal_vx = (vx_sim_sm[fov].ravel() - vx_ball_cal[fov].ravel())
            df.loc[(idx, kernel), 'RMSE_cal_vx'] = np.sqrt(np.mean(error_cal_vx ** 2))
            df.loc[(idx, kernel), 'MAE_cal_vx'] = np.mean(np.abs(error_cal_vx))

            error_cal_vy = (vy_sim_sm[fov].ravel() - vy_ball_cal[fov].ravel())
            df.loc[(idx, kernel), 'RMSE_cal_vy'] = np.sqrt(np.mean(error_cal_vy ** 2))
            df.loc[(idx, kernel), 'MAE_cal_vy'] = np.mean(np.abs(error_cal_vy))

            # Calculate correlation with sim simulation
            df.loc[(idx, kernel), 'corr_uncal'] = calc_c_pearson(vx_sim_sm, vx_ball_uncal, vy_sim_sm, vy_ball_uncal, fov=fov)
            df.loc[(idx, kernel), 'corr'] = calc_c_pearson(vx_sim_sm, vx_ball_cal, vy_sim_sm, vy_ball_cal, fov=fov)
            df.loc[(idx, kernel), 'corr_top'] = calc_c_pearson(vx_sim_sm, vx_top_cal, vy_sim_sm, vy_top_cal, fov=fov)
            df.loc[(idx, kernel), 'corr_bot'] = calc_c_pearson(vx_sim_sm, vx_bot_cal, vy_sim_sm, vy_bot_cal, fov=fov)

            df.loc[(idx, kernel), 'MAPE'] = np.median(np.abs((v_sim - v_ball_cal)[fov] / v_sim[fov]).ravel()) * 100

            # Top/bottom discrepancy
            v_ball_discrep = np.abs(v_ball_top_cal - v_ball_bot_cal)
            df.loc[(idx, kernel), 'MAE_discrep'] = v_ball_discrep[fov].mean() / np.sqrt(2)
            df.loc[(idx, kernel), 'MAPD'] = np.median((v_ball_discrep[fov] / v_ball_cal[fov]).ravel()) / np.sqrt(2) * 100


# Don't ignore the index, it is a multi-level index that needs to be preserved.
df.to_csv(Path(inputs.outputdir, f'correlation_dataframe_{kernel}.csv'))
