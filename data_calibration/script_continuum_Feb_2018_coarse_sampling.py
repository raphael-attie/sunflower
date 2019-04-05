import os
import drms
import pandas as pd
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
from sunpy.time import parse_time
import datetime
from datetime import timedelta
import numpy as np
plt.ion()


def gaussian(x, pos, wid):
    g = np.exp(-((x-pos)/(0.6005615*wid))**2)
    return g


def butterworth(x, pos, wid, norder):
    btwf = 1. / (1 + ((x-pos) / wid)**(2 * norder))
    return btwf


def fourier_filter(signal_length, duration, f0, fw, mode, ffunction):

    """
    From https://terpconnect.umd.edu/~toh/spectrum/FouFilter.m with a correction of how to build the filter,
    adapted from Matlab to Python

    :param signal_length: number of samples in the signal
    :param duration: total duration of the signal in physical units
    :param f0: center frequency in Hz
    :param fw: frequency width in Hz
    :param mode: whether we 'keep' (or 1) or 'reject' (or 0) the selected frequency band
    :return: filtered signal
    """

    assert(mode is 'keep' or mode is 'reject'), "fourier_filter mode argument not recognized. Must be 'keep' or 'reject'"

    center = f0 * duration
    width = fw * duration

    lft1 = np.arange(0, np.floor(signal_length / 2))
    lft2 = np.arange(np.floor((signal_length / 2)), signal_length)
    # computer filter shape
    ffilter1 = ffunction(lft1, center, width)
    ffilter2 = ffunction(lft2, signal_length - center, width)
    # Concatenate the filters
    ffilter = np.concatenate((ffilter1, ffilter2))

    if mode is 'reject':
        ffilter = 1 - ffilter


    return ffilter, lft1, lft2


def process_cal(df):

    df['DATE__OBS'] = df['DATE__OBS'].apply(lambda x: parse_time(x))
    df_cal2 = df_cal.set_index('DATE__OBS')
    df_cal2 = df_cal2.resample('12H').fillna('nearest')
    df_cal3 = df_cal2.copy()
    df_cal3 = df_cal3.resample('24H').mean()
    df_cal2['DSUN_OBS'] = df_cal2['DSUN_OBS']/au
    df_cal3['DSUN_OBS'] = df_cal3['DSUN_OBS']/au
    df_cal2['dmod_datamean'] = df_cal2[data_key] * df_cal2['DSUN_OBS'] ** 2
    df_cal3['dmod_datamean'] = df_cal3[data_key] * df_cal3['DSUN_OBS'] ** 2
    df_cal3.rename(columns={'dmod_datamean': 'dmod_mean_binned'}, inplace=True)

    return df_cal2, df_cal3


savedir = '/Users/rattie/Data/SDO/TSI/'
data_key = 'DATAMEAN'
au = 149597870691.0
c = drms.Client()

c.keys('hmi.Ic_45s')

pickled_file = os.path.join(savedir, 'df_datamedn_Feb_2018.pkl')
df_data = c.query('hmi.Ic_45s[2018.01.01_TAI/90d@24h]', key='T_REC, DATAMEDN, DATAMEAN, DSUN_OBS, CAMERA')
#df_data = pd.read_pickle(pickled_file)
df_data['T_REC'] = df_data['T_REC'].apply(lambda x: parse_time(x))
df_data = df_data.set_index('T_REC')
df_data['DSUN_OBS'] = df_data['DSUN_OBS']/au
df_data['DSUN_OBS2'] = df_data['DSUN_OBS']**2
df_data2 = df_data.interpolate()

datetimes_arr = df_data2.index.values
df_data2['dmod_datamean'] = df_data2[data_key] * df_data2['DSUN_OBS2']

# Calibration data
pickled_file2 = os.path.join(savedir, 'df_cal_2016_2019.pkl')
df_cal = pd.read_pickle(pickled_file2)
df_cal2, df_cal3 = process_cal(df_cal)

combined_index = df_data2.index.union( df_cal2.index )


fig = plt.figure(figsize=(18,11))
ax1 = plt.subplot(2,1,1)
df_data2.plot(ax=ax1, y=[data_key, 'dmod_datamean'], xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)],
              style='-', grid=True)
ax1.set_xlabel('Date')
ax1.legend(['DATAMEAN', r'DATAMEAN x DSUN_OBS$^2$'], loc='best')

# if data_key == 'DATAMEAN':
#     #ax1.set_ylim([46600, 47050])
# elif data_key == 'DATAMEDN':
#     ax1.set_ylim([48000, 48450])

ax1.grid(True)
ax1.set_title('{:s}'.format(data_key))

ax2 = plt.subplot(2,1,2)
df_cal2.plot(ax=ax2, y=[data_key, 'dmod_datamean'], style='.', xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)], grid=True)
df_cal3.plot(ax=ax2, y=[data_key, 'dmod_mean_binned'], style=[':', '-.'], color='black', xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)], grid=True)
ax2.set_ylim([38000, 39500])
ax3 = ax2.twinx()
df_data2.plot(ax=ax3, y=['DSUN_OBS2'], style='--', color='green', xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)], grid=True)
ax2.set_title('Calibration DATAMEAN (hmi.lev1_cal[][][?FID=5117?]) ')
ax2.legend(['cal DATAMEAN', r'cal DATAMEAN x DSUN_OBS$^2$', 'cal DATAMEAN (averaged)', r'cal DATAMEAN (averaged)x DSUN_OBS$^2$'], loc='center left')
ax3.legend([r'DSUN_OBS$^2$'])
plt.tight_layout()

plt.savefig(os.path.join(savedir, 'frames_datamean_with_calibration_Feb_2018.png'))

# ndays = round(duration / (3600 * 24))
# tsuffix = parse_time(datetimes_arr[0]).strftime('%Y-%m-%d')
#plt.savefig('/Users/rattie/Data/SDO/TSI/filter_true_sdo_data_{:0.0f}days_@24hr_{:s}_{:d}d_{:s}.png'.format(duration / (24*3600), tsuffix, ndays, data_key))
