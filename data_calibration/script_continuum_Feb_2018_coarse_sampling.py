import os
import drms
import pandas as pd
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
from sunpy.time import parse_time
import datetime
import numpy as np
from scipy.signal import detrend
plt.ion()


def process_cal(df):

    df['DATE__OBS'] = df['DATE__OBS'].apply(lambda x: parse_time(x))
    df_cal2 = df_cal.set_index('DATE__OBS')
    df_cal2 = df_cal2.resample('12H').fillna('nearest')
    df_cal3 = df_cal2.copy()
    df_cal3 = df_cal3.resample('24H').mean()
    df_cal_24h = df_cal3.copy()
    df_cal2['DSUN_OBS'] = df_cal2['DSUN_OBS']/au
    df_cal3['DSUN_OBS'] = df_cal3['DSUN_OBS']/au
    df_cal2['dmod_datamean'] = df_cal2[data_key] * df_cal2['DSUN_OBS'] ** 2
    df_cal3['dmod_datamean'] = df_cal3[data_key] * df_cal3['DSUN_OBS'] ** 2
    df_cal3.rename(columns={'dmod_datamean': 'dmod_mean_binned'}, inplace=True)

    return df_cal2, df_cal3, df_cal_24h


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
df_cal2, df_cal3, df_cal_24h = process_cal(df_cal)


#combined_index = df_data2.index.union( df_cal2.index )

# TODO: need to synchronize df_data2 and df_cal and divide df_data by df_cal to flat-field it.

df_cal_sync, df_data_sync = df_cal_24h.align(df_data2)
df_data_sync['dmod_datamean2'] = df_data_sync['DATAMEAN']/df_cal_sync['DATAMEAN']


# Synchronize df_data2 and df_cal

fig = plt.figure(figsize=(18,11))
ax1 = plt.subplot(3,1,1)
df_data2.plot(ax=ax1, y=[data_key, 'dmod_datamean'], xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)],
              style='-', grid=True)
ax1.set_xlabel('Date')
ax1.legend(['DATAMEAN', r'DATAMEAN x DSUN_OBS$^2$'], loc='best')

ax1.grid(True)
ax1.set_title('{:s}'.format(data_key))

ax2 = plt.subplot(3,1,2)
df_cal2.plot(ax=ax2, y=[data_key, 'dmod_datamean'], style='.', xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)], grid=True)
df_cal3.plot(ax=ax2, y=[data_key, 'dmod_mean_binned'], style=[':', '-.'], color='black', xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)], grid=True)
ax2.set_ylim([38000, 39500])
ax3 = ax2.twinx()
df_data2.plot(ax=ax3, y=['DSUN_OBS2'], style='--', color='green', xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)], grid=True)
ax2.set_title('Calibration DATAMEAN (hmi.lev1_cal[][][?FID=5117?]) ')
ax2.legend(['cal DATAMEAN', r'cal DATAMEAN x DSUN_OBS$^2$', 'cal DATAMEAN (averaged)', r'cal DATAMEAN (averaged)x DSUN_OBS$^2$'], loc='center left')
ax3.legend([r'DSUN_OBS$^2$'])


ax3 = plt.subplot(3,1,3)
df_data_sync.plot(ax=ax3, y=['dmod_datamean2'], xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)], ylim=[1.19, 1.21],
              style='-', grid=True)

ax3.set_xlabel('Date')
ax3.legend(['dmod_datamean2 (~DATAMEAN detrented)'], loc='best')

plt.tight_layout()


plt.savefig(os.path.join(savedir, 'frames_datamean_with_calibration_Feb_2018.png'))

# ignore the calibration data and work again on the true DATAMEAN, extract to numpy and detrend it.
dmean = df_data2['dmod_datamean'].values
# Detrend
dmean_detrended = detrend(dmean, type='linear')
df_data2['DATAMEAN_DETRENDED'] = dmean_detrended + abs(dmean_detrended.min())
mask = (df_data2.index >= np.datetime64('2018-02-01')) & (df_data2.index <= np.datetime64('2018-02-28'))

df_datamean_detrended = df_data2['DATAMEAN_DETRENDED'].copy().loc[mask]


df_datamean_detrended.to_csv(os.path.join(savedir, 'intensity_mean_detrended.csv'))


fig = plt.figure(figsize=(18,11))
ax1 = plt.subplot(1,1,1)
df_data2.plot(ax=ax1, y=['DATAMEAN_DETRENDED'], xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)],
              style='-', grid=True, ylim=[0,60])
ax1.set_xlabel('Date')
ax1.legend(['DATAMEAN_DETRENDED'], loc='best')

ax1.grid(True)
ax1.set_title('{:s}'.format(data_key))

plt.tight_layout()

plt.savefig(os.path.join(savedir, 'datamean_demodulated_detrended_Feb_2018.png'))

df_data2_feb_2018= df_data2.loc[mask]
del df_data2_feb_2018['CAMERA']
del df_data2_feb_2018['DSUN_OBS2']
df_data2_feb_2018.to_csv(os.path.join(savedir, 'intensity_mean_demodulated_detrended_feb_2018.csv'), header=True)

