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

savedir = '/Users/rattie/Data/SDO/TSI/'
pickled_file = os.path.join(savedir, 'df_cal_2016_2019.pkl')

data_key = 'DATAMEAN'
au = 149597870691.0
c = drms.Client()

# df_cal = c.query('hmi.lev1_cal[2016.01.01-2019.01.01][][?FID=5117 AND CAMERA=2?]', key='DATE__OBS, DATAMEDN, DATAMEAN, DSUN_OBS')
# df_cal.to_pickle(pickled_file)

df_cal = pd.read_pickle(pickled_file)

df_cal['DATE__OBS'] = df_cal['DATE__OBS'].apply(lambda x: parse_time(x))
df_cal2 = df_cal.set_index('DATE__OBS')
df_cal2 = df_cal2.resample('12H').fillna('nearest')
df_cal3 = df_cal2.copy()
df_cal3 = df_cal3.resample('24H').mean()
time_cal = df_cal2.index.copy()
dt_cal = time_cal - time_cal[0]
dt_vec_sec = np.array([dt.total_seconds() for dt in dt_cal])
dt_vec_days = np.array([dt.total_seconds() for dt in dt_cal]) / (3600 * 24)
df_cal2['dmod_datamean'] = df_cal2[data_key] * (df_cal2['DSUN_OBS']/au)**2
df_cal3['dmod_datamean'] = df_cal3[data_key] * (df_cal3['DSUN_OBS']/au)**2
df_cal3.rename(columns={'dmod_datamean':'dmod_mean_binned'}, inplace=True)

dates = df_cal2.index.values


fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(18,11))
df_cal2.plot(ax=axes[0], y=[data_key, 'dmod_datamean'], style='.', grid=True)
#df_cal3.plot(ax=axes[0], y=[data_key, 'dmod_mean'], style='o', color='black')
axes[0].set_title('DATAMEAN of hmi.lev1_cal FID=5117 camera 2')
axes[0].axvline(x=pd.Timestamp('2018-01-31'), lw=1, color='black', ls='--')
axes[0].axvline(x=pd.Timestamp('2018-03-01'), lw=1, color='black', ls='--')
axes[0].set_ylim([36000, 42000])
axes[0].legend(['DATAMEAN', r'DATAMEAN x DSUN_OBS$^2$'])

df_cal2.plot(ax=axes[1], y=[data_key, 'dmod_datamean'], style='.', xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)], grid=True)
df_cal3.plot(ax=axes[1], y=[data_key, 'dmod_mean_binned'], style=[':', '-.'], color='black', xlim=[datetime.date(2018, 2, 1), datetime.date(2018, 2, 28)], grid=True)
axes[1].set_ylim([38000, 39500])
axes[1].legend(['DATAMEAN', r'DATAMEAN x DSUN_OBS$^2$', 'DATAMEAN (averaged)', r'DATAMEAN (averaged)x DSUN_OBS$^2$'])
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'calibration_frames_datamean.png'))

#TODO: Get rid of the sun-sdo distance modulation and look at what's left.


