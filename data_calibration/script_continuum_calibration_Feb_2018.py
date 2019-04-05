import os
import drms
import pandas as pd
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
from sunpy.time import parse_time
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



savedir = '/Users/rattie/Data/SDO/TSI/'
data_key = 'DATAMEDN'

c = drms.Client()

c.keys('hmi.Ic_45s')

pickled_file = os.path.join(savedir, 'df_datamedn_Feb_2018.pkl')
df_data = c.query('hmi.Ic_45s[2018.01.01_TAI/90d@6m]', key='T_REC, DATAMEDN, DATAMEAN, DSUN_OBS')
df_data.to_pickle(pickled_file)
df_data2 = df_data.interpolate()
# df_data = pd.read_pickle(pickled_file)
# df_data2 = df_data.interpolate()
# Get 80 days centered on 160 days
# t1 = round(40*24*60/6)
# t2 = round(120*24*60/6)
# df_datamedn2 = df_datamedn2.iloc[t1:t2]

df_data2['T_REC'] = df_data2['T_REC'].apply(lambda x: parse_time(x))
datetimes_arr = df_data2['T_REC'].values
sig0 = df_data[data_key].values # non-interpolated data
sig = df_data2[data_key].values
#sig = sig - sig.mean()

# Sampling time of 6 min
ts = 6*60
duration = (datetimes_arr[-1] - datetimes_arr[0]).item() / 1e9
time_vec = np.arange(0, duration+1, ts)
time_vec_hr = time_vec / 3600
time_vec_days = time_vec / 3600 / 24
# Windowing
win = np.kaiser(len(sig), 5) #np.hamming(len(sig))
sigw = sig * win
# Create the filter
fw = 0.002e-3
fsig = np.fft.fft(sig)
fsigw = np.fft.fft(sigw)
Nfft = len(time_vec)
# Periodicities and frequencies to filter out
periods = np.array([4, 4.8, 6, 8, 12, 24]) * 3600
freqs = 1/periods
ffilters, lft1s, lft2s = zip(*[fourier_filter(Nfft, duration, f, fw, 'keep', gaussian) for f in freqs])
# Combined filter, scale-reverting the above
comb_filter = (1-ffilters[0]) * (1-ffilters[1]) * (1-ffilters[2]) * (1-ffilters[3]) * (1-ffilters[4]) * (1-ffilters[5])

# Apply single filter
fsigf = fsig * comb_filter
fsigfw = fsigw * comb_filter
filtered_sig = np.real(np.fft.ifft(fsigf))
filtered_sigw = np.real(np.fft.ifft(fsigfw))
# Sampling frequency
fs = 1 / ts
# Get frequencies associated with the FFT and power spectrum
df = fs/Nfft
ffreqs = np.arange(0, Nfft) * df
ffreqs_m = ffreqs*1000
power = np.abs(fsig)
powerw = np.abs(fsigw)


#  plot
time_offset = 31
xlim = [1, 28]
t1 = round(time_offset * 24*60/6)
t2 = round((time_offset + xlim[1])*24*60/6)
tslice = slice(t1,t2)
tvec = time_vec_days - time_offset + 1

title_date = (parse_time(datetimes_arr[0]) + timedelta(days=time_offset+2)).strftime('%b-%Y')

fig = plt.figure(figsize=(18,11))
ax1 = plt.subplot(3,1,1)
ax1.plot(tvec[tslice], sig[tslice], 'r--', label='HMI signal (interpolated)')
ax1.plot(tvec[tslice], sig0[tslice], color='black', ls='-', label='HMI signal (uninterpolated)')
ax1.set_xticks(np.arange(xlim[0], xlim[1]+1))
#ax1.plot(tvec, sigw, 'r-', alpha=0.7, label='windowed signal')
#ax1.set_ylim([50000, 50400])
ax1.set_xlabel('Time (days)')
ax1.set_xlim(xlim)
ax1.grid(True)
ax1.set_title('{:s} in {:s}'.format(data_key, title_date))
ax1.legend(loc='upper left')
ax1R = ax1.twinx()
ax1R.plot(tvec, win, color='gray', ls='--', label = 'windowing')
ax1R.legend(loc='upper right')

ax2 = plt.subplot(3,1,2)
ax2.plot(ffreqs_m, power, 'b-', lw=2, label='power spectrum')
ax2.plot(ffreqs_m, powerw, 'r-', lw=2, alpha=0.7, label='windowed power spectrum')
ax2.set_yscale('log')
ax2.set_xlim([0, 0.1])
ax2.set_ylim([1e1, 1e6])
ax2.grid(True)
ax2.legend(loc='upper left')
ax2.set_xlabel('Freq (mHz)')

ax2R = ax2.twinx()
for i in range(len(freqs)):
    ax2R.plot(ffreqs_m, ffilters[i], alpha=0.8, ls='--', lw=1, label='filter @{:0.1f} hr'.format(periods[i]/3600))

ax2R.plot(ffreqs_m, comb_filter, 'k-', alpha=0.8, lw=1, label='comb. filter'.format(periods[0]/3600, periods[1]/3600))
ax2R.set_ylim([-0.1, 1.2])
ax2R.legend(loc='upper right')
ax2.set_title('Power spectrum')


ax3 = plt.subplot(3,1,3)
ax3.plot(tvec[tslice], filtered_sig[tslice], 'k-', label='filtered signal (gaussian)')
ax3.plot(tvec[tslice], (filtered_sigw/win)[tslice], 'r-', label='filtered signal (windowed, gaussian)')
ax3.grid(True)
ax3.legend(loc = 'upper right')
ax3.set_xlabel('Time (days)')
ax3.set_xlim(xlim)
#ax3.set_ylim([50000, 50400])
#ax3.set_ylim([-100, 100])
ax3.set_xticks(np.arange(xlim[0], xlim[1]+1))
ax3.set_title('filtered data (free of SDO-orbit systematics)')

plt.tight_layout()

ndays = round(duration / (3600 * 24))
tsuffix = parse_time(datetimes_arr[0]).strftime('%Y-%m-%d')
plt.savefig('/Users/rattie/Data/SDO/TSI/filter_true_sdo_data_{:0.0f}days_fw{:0.0f}em6_{:s}_{:d}d_{:s}.png'.format(duration / (24*3600), fw*1e6, tsuffix, ndays, data_key))
