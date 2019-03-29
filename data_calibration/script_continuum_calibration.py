import os
import drms
import pandas as pd
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
from sunpy.time import parse_time
import numpy as np
from scipy import fftpack
from scipy.signal import butter, lfilter, freqz, find_peaks, iirnotch
from scipy.signal import filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b,a


def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b,a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def notch_filter(data, f0, fs):
    Q = 30
    b, a = iirnotch(f0, Q)
    y = lfilter(b, a, data)
    return y


def power_spectrum(sig, time_step, windowing=1):

    datafft = fftpack.fft(sig - sig.mean() * windowing)
    # power spectrum
    power = np.abs(datafft)
    sample_freqs = fftpack.fftfreq(sig.size, d=time_step)
    # Focus only on positive frequencies
    pos_mask = np.where(sample_freqs > 0)[0]
    power2 = power[pos_mask]
    pos_freq = sample_freqs[pos_mask]

    return datafft, power2, pos_freq


def get_npeaks(power_sp, n):
    peak_idx = find_peaks(power_sp, distance=10)[0]
    argsort_peaks = np.argsort(power_sp[peak_idx])
    peak_idx2 = peak_idx[argsort_peaks[::-1]]

    freqs_10max = pos_freqs[peak_idx2[0:n]]

    return freqs_10max

savedir = '/Users/rattie/Data/SDO/TSI/'
c = drms.Client()

c.keys('hmi.Ic_45s')

df_datamedn = c.query('hmi.Ic_45s[2016.04.01_TAI/15d@6m]', key='T_REC, DATAMEDN')
df_datamedn2 = df_datamedn.interpolate()
df_datamedn2['T_REC'] = df_datamedn2['T_REC'].apply(lambda x: parse_time(x))
datetimes_arr = df_datamedn2['T_REC'].values
data = df_datamedn2['DATAMEDN'].values

# Sampling time of 6 min
time_step = 6*60
duration = (datetimes_arr[-1] - datetimes_arr[0]).item() / 1e9
time_vec = np.arange(0, duration+1, time_step)
time_vec_hr = time_vec / 3600

plt.figure(figsize=(10,5))
plt.plot(time_vec_hr, data, 'k-', label = 'original data')
plt.xlabel('Time (hr)')
plt.tight_layout()
plt.show()

# FFT
# Kaiser window
win = np.hamming(len(data))#np.kaiser(len(data), 5)
# power spectrum
datafft1, powersp1, pos_freqs = power_spectrum(data, time_step, windowing=1)
datafft2, powersp2, _ = power_spectrum(data, time_step, windowing=win)
# Get the peak frequencies
freqs_10max1 = get_npeaks(powersp1, 8)
periods_10max1 = np.sort(1 / freqs_10max1 / 3600)

freqs_10max2 = get_npeaks(powersp2, 8)
periods_10max2 = np.sort(1 / freqs_10max2 / 3600)


plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(time_vec_hr, data - data.mean(), 'k-', label='original data')
plt.plot(time_vec_hr, (data - data.mean())*win, 'r-', alpha=0.6, label='Hamming-windowed data')
plt.xlabel('Time (hr)')
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.semilogy(pos_freqs*1000, powersp1, 'k-', label='from original data')
plt.semilogy(pos_freqs*1000, powersp2, 'r-', alpha=0.6, label= 'from Hamming window data')
for freq in freqs_10max1:
    plt.axvline(freq*1000, color='blue', ls='--', lw=1)
    plt.text(freq*1000 - 0.002, 4e5, '{:0.2f} hr'.format(1/freq/3600), rotation=90)
plt.xlabel('Frequency (mHz)')
plt.legend(loc='best')
plt.axis([0, 0.1, 1e2, 1e7])
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(savedir, 'power_spectrum_15days'))



# # Make a high pass filter, removing periodicities of 8 hr and above, i.e remove low frequencies < 1 / 8hr
# lowcut = 1 / 4 / 3600
# periods = np.array([24, 12, 8, 6, 4])*3600
# fcuts = 1 / periods
# lowcuts = 1/(periods+10/60*3600)
# highcuts = 1/(periods-10/60*3600)
# fs = 1/time_step
# for low, high in zip(lowcuts, highcuts):
#     print('low={:f} high={:f} (mHz)'.format(low*1000,high*1000))
#
# butter_filters = [butter_bandpass(low, high, fs, order=3) for low, high in zip((lowcuts, highcuts))]
#
# ## Look at the high pass filter
# plt.figure(figsize=(10,6))
#
# b, a = butter_highpass(lowcut, fs, order=5)
# w, h = freqz(b, a, worN=2000)
# plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = 3" )
# plt.axvline(lowcut, color='black', ls='--')
# plt.xlabel('Frequency (mHz)')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Try notch filters
#
# f0 = fcuts[0]
# Q = 30.0
# b, a = iirnotch(f0, Q, fs=fs)
# filtered_data = filtfilt(b, a, data)
#
# # Filter the data
# datafft1 = fftpack.fft(data)
# sample_freqs = fftpack.fftfreq(data.size, d=time_step)
# # With butterworth
# filtered_data1 = butter_highpass_filter(data, lowcut, fs, order=6)
# # or straight cut
# datafft1[np.abs(sample_freqs) <= lowcut] = 0
# filtered_data2 = np.real(fftpack.ifft(datafft1))
#
# tstart = np.min(np.where(time_vec_hr > 48)[0])
# tend = np.max(np.where(time_vec_hr < 48 + 3*24)[0])
# tslice = slice(tstart, tend)
#
# plt.figure(figsize=(10,5))
# plt.plot(time_vec_hr[tslice]/24, data[tslice], 'k-', label = 'original data')
# plt.plot(time_vec_hr[tslice]/24, filtered_data1[tslice] + data.mean(), 'b-', label = 'filtered data (butterworth high-pass)')
# plt.plot(time_vec_hr[tslice]/24, filtered_data2[tslice] + data.mean(), 'r-', alpha=0.6, label = 'filtered data (straight high-pass)')
# plt.xlabel('Time (days)')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
