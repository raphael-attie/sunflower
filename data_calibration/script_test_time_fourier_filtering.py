import numpy as np
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
from functools import partial
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


# Seed the random number generator
np.random.seed(1234)
# Sampling time of 6 min
ts = 6*60
# Sampling frequency
fs = 1 / ts
# 4 hr period
period = 4 * 3600
f0 = 1/period
# Duration of 2 days
duration = 6 * 24 * 3600
# Generate reference signal
time_vec = np.arange(0, duration, ts)
time_vec_hr = time_vec/3600
Nfft = len(time_vec)
sig0 = np.sin(2*np.pi/period * time_vec)
# Generate noise
noise_max = 1
noise = np.random.randn(time_vec.size)
sig = sig0 + noise_max * noise / noise.max()
win = np.hamming(len(sig)) #np.kaiser(len(data), 5)

# Create the filter
fw = 0.005e-3
mode = 'keep'
fsig = np.fft.fft(sig)
ffilter, lft1, lft2 = fourier_filter(Nfft, duration, f0, fw, mode, gaussian)
# Apply filter
fsigf = fsig * ffilter
filtered_sig = np.real(np.fft.ifft(fsigf))

# Get frequencies associated with the FFT and power spectrum

df = fs/Nfft
freqs = np.arange(0, Nfft) * df
freqs_m = freqs*1000
power = np.abs(fsig)**2

#  plot
fig = plt.figure(figsize=(16,10))
ax1 = plt.subplot(2,2,1)
ax1.plot(time_vec_hr, sig, 'k-')
ax1.set_ylim([-2, 2])
ax1.set_xlabel('Time (hr)')
ax1.grid(True)
ax1.set_title('Signal to filter')

ax2 = plt.subplot(2,2,2)
ax2.plot(freqs_m, power, lw=2, label='power spectrum')
ax2.set_yscale('log')
ax2.set_ylim([1, 1e7])
ax2.grid(True)
ax2.legend(loc='upper left')
ax2.set_xlabel('Freq (mHz)')
ax2R = ax2.twinx()
ax2R.plot(freqs_m, ffilter, 'r--', alpha=0.5, lw=1, label='filter')
ax2R.set_ylim([-0.1, 1.2])
ax2R.legend(loc='upper right')
ax2.set_title('Power spectrum')

ax3 = plt.subplot(2,2,3)
ax3.plot(time_vec_hr, sig0, 'b-', label='reference signal (noise-free)')
ax3.plot(time_vec_hr, filtered_sig, 'r--', alpha=0.7, label='filtered signal (gaussian)')
#plt.plot(t, filtered_y2, 'g--', alpha=1, label='filtered signal (butterworth')
ax3.set_ylim([-2, 2])
ax3.grid(True)
ax3.legend()
ax3.set_xlabel('Time (hr)')
ax3.set_title('Reference and filtered signal')


ax2_L = fig.add_axes([0.55, 0.11, 0.15, 0.35])
ax2_R = fig.add_axes([0.75, 0.11, 0.15, 0.35])

ax2_L.plot(freqs_m, power)
ax2_L.grid(True)
ax2_L.set_xlim([(f0-0.01e-3)*1000, (f0+0.01e-3)*1000])
ax2_L.ticklabel_format(axis='y', style='sci', scilimits=(0,2))
ax2_L2 = ax2_L.twinx()
ax2_L2.plot(freqs_m, ffilter, 'r-')

ax2_R.plot(freqs_m, power)
ax2_R.grid(True)
ax2_R.set_xlim([freqs_m[-1] - (f0+0.01e-3)*1000, freqs_m[-1] - (f0-0.01e-3)*1000])
ax2_R.ticklabel_format(axis='y', style='sci', scilimits=(0,2))
ax2_R2 = ax2_R.twinx()
ax2_R2.plot(freqs_m, ffilter, 'r-')

