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
# Sampling time
ts = 6*60
# Sampling frequency
fs = 1 / ts
# 4 hr period
periods = np.array([4, 6, 8, 12, 24]) * 3600
freqs = 1/periods
weights = [.05, .3, .5, 1, .5]

# Duration of 10 days
duration = 30 * 24 * 3600
# Generate reference signal
time_vec = np.arange(0, duration, ts)
time_vec_hr = time_vec/3600
Nfft = len(time_vec)
sigs = [100*w*np.sin(2*np.pi * f * time_vec) for f, w in zip(freqs, weights)]
# Generate noise
noise_min = 1
noise_max = 200
noise = np.random.uniform(low=noise_min, high=noise_max, size=time_vec.size)
sig = sigs[0] + sigs[1] + sigs[2] + sigs[3] + sigs[4] + noise
win = np.hamming(len(sig)) #np.kaiser(len(data), 5)

# Create the filter
fw = 0.001e-3
fsig = np.fft.fft(sig)
ffilters, lft1s, lft2s = zip(*[fourier_filter(Nfft, duration, f, fw, 'keep', gaussian) for f in freqs])
# Combined filter
comb_filter = (1-ffilters[0]) * (1-ffilters[1]) * (1-ffilters[2]) * (1-ffilters[3]) * (1-ffilters[4])
# Apply single filter
fsigf = fsig * comb_filter
filtered_sig = np.real(np.fft.ifft(fsigf))

# Get frequencies associated with the FFT and power spectrum

df = fs/Nfft
ffreqs = np.arange(0, Nfft) * df
ffreqs_m = ffreqs*1000
power = np.abs(fsig)

#  plot
fig = plt.figure(figsize=(18,11))
ax1 = plt.subplot(3,1,1)
ax1.plot(time_vec_hr, sig, 'k-', alpha=0.7, label='real signal (incl. noise)')
ax1.set_ylim([noise.mean() - 300, noise.mean() + 300])
ax1.set_xlabel('Time (hr)')
ax1.set_xlim([70, 100])
ax1.grid(True)
ax1.set_title('Signal to filter')
ax1.legend(loc='upper left')
ax1R = ax1.twinx()
for i in range(len(freqs)):
    ax1R.plot(time_vec_hr, sigs[i], label='{:0.0f}-hr signal'.format(periods[i]/3600))
ax1R.legend(loc='upper right')

ax2 = plt.subplot(3,1,2)
ax2.plot(ffreqs_m, power, lw=2, label='power spectrum')
ax2.set_yscale('log')
ax2.set_xlim([0, 0.1])
ax2.set_ylim([1e2, 1e7])
ax2.grid(True)
ax2.legend(loc='upper left')
ax2.set_xlabel('Freq (mHz)')
ax2R = ax2.twinx()
for i in range(len(freqs)):
    ax2R.plot(ffreqs_m, ffilters[i], alpha=0.8, ls='--', lw=1, label='filter @{:0.0f} hr'.format(periods[i]/3600))
ax2R.plot(ffreqs_m, comb_filter, 'k-', alpha=0.8, lw=1, label='comb. filter'.format(periods[0]/3600, periods[1]/3600))

ax2R.set_ylim([-0.1, 1.2])
ax2R.legend(loc='upper right')
ax2.set_title('Power spectrum')

#TODO: Add MAE (Mean absolute error) on the 3rd graph...
AE = noise - filtered_sig
NE = (noise - filtered_sig) / (noise_max - noise_min) * 100
RE = (noise - filtered_sig) / noise * 100


ax3 = plt.subplot(3,1,3)
ax3.plot(time_vec_hr, noise, 'k-', label='reference signal (noise-free)')
ax3.plot(time_vec_hr, filtered_sig, 'r-', alpha=0.6, label='filtered signal (gaussian)')
#plt.plot(t, filtered_y2, 'g--', alpha=1, label='filtered signal (butterworth')
ax3.set_xlim([70, 100])
#ax3.set_ylim([0, 0.4])
ax3.grid(True)
ax3.legend(loc = 'upper left')
ax3.set_xlabel('Time (hr)')
ax3.set_title('Reference and filtered signal. MNAE: {:0.1f}% ; MRE: {:0.1f}%'.format(np.median(np.abs(NE)), np.median(np.abs(RE))))
ax3R = ax3.twinx()
ax3R.plot(time_vec_hr, NE, 'b-', label = 'relative error')
ax3R.set_ylim([-10, 10])
ax3R.legend(loc = 'upper right')

plt.tight_layout()

plt.savefig('/Users/rattie/Data/SDO/TSI/filter_synthetic_sdo_data_{:0.0f}days.png'.format(duration / (24*3600)))

bins1 = np.arange(-50, 50, 0.5)
bins2 = np.arange(0, 11, 0.1)

fig = plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
plt.hist(NE, bins=bins1, linewidth=1, edgecolor='black', density=True)
plt.xlim([-10,10])
plt.grid(True)
plt.xlabel('Percentage error')
plt.ylabel('Density')
plt.title('Distribution of relative error')
ax3 = plt.subplot(1,2,2)
plt.hist(np.abs(NE), bins=bins2, linewidth=1, edgecolor='black', density=True, label='MNAE: {:0.0f}'.format(np.median(np.abs(NE))))
plt.xlabel('Percentage error')
plt.ylabel('Counts')
plt.title('Distribution of absolute relative error')
ax3R = ax3.twinx()
ax3R.hist(np.abs(NE), bins=bins2, linewidth=1, edgecolor='red', density=True, cumulative=True, histtype='step')
#plt.grid(True)
ax3R.set_ylabel('Cumulative density')
ax3R.set_ylim([0.4, 1.1])
ax3R.set_xlim([0,10])
ax3R.grid(True)
plt.tight_layout()
