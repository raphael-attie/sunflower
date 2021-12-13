import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fitstools as fts
from filters import matrix_ffilter_image
from timeit import timeit
from importlib import reload

"""
Fourier analysis of SDO data, frequency domain (t -> omega)
"""

pi = np.pi
cos = np.cos
sin = np.sin


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def get_freq_idx(freqs, fcut1, fcut2):

    mask1 = np.logical_and(freqs >= fcut1, freqs <= fcut2)
    idx = np.where(mask1)[0]
    return idx


def gap_signal(signal, gap_duration, gap_times):
    # Make the signal with gaps
    sgaps = signal.copy()
    gaps0 = np.arange(gap_duration)
    gaps = [gap_time + gaps0 for gap_time in gap_times]
    for gap in gaps:
        sgaps[gap] = 0
    return sgaps


def get_pspectra(signal, tstart, interval, fcut1, fcut2):

    # Create a view that chuncks the data at every tstart element during over interval. No copy.
    if signal.ndim == 1:
        s2 = signal[np.newaxis, tstart:tstart+interval]
    else:
        s2 = signal[:, tstart:tstart+interval]
        # Subracting the mean is useless in this context, as we show the power of non-zero frequencies, that excludes the DC component.

    N = s2.shape[1]
    # Sampling interval
    T = 45
    # Sampling frequency in Hz
    Fs = 1 / 45
    # Spacing between frequency points
    #df = Fs / N
    # freqs = fs/2 * [-1, ... 0, ..., 1-df] => step = 2/n
    # Vector in mHz
    # freqs = fs/2 * np.arange(-1, 1, 2/n) * 1000
    freqs = np.fft.rfftfreq(N, T) * 1000

    # Power spectral density estimate over the time axis = 1.
    psd = 1/(Fs*N) * np.abs(np.fft.rfft(s2, axis=1))**2
    psd[:, 1:-1] *= 2
    # Extract elements at selected frequencies.
    idx = get_freq_idx(freqs, fcut1, fcut2)
    psd = psd[:,idx]

    return psd


def get_pspectra_mirror(signal, tstart, interval, fcut1, fcut2):

    if signal.ndim == 1:
        s2 = signal[np.newaxis, tstart:tstart+interval]
    else:
        s2 = signal[:, tstart:tstart+interval]
    #s2 -= s2.mean()

    s3 = np.fliplr(s2)
    s4 = np.append(s2, s3, axis=1)
    sm = np.append(s3, s4, axis=1)
    #sm -= sm.mean()


    # s += np.random.normal(0,0.1,s.size)
    # sgaps += np.random.normal(0,0.1,s.size)

    N = sm.shape[1]
    # Sampling interval
    T = 45
    # Sampling frequency in Hz
    Fs = 1 / 45
    # Spacing between frequency points
    #df = Fs / n
    # freqs = fs/2 * [-1, ... 0, ..., 1-df] => step = 2/n
    # Vector in mHz
    # freqs = fs/2 * np.arange(-1, 1, 2/n) * 1000
    freqs = np.fft.rfftfreq(N, T) * 1000

    # Power spectral density estimate over the time axis = 1.
    psd = 1 / (Fs * N) * np.abs(np.fft.rfft(sm, axis=1)) ** 2
    psd[:, 1:-1] *= 2
    # Extract elements at selected frequencies.
    idx = get_freq_idx(freqs, fcut1, fcut2)
    psd = psd[:, idx]

    return psd


def get_pspectra_mean(signal, interval, fcut1, fcut2, get_spectra_func):

    tstarts = np.arange(0, signal.shape[1], interval)
    plist = [get_spectra_func(signal, tstart, interval, fcut1, fcut2) for tstart in tstarts]
    psum = np.array([pspectrum.mean(axis=1).mean() for pspectrum in plist])

    return psum

def running_mean(x, N):
    c = np.cumsum(np.insert(x, 0, 0))
    c2 = (c[N:] - c[:-N]) / N
    c3 = x.copy()
    c3[N - 1:] = c2
    return c3

###### Load the data
f1        = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'
f2        = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_Dopplergram.fits'


h   = fts.fitsheader(f1)
nt    = h['ZNAXIS3']
nx    = h['ZNAXIS1']
ny    = h['ZNAXIS2']

# Read whole cube
icube = fts.fitsread(f1)
sicube = icube[:, 0:int(nx/8), :]
# Mean over all the pixels
imean = sicube.reshape([sicube.shape[0]*sicube.shape[1], sicube.shape[2]]).mean(axis=0)

dcube = fts.fitsread(f2)
dscube = dcube[:, 0:int(nx/8), :]
# Mean over all the pixels
dmean = dscube.reshape([dscube.shape[0]*dscube.shape[1], dscube.shape[2]]).mean(axis=0)
# This will show the 24-h orbital effect. need to average over that.
N = int(24*60*60/45)
dmean_avg = running_mean(dmean, N)

# Time cadence
T = 45
minutes = 1/60
hours = 1/3600

timeline0 = np.arange(nt)*T

plt.figure(0)
plt.plot(timeline0*hours, imean/imean.max(), 'g-')
plt.plot(timeline0*hours, dmean, 'r-')
plt.xlabel('Time [hr]')
plt.ylabel('Mean')

plt.savefig("/Users/rattie/Desktop/mean_intensity_continuum.png")


# Time span of each integration window, in number of frames
interval = 40 # 30 min

# Cut frequencies (mHz)
fcut1 = 2.222
fcut2 = 4.444

# RAM friendly: Work in 8 chunks over y axis and loop over these.
nchunks = 8
nyc = int(ny/nchunks)
yaxis_chunks = np.arange(ny).reshape([nchunks, nyc])

psd_list = []

for i in range(nchunks):

    yslice = slice(yaxis_chunks[i, 0], yaxis_chunks[i,-1]+1)
    s = fts.fitsread(f2, slice(0, int(nx/8)), yslice, slice(0, nt))
    s2 = s.view(dtype=np.float32).reshape([int(s.shape[0] * s.shape[1]), s.shape[2]])
    psd = get_pspectra_mean(s2, interval, fcut1, fcut2, get_pspectra_mirror)
    psd_list.append(psd)

psd = sum(psd_list)/nchunks


# Get a proper timeline in physical time.

tstarts = np.arange(0, nt, interval)
# time line in seconds
timeline = (tstarts + interval/2)*T



plt.figure(2)
plt.plot(timeline * hours, psd/psd.mean(), 'r-')
plt.xlabel("time [hr]")
plt.ylabel("Mean power spectral density")
plt.savefig("/Users/rattie/Desktop/continuumPSD_normalized.png")
# Power Spectral Density averaged over field of view
#%timeit psd = get_pspectra_mean(s2, interval, fcut1, fcut2, get_pspectra)
