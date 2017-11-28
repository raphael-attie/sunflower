import os
import numpy as np
import matplotlib.pyplot as plt
from fitstools import fitsread
from filters import matrix_ffilter_image
import fitsio
import matplotlib.pyplot as plt
from timeit import timeit

"""
Benchmark fourier analysis on 2D arrays made from replicating 1D data,
using same function as fourier_analysis1.py
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
#    s2 -= s2.mean(axis=1)[:,np.newaxis]

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


###### Create the signal


# Time array over 5 days, in seconds
n0 = 24*5*3600
t0 = np.arange(n0)

# Generate a 5-min period signal sampled at 1s, and another.
w0 = 2*pi / (60*5)
s0 = sin(w0*t0)
# Generate other signals with a very slow frequency / long period of 24 hours
Ts = np.array([24*60])
w02 = 2*pi / (60*Ts)

a = np.array([sin(w*t0) for w in w02])
s_orbital = s0 + 0.5*a.sum(axis=0)

# Sample every 45s,
T = 45
# Sample every 45s
sref = s0[0::T]
s45_orbital = s_orbital[0::T]
# Sampled timeline and sampled singal length
t = t0[0::T]
n = t.size

# Add noise
# s += s*np.random.normal(0,0.1, n)
# s2 += s*np.random.normal(0,0.1, n)


# 1 hr in # of time step in s is 80
hour = 80
day = 24 * hour
gap_duration = 1*hour
start = 7*hour
# Gaps: They are spaced by 24 hours. The 1st starts 8 hours after the begining.
gap_times = np.arange(5) * day + start

sgaps = gap_signal(sref, gap_duration, gap_times)
sgaps_orbital = gap_signal(s45_orbital, gap_duration, gap_times)

sgaps = np.tile(sgaps[np.newaxis, :], [10, 1])
sgaps_orbital = np.tile(sgaps_orbital[np.newaxis, :], [10, 1])
# sgaps -= np.mean(sgaps, axis=0)
# sgaps += np.random.normal(0,0.1,s.size)

# Cut frequencies (mHz)
fcut1 = 2.222
fcut2 = 4.444
# Set the times over which we want each power spectrum
# Interval in # of frames: 30 in = 40 frames
interval = 40


%timeit get_pspectra_mean(sgaps, interval, fcut1, fcut2, get_pspectra_mirror)

