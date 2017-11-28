import os
import numpy as np
import matplotlib.pyplot as plt
from fitstools import fitsread
from filters import matrix_ffilter_image
import fitsio
import matplotlib.pyplot as plt
from timeit import timeit

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

pi = np.pi
cos = np.cos
sin = np.sin

def get_freq_idx(freqs, fcut1, fcut2):

    mask1 = np.logical_and(freqs >= fcut1, freqs <= fcut2)
    idx = np.where(mask1)[0]

    return idx

def gap_signal(s, gap_duration, gap_times):
    # Make the signal with gaps
    sgaps = s.copy()
    gaps0 = np.arange(gap_duration)
    gaps = [gap_time + gaps0 for gap_time in gap_times]
    for gap in gaps:
        sgaps[gap] = 0

    return sgaps


def get_pspectra(signal, T, tstart, interval, fcut1, fcut2):

    # Create a view that chuncks the data at every tstart element during over interval. No copy.
    if signal.ndim == 1:
        s2 = signal[np.newaxis, tstart:tstart+interval]
    else:
        s2 = signal[:, tstart:tstart+interval]
    # Subracting the mean is useless in this context, as we show the power of non-zero frequencies, that excludes the DC component.
    #s2 -= s2.mean(axis=1)[:,np.newaxis]

    N = s2.shape[1]

    # Sampling frequency in Hz
    Fs = 1 / T
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

def get_pspectra_mirror(signal, T, tstart, interval, fcut1, fcut2):

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

    # Sampling frequency in Hz
    Fs = 1 / T
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


def get_pspectra_mean(signal, T, interval, fcut1, fcut2, get_spectra_func):

    if signal.ndim == 1:
        signal_length = signal.shape[0]
    else:
        signal_length = signal.shape[1]

    tstarts = np.arange(0, signal_length, interval)
    plist = [get_spectra_func(signal, T, tstart, interval, fcut1, fcut2) for tstart in tstarts]
    psum = np.array([pspectrum.mean(axis=1).mean() for pspectrum in plist])

    return psum


# def plot_analysis(s, sgaps, tslice, timeline):
#
#     unit_short = 1 / 60
#     unit_long = 1 / 3600
#
#     plt.figure(figsize=(10, 10))
#     plt.subplot(2, 1, 1)
#     plt.plot(t[tslice] * unit_short, s[tslice] * unit_short, 'b-')
#     plt.plot(t[tslice] * unit_short, sgaps[tslice] * unit_short, 'r--')
#     plt.xlabel('time [min]')
#     plt.ylabel('Signal intensity')
#
#     plt.subplot(2, 1, 2)
#     plt.plot(timeline * unit_long, psum, 'b-')
#     plt.plot(timeline * unit_long, psum_gaps, 'r--')
#     plt.xlabel('time [hr]')
#     plt.ylabel('Integrated Power')


filename        = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_Dopplergram.fits'
# Get the whole cube.


# im    = fitsread(filename, slice(0, 1), slice(None), slice(None))
# im2   = im - im.mean()

# plt.figure(1, figsize=(17,8))
# plt.subplot(1,2,1)
# plt.imshow(im, vmin=-1000, vmax=1000, origin='lower', cmap='RdBu')
# plt.colorbar()
# plt.subplot(1,2,2)
# plt.imshow(im2, vmin=-1000, vmax=1000, origin='lower', cmap='RdBu')
# plt.colorbar()

# Time array over 5 days, in seconds
n0 = 24*5*3600
t0 = np.arange(n0)

# Generate a 5-min period signal sampled at 1s
# Amplitude

w5min = 2*pi / (60*5)
s0 = 100 * cos(w5min*t0)
# Generate other signals with a very slow frequency / long period of 24 hours and 28 days
w02 = 2*pi / (24*60*60) # 24 hours period
w03 = 2*pi / (12*24*60*60) # 28 days period


#a = 3000 * sin(w02*t0) + 1800*sin(w03*t0)
w = 2*pi / (28*24*60*60) # 24 hours period

a = 3200 * cos(w02*t0) + 1800*cos(w03*(t0 - 60*60*24*5))

wmod = 2*pi / (60*60*24*9)
s_orbital0 = (s0 + a)
s_orbital = (s0 + a) * cos(pi/180 * 30 * cos(wmod*t0))



# Sample every 45s,
T = 45
# Sample every 45s
sref = s0[0::T]
s45_orbital0 = s_orbital0[0::T]
s45_orbital = s_orbital[0::T]

# Add noise
# sref += np.random.normal(0,0.1, n)
#s45_orbital += 200*np.random.normal(0,0.1, s45_orbital.size)

# Sampled timeline and sampled singal length
t = t0[0::T]
n = t.size


# 1 hr in # of time step in s is 80
hour = 80
day = 24 * hour
gap_duration = int(1*hour)
start = 7*hour
# Gaps: They are spaced by 24 hours. The 1st starts 8 hours after the begining.
gap_times = np.arange(5) * day + start

# sgaps = gap_signal(sref, gap_duration, gap_times)
# sgaps_orbital = gap_signal(s45_orbital, gap_duration, gap_times)

# Cut frequencies (mHz)
fcut1 = 2.222 # (3.75 min)
fcut2 = 4.444 # (7.5 min)

# Set the times over which we want the power spectrum
# Interval in # of frames: 30 in = 40 frames
interval = 80


# mywrap = wrapper(get_pspectra_sum, s, sgaps, interval, fcut1, fcut2)
# print(timeit(mywrap, number=1000))

psum = get_pspectra_mean(sref, T, interval, fcut1, fcut2, get_pspectra)
psum_orbital = get_pspectra_mean(s45_orbital, T, interval, fcut1, fcut2, get_pspectra)

# Without modulation
psum_orbital_mirror0 = get_pspectra_mean(s45_orbital0, T, interval, fcut1, fcut2, get_pspectra_mirror)
# With modulation
psum_orbital_mirror = get_pspectra_mean(s45_orbital, T, interval, fcut1, fcut2, get_pspectra_mirror)

### Display

tslice = slice(gap_times[0]-interval, gap_times[0] + 2*gap_duration)

# Build a timeline with interval = 30-min ~ 40-frames for the power spectra plots
# Start times (in units of frames)
tstarts = np.arange(0, sref.size, interval)
# time line in seconds
timeline = (tstarts + interval/2)*T


unit_short = 1 / 60
unit_long = 1 / 3600



plt.figure(1, figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(t * unit_long, s45_orbital0, 'b-', label="solar rotation + 24-hour orbital effect")
plt.plot(t * unit_long, s45_orbital, 'r-', label=" with weak modulation of 9-days period", alpha=0.6)
plt.xlabel('time [hr]')
plt.ylabel('Signal intensity')
#plt.axis([5, 15, -sgaps_orbital.max(), sgaps_orbital.max()])
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
#plt.plot(timeline * unit_long, psum/psum.mean(), 'b-', label = 'Reference')
#plt.plot(timeline * unit_long, psum_orbital/psum_orbital.mean(), 'r-', label="psd Without windowing or mirorring")

plt.plot(timeline * unit_long, psum_orbital_mirror0/psum_orbital_mirror0.mean(), 'b-', label="psd with orbital and solar rotation")
plt.plot(timeline * unit_long, psum_orbital_mirror/psum_orbital_mirror.mean(), 'r-', label="psd with 9 days modulation")

plt.xlabel('time [hr]')
plt.ylabel('Integrated Power (normalized unit)')
plt.axis([0, 24*5, 0, 1.4])
plt.legend()
plt.grid()
plt.show()




# # Filtered signal
# s2 = np.real(np.fft.ifft(np.fft.ifftshift(sf_filtered)))
# sgaps2 = np.real(np.fft.ifft(np.fft.ifftshift(sfgaps_filtered)))
#
#
# # cut frequencies (for both positive (1) and negative freqs (2))
# cut_idx1, cut_idx2, _ = get_freq_idx(fcut1, fcut2)
#
# # Axis start time for plot
# t1 = 3600 * 7
# t2 = t1 + 3600 * 4
#
#
# plt.close()
# plt.figure(2, figsize=(16,10))
# plt.subplot(3,1,1)
# #plt.plot(t0, s0, 'b-')
# plt.plot(t, s, 'b-')
# plt.plot(t, sgaps, 'r--')
# plt.axis([t1, t2, -1, 1])
# plt.xlabel('Time (s)')
# plt.ylabel('Signal intensity')
#
#
#
#
# plt.subplot(3,1,2)
# plt.plot(freqs, sFgaps, 'r--')
# plt.plot(freqs, sF, 'b-')
#
# plt.yscale('log')
# plt.axis([0, 6, 0, sF.max()])
# plt.xlabel('Freq [mHz]')
# plt.ylabel('Power spectrum')
#
# plt.axvline(x=freqs[fcut_idx1[0]], ls='--', color='black', lw=1)
# plt.axvline(x=freqs[fcut_idx1[-1]], ls='--', color='black', lw=1)
# plt.axvline(x=freqs[fcut_idx2[0]], ls='--', color='black', lw=1)
# plt.axvline(x=freqs[fcut_idx2[-1]], ls='--', color='black', lw=1)
#
# plt.subplot(3,1,3)
# plt.plot(t, s2, 'b-')
# plt.plot(t, sgaps2, 'r--')
# plt.axis([t1, t2, -1, 1])
# plt.xlabel('Time (s)')
# plt.ylabel('Filtered Signal intensity')
#
#
#
# plt.figure(3, figsize=(16,10))
# plt.plot(s2, 'b-')
# plt.plot(sgaps2, 'r-')
# plt.axis([0, 3500, -1, 1])
# plt.xlabel('# of frames')
# plt.ylabel('Signal')