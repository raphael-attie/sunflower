import os
import numpy as np
import fitsio
import matplotlib.pyplot as plt
from filters import han2d_lpf
from filters import han2d_hpf
from filters import han2d_bandpass
from filters import ffilter_image
from filters import azimuthal_average
from fitstools import fitsread
from fitstools import fitsheader
from skimage.measure import compare_ssim as ssim
from filters import matrix_ffilter_image

plt.ioff()

outputdir   = '/Users/rattie/Data/SDO/HMI/EARs/AR11130_2010_11_27/filtered_examples/'
file        = '/Users/rattie/Data/SDO/HMI/EARs/AR11130_2010_11_27/mtrack_20101126_170034_TAI_20101127_170034_TAI_LambertCylindrical_continuum.fits'
# Get the header
h       = fitsheader(file)
# Get the 1st image
image   = fitsread(file, n=1)


# Fourier transform, shifted
fimage = np.fft.fftshift(np.fft.fftn(image))
# power spectrum
pimage = np.abs(fimage) **2
# Azimuthal average - Radial profile
radial_prof = azimuthal_average(pimage)
# Build the Fourier axis and corresponding spatial scales.
N           = 512
df          = 1/N
n_freqs     = len(radial_prof)
freqs       = np.arange(n_freqs) * df
scales      = 1 / freqs

# Large scale limit. Larger periodic features are suppressed
large_scales = list(range(2, 50))[::-1]
large_scales.insert(0, 0)
# Convert to numpy array.
# large_scales = np.array(large_scales)
# Small scale limit. Smaller periodic features are suppressed
small_scales    = np.arange(0, 3, 0.1).tolist()

### Band-pass filter
# Build filter
bpf_images, filters_bpf, sigmas, ssims = matrix_ffilter_image(image, small_scales, large_scales)

filters_lpf   = [ffilters for ffilters in filters_bpf[0]]
filters_hpf   = [ffilters[0] for ffilters in filters_bpf]

### Look at 1D profiles of all filters
filter_bpf_1d   = [[azimuthal_average(ffilter) for ffilter in ffilters] for ffilters in filters_bpf]
filter_lpf_1d   = [azimuthal_average(ffilter) for ffilter in filters_lpf]
filter_hpf_1d   = [azimuthal_average(ffilter) for ffilter in filters_hpf]

# Show a few samples of filtered data
lscales = [0, 20, 15, 10, 5, 4, 3, 2]
sscales = [0, 1, 1.5, 2, 2.5]
linds   = [large_scales.index(scale) for scale in lscales]
sinds   = [small_scales.index(scale) for scale in sscales]

# linds = [np.where(large_scales == scale)[0].astype(int) for scale in lscales]
# sinds = [np.where(small_scales == scale)[0].astype(int) for scale in sscales]

nlarge = len(lscales)
nsmall = len(sscales)

# Show some samples of filtered images
plt.figure(1, figsize=(19, 11))
plt.clf()
span = 50
x0 = 100
y0 = 100
for j in range(nsmall):
    for i in range(nlarge):
        plt.subplot(nsmall, nlarge, nlarge * j + i + 1)
        plt.imshow(bpf_images[linds[i]][sinds[j]][y0:y0+span, x0:x0+span], cmap='gray', origin='lower')
        plt.title("bpf =[%0.1f , %0.1f]px" % (sscales[j], lscales[i]))
        plt.text(1,1, "$\sigma$ = %d" % (sigmas[linds[i], sinds[j]]), color='yellow', fontweight='normal', fontsize=12)
        plt.text(1, 6, "ssim = %0.3f" % (ssims[linds[i], sinds[j]]), color='yellow', fontweight='normal', fontsize=12)
        plt.gca().set_xticks(list(range(0, span, 4)))
        plt.gca().set_yticks(list(range(0, span, 4)))
        plt.gca().tick_params(labelbottom='off')
        plt.gca().tick_params(labelleft='off')
plt.tight_layout()
fname = os.path.join(outputdir, 'filtered_samples_120dpi.pdf')
plt.savefig(fname, dpi=120, bbox_inches='tight')

# Show SSIMs
fig = plt.figure(2, figsize=(16,9))

ax1 = fig.add_subplot(111)
ax1.imshow(ssims.transpose(), cmap='magma')
plt.xlabel('Array columns')
plt.ylabel('Array rows')
plt.title('SSIM')

xticks = list(range(0, len(large_scales), 2))
yticks = list(range(0, len(small_scales), 2))

xlabels = [str(large_scales[xtick]) for xtick in xticks]
ylabels = ["%0.1f"%(small_scales[ytick]) for ytick in yticks]

ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.set_xticklabels(xlabels)
ax1.set_yticklabels(ylabels)
ax1.set_xlabel('Spatial scales of high-pass cut-off (px)')
ax1.set_ylabel('Spatial scales of low-pass cut-off (px)')


fname = os.path.join(outputdir, 'ssims_matrix_120dpi.pdf')
plt.savefig(fname, dpi=120, bbox_inches='tight')

# Print full-sized image
plt.figure(3, figsize=(18, 10))

n_hpf = len(large_scales)
n_lpf = len(small_scales)

for i in range(n_lpf):
    for j in range(n_hpf):
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(bpf_images[i][j], cmap='gray', origin='lower')
        plt.title("bpf [%0.1f ; %0.1f] px" % (large_scales[j], small_scales[i]))
        plt.subplot(1, 2, 2)
        plt.plot(scales, filter_bpf_1d[i][j], 'k-', label='band-pass')
        plt.plot(scales, filter_hpf_1d[j], 'g-', label='high-pass < %0.1f px' %(large_scales[j]))
        plt.plot(scales, filter_lpf_1d[i], 'b-', label='low-pass > %0.1f px' %(small_scales[i]))
        plt.xlim([0, 50])
        plt.legend(loc = 'center right')
        plt.xlabel('Spatial scales (px)')
        plt.ylabel('Fourier mask (radial profile)')
        fname = os.path.join(outputdir, 'filtered_bpf_%0.1f_%0.1f_px.jpeg' % (small_scales[i], large_scales[j]))
        plt.savefig(fname, dpi=90, bbox_inches='tight')


