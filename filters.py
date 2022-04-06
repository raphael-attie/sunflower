import numpy as np
from skimage.metrics import structural_similarity as ssim

def han1d_hpf(n, T, rc):
    # 1D hanning filter
    # T: Sampling interval
    # rc: Small scale limit (time or spatial domain)
    # n: length of array


    if rc == 0:
        hwindow = np.ones([n, n])
        return hwindow

    # Sampling frequency
    fs = 1/T
    # Spacing between frequency points
    df = fs / n
    # Freqs vector
    freqs = np.fft.fftfreq(n, T)
    # Cut frequency
    fc = 1/rc
    # Frequency index
    fc = np.round(n * fc)

    x = np.arange(n)
    # symetric grid of radial distances
    f = np.sqrt((xgrid - (n / 2 - 0.5)) ** 2 + (ygrid - (n / 2 - 0.5)) ** 2)
    # Hanning window that decreases as r decreases to zero.
    hwindow = 0.5 - 0.5 * np.cos(np.pi * f / (2 * fc))
    # Keep all high frequencies
    mask_f = f > 2 * fc
    hwindow[mask_f] = 1

    return hwindow

def han2d_hpf(n, rc):
    # 2D hanning filter, assumes a square image size
    # rc: Small spatial scale limit (px). Spatial periods > rc will be suppressed.
    # n: length of array

    if rc == 0:
        hwindow = np.ones([n, n])
        return hwindow

    freq_lowcut = 1/rc
    # Frequency index
    fc = np.round(n * freq_lowcut)

    xgrid, ygrid = np.meshgrid(np.arange(n), np.arange(n))
    # symetric grid of radial distances
    f = np.sqrt((xgrid - (n / 2 - 0.5)) ** 2 + (ygrid - (n / 2 - 0.5)) ** 2)
    # Hanning window that decreases as r decreases to zero.
    hwindow = 0.5 - 0.5 * np.cos(np.pi * f / (2 * fc))
    # Keep all high frequencies
    mask_f = f > 2 * fc
    hwindow[mask_f] = 1

    return hwindow

def han2d_lpf(n, rc):
    # 2D hanning filter, assumes a square image size
    # rc: Large spatial scale limit (px). Spatial periods < rc will be suppressed.
    # n: length of array
    if rc == 0:
        hwindow = np.ones([n, n])
        return hwindow

    freq_highcut = 1 / rc
    # Frequency index
    fc = np.round(n * freq_highcut)

    xgrid, ygrid = np.meshgrid(np.arange(n), np.arange(n))
    # symetric grid of radial distances
    f = np.sqrt((xgrid - (n / 2 - 0.5)) ** 2 + (ygrid - (n / 2 - 0.5)) ** 2)
    hwindow = 0.5 + 0.5 * np.cos(np.pi * f / (2 * fc))
    # Cut high frequencies
    mask_f = f > 2 * fc
    hwindow[mask_f] = 0

    return hwindow


def han2d_bandpass(n, small_scale, large_scale):
    # 2D hanning filter, assumes a square image size

    hwindow1 = han2d_lpf(n, small_scale)
    hwindow2 = han2d_hpf(n, large_scale)

    hwindow = hwindow1 * hwindow2

    return hwindow


def ffilter_image(image, fourier_filter, dx=0, dy=0):

    # 2D hanning filter, assumes a square image size
    # filter: filter applied to the fourier transform

    # Get the fourier transform
    fimage = np.fft.fftshift(np.fft.fftn(image))
    # Apply 2D filter to fourier transform
    windowed_fimage = fimage * fourier_filter

    if dx != 0 or dy != 0:

        windowed_fimage = phase_shift(windowed_fimage)

    filtered_image = np.real(np.fft.ifftn(np.fft.ifftshift(windowed_fimage)))

    return filtered_image.copy(order='C')

def phase_shift(fimage, dx, dy):

    dims = fimage.shape
    x, y = np.meshgrid(np.arange(-dims[1] / 2, dims[1] / 2), np.arange(-dims[0] / 2, dims[0] / 2))

    kx = -1j * 2 * np.pi * x / dims[1]
    ky = -1j * 2 * np.pi * y / dims[0]

    shifted_fimage = fimage * np.exp(-(kx * dx + ky * dy))

    return shifted_fimage

def translate_by_phase_shift(image, dx, dy):

    if dx==0 and dy==0:
        return image
    # Get the fourier transform
    fimage = np.fft.fftshift(np.fft.fftn(image))
    # Phase shift
    fimage = phase_shift(fimage, -dx, -dy)
    # returns inverse transform -> translated image
    return np.real(np.fft.ifftn(np.fft.ifftshift(fimage)))



def matrix_ffilter_image(image, small_scales, large_scales):

    # image size
    n = image.shape[0]

    # Build filter. First list dimension is over large_scales -> hpf
    ffilter_bpf = [[han2d_bandpass(n, small_scale, large_scale) for small_scale in small_scales] for large_scale in large_scales]
    # Apply filter
    bpf_images = [[ffilter_image(image, ffilter) for ffilter in ffilters] for ffilters in ffilter_bpf]

    span = int(n / 2)
    n_hpf = len(large_scales)
    n_lpf = len(small_scales)

    ssims   = np.zeros([n_hpf, n_lpf])
    sigmas  = np.zeros([n_hpf, n_lpf])

    for j in range(n_lpf):
        for i in range(n_hpf):
            s1 = bpf_images[i][j][0:span, :]
            s2 = bpf_images[i][j][span:, :]
            ssims[i, j] = ssim(s1, s2, data_range=bpf_images[i][j].max() - bpf_images[i][j].min())
            sigmas[i, j] = bpf_images[i][j].std()

    #ffilter_bpf = np.array(ffilter_bpf)

    return bpf_images, ffilter_bpf, sigmas, ssims




def azimuthal_average(image, center=None):
    """
    from http://www.astrobetter.com/wiki/tiki-index.php?page=python_radial_profiles
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof
