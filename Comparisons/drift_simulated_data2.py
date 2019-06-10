import os, glob
import numpy as np
import fitstools
import balltracking.balltrack as blt


def filter_function(image):
    fimage = blt.filter_image(image)
    return fimage


if __name__ == '__main__':
    # input data, list of files
    # glob.glob does not order numbered files by defaultl, the order is as appeared in the file system.
    datafiles = sorted(glob.glob('/Users/rattie/Data/Ben/SteinSDO/SDO_int*.fits'))
    ### Ball parameters
    # Use 80 frames (1 hr)
    nframes = 80
    # Ball radius
    rs = 2
    # depth factor
    dp = 0.3  # 0.2
    # Multiplier to the standard deviation.
    sigma_factor = 1  # 1#2
    # Select only a subset of nframes files
    selected_files = datafiles[0:nframes]
    # Load the nt images
    images = fitstools.fitsread(selected_files)
    # Must make even dimensions for the fast fourier transform
    images2 = np.zeros([264, 264, images.shape[2]])
    images2[0:263, 0:263, :] = images.copy()
    images2[263, :] = images.mean()
    images2[:, 263] = images.mean()



    dv = 0.02
    vx_rates = np.arange(-0.2, 0.2+dv, dv)
    vx_rates[int(len(vx_rates)/2)] = 0
    ndrifts = len(vx_rates)
    # The drift can optionnally be on both direction, not just on the x-axis
    drift_rates = np.stack((vx_rates, np.zeros(ndrifts)), axis=1).tolist()


    # output directory for the drifting images
    outputdir = '/Users/rattie/Data/Ben/SteinSDO/calibration2/unfiltered'
    subdirs = [os.path.join(outputdir, 'drift_{:02d}'.format(i)) for i in range(len(drift_rates))]
    cal = blt.Calibrator(images2, drift_rates, nframes, rs, dp, sigma_factor, outputdir,
                         output_prep_data=False, use_existing=False, tracking=False, normalization=False,
                         filter_function=None, subdirs=subdirs,
                         nthreads=5)

    cal.drift_all_rates()

    # output directory for the drifting images
    outputdir = '/Users/rattie/Data/Ben/SteinSDO/calibration2/filtered'
    subdirs_filtered = [os.path.join(outputdir, 'drift_{:02d}'.format(i)) for i in range(len(drift_rates))]
    cal_filtered = blt.Calibrator(images2, drift_rates, nframes, rs, dp, sigma_factor, outputdir,
                         output_prep_data=False, use_existing=False, tracking=False, normalization=False,
                         filter_function=filter_function, subdirs=subdirs_filtered,
                         nthreads=5)

    cal_filtered.drift_all_rates()
