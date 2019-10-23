import os, glob
import numpy as np
import fitstools
import balltracking.balltrack as blt


def filter_function(image):
    fimage = blt.filter_image(image)
    return fimage


if __name__ == '__main__':
    # input data, list of files
    filepath = '/Users/rattie/Data/SDO/HMI/continuum/Lat_0/mtrack_20110627_200034_TAI_20110628_000033_TAI_Postel_060.4_00.0_continuum.fits'
    ### Ball parameters
    # Use 80 frames (1 hr)
    nframes = 80
    # Ball radius
    rs = 2
    # depth factor
    dp = 0.3  # 0.2
    # Multiplier to the standard deviation.
    sigma_factor = 1  # 1#2
    # Load the nt images
    images = fitstools.fitsread(filepath, tslice=slice(0, nframes))

    dv = 0.04
    vx_rates = np.arange(-0.2, 0.21, dv)
    vx_rates[int(len(vx_rates)/2)] = 0
    ndrifts = len(vx_rates)
    # The drift can optionnally be on both direction, not just on the x-axis
    drift_rates = np.stack((vx_rates, np.zeros(ndrifts)), axis=1).tolist()


    # output directory for the drifting images
    outputdir = '/Users/rattie/Data/sanity_check/hmi_series'
    subdirs = [os.path.join(outputdir, 'drift_{:02d}'.format(i)) for i in range(len(drift_rates))]
    cal = blt.Calibrator(images, drift_rates, nframes, rs, dp, sigma_factor, outputdir,
                         output_prep_data=False, use_existing=False, tracking=False, normalization=False,
                         filter_function=None, subdirs=subdirs,
                         nthreads=5)

    cal.drift_all_rates()

    # # output directory for the drifting images
    # outputdir = '/Users/rattie/Data/Ben/SteinSDO/calibration2/filtered'
    # subdirs_filtered = [os.path.join(outputdir, 'drift_{:02d}'.format(i)) for i in range(len(drift_rates))]
    # cal_filtered = blt.Calibrator(images2, drift_rates, nframes, rs, dp, sigma_factor, outputdir,
    #                      output_prep_data=False, use_existing=False, tracking=False, normalization=False,
    #                      filter_function=filter_function, subdirs=subdirs_filtered,
    #                      nthreads=5)
    #
    # cal_filtered.drift_all_rates()
