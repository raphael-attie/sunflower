import os, glob
import numpy as np
from scipy.io import readsav
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
import fitstools
import fitsio
import balltracking.balltrack as blt


def balltrack_calibration(datafiles, fwhm, intsteps, fov_slices):

    # Select only a subset of nframes files
    selected_files = datafiles[0:nframes]
    # Load the nt images
    images = fitstools.fitsread(selected_files)
    # Must make even dimensions for the fast fourier transform
    images2 = np.zeros([264, 264, images.shape[2]])
    images2[0:263, 0:263, :] = images.copy()
    images2[263, :] = images.mean()
    images2[:, 263] = images.mean()

    if reprocess_bt:
        cal = blt.Calibrator(images2, drift_rates, nframes, rs, dp, sigma_factor, outputdir,
                             intsteps=intsteps,
                             output_prep_data=False, use_existing=use_existing,
                             nthreads=5)

        ballpos_top_list, ballpos_bottom_list = cal.balltrack_all_rates()
    else:
        print('Load existing tracked data at all rates')
        ballpos_top_list = np.load(os.path.join(outputdir, 'ballpos_top_list.npy'))
        ballpos_bottom_list = np.load(os.path.join(outputdir, 'ballpos_bottom_list.npy'))

    trange = [0, nframes]
    xrates = np.array(drift_rates)[:, 0]
    a_top, vxfit_top, vxmeans_top, residuals_top = blt.fit_calibration(ballpos_top_list, xrates, trange, fwhm,
                                                        images.shape[0:2], fov_slices,
                                                        return_flow_maps=False)
    a_bottom, vxfit_bottom, vxmeans_bottom, residuals_bottom = blt.fit_calibration(ballpos_bottom_list, xrates, trange, fwhm,
                                                                 images.shape[0:2], fov_slices,
                                                                 return_flow_maps=False)

    return a_top, vxfit_top, vxmeans_top, residuals_top, a_bottom, vxfit_bottom, vxmeans_bottom, residuals_bottom


def average_calibration(vxfit_top, vxfit_bottom):
    # Fit the averaged calibrated balltrack velocity
    vxmeans_bt = (vxfit_top + vxfit_bottom)/2
    # #vxmeans_bt -= vxmeans_bt[4]
    p = np.polyfit(vx_rates, vxmeans_bt, 1)
    a_avg = 1 / p[0]
    vxfit_avg = a_avg * (vxmeans_bt - p[1])
    # # Calculate residuals
    bt_residuals = np.abs(vxmeans_bt - vx_rates)
    #bt_residuals_fit = np.abs(vxfit_avg - vx_rates)
    print('a_avg = {:0.2f}, p[1] = {:0.2f}'.format(a_avg, p[1]))
    return vxmeans_bt, a_avg, vxfit_avg, bt_residuals



def dv_analysis(dvx_files, vx_ratesu):
    vxmeans = []
    for i in range(len(dvx_files)):
        print('Fits file dvx_files[{:d}]: {:s}'.format(i, dvx_files[i]))
        vx = fitsio.read(dvx_files[i])
        vxmeans.append(vx[fov_slices[1], fov_slices[0]].mean())
    vxmeans = np.array(vxmeans)# - vxmeans[4]
    ## Calibration parameters: y_measured = p0*x_true + p1 => true = (measured - p1)*1/p0
    p = np.polyfit(vx_ratesu, vxmeans, 1)
    a_dv = 1 / p[0]
    vxfits = a_dv * (vxmeans - p[1])
    # Calculate residuals
    dv_residuals0 = np.abs(vxmeans - vx_ratesu)
    dv_residuals = np.abs(vxfits - vx_ratesu)

    return vxmeans, a_dv, vxfits, dv_residuals0, dv_residuals




if __name__ == '__main__':

    nframes = 80

    true_vel_dir = '/Users/rattie/Data/Ben/SteinSDO/'
    true_vx_files = sorted(glob.glob(os.path.join(true_vel_dir, 'SDO_vx*.fits')))[0:nframes]
    true_vx = np.array([fitsio.read(vxfile) for vxfile in true_vx_files])

    # DeepVel data directory
    dv_dir = '/Users/rattie/Data/Ben/SteinSDO/DeepVel_Drift_data_output'

    # Set if we balltrack again or use previous results
    reprocess_bt = False
    # Set if we use existing drifted images
    use_existing = True


    dvdirs = [os.path.join(dv_dir, 'drift_{:d}'.format(i)) for i in range(9)]
    dvx_files = [os.path.join(dvdir, 'DeepVel_vx1_drift_000-079.fits') for dvdir in dvdirs]

    # Here we need to exclude the region that got circularly shifted by the Fourier phase shit ~ 23 pixels in both horizontal directions (left-right)
    # Exclude also FWHM pixels in either side of the vertical axis.

    unit = 368000 / 60
    unit_str = '[m/s]'

    times = [9, 6, 3] # in minutes <=> nb of frames. Divide by it to get back to px/frame.
    fwhms = [15, 7]

    npts = 9
    imsize = 263 # actual size is 264 but original size was 263 then padded to 264 to make it even for the Fourier transform
    vx_rates = np.linspace(-0.2, 0.2, npts)
    drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()
    # Select a subfield excluding edge effects
    trim = int(vx_rates.max() * nframes + fwhms[0] + 2)
    fov_slices = np.s_[trim:imsize - trim, trim:imsize-trim]

    # True mean of Vx
    true_vx_tavg = true_vx.mean(axis=0)
    true_vx_mean = true_vx_tavg[fov_slices].mean()

    vx_ratesu = vx_rates * unit
    bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]


    # Loop through the matrix data
    vxmeans, a_dv, vxfits, dv_residuals0, dv_residuals = dv_analysis(dvx_files, vx_ratesu)
    vxfits_arr =  np.array(vxfits)
    dv_residuals0_arr = np.array(dv_residuals0)
    dv_residuals_arr = np.array(dv_residuals)

    # convert units of lct
    vxmeansu = vxmeans# * unit
    vxfitsu = vxfits_arr# * unit
    dv_residuals0u = dv_residuals0_arr# * unit
    dv_residualsu = dv_residuals_arr# * unit



    ###########################################################
    ################## Balltracking ###########################
    ###########################################################

    # input data, list of files
    # glob.glob does not order numbered files by defaultl, the order is as appeared in the file system.
    datafiles = sorted(glob.glob('/Users/rattie/Data/Ben/SteinSDO/SDO_int*.fits'))
    # output directory for the drifting images
    outputdir = '/Users/rattie/Data/Ben/SteinSDO/calibration/'
    plotdir = os.path.join('/Users/rattie/Data/Ben/SteinSDO/plots/')
    ### Ball parameters
    # Use 80 frames (1 hr)
    nframes = 80
    # Ball radius
    rs = 2
    # depth factor
    dp = 0.3#0.2
    # Multiplier to the standard deviation.
    sigma_factor = 1#1#2
    # Nb of integration steps between images
    intsteps = 5
    # Plot figure file name suffix
    fig_suffix = 'intsteps_{:d}_dp_{:0.2f}_sigma_factor_{:0.2f}'.format(intsteps, dp, sigma_factor)
    dpi = 300
    # Select a subfield excluding edge effects
    fov_slices_bt = [fov_slices, ]
    ##########################################
    ######  Smoothing at FWHM = 7 px #########
    fwhm = 7
    a_top1, vxfit_top1, vxmeans_top1, res_top1, a_bot1, vxfit_bot1, vxmeans_bot1, res_bot1 = balltrack_calibration(datafiles, fwhm, intsteps, fov_slices_bt)
    vxmeans_bt1, a_avg1, vxfit_avg1, bt_residuals1 = average_calibration(vxfit_top1, vxfit_bot1)


    ##########################################
    ######  Smoothing at FWHM = 15 px #########
    fwhm = 15
    a_top2, vxfit_top2, vxmeans_top2, res_top2, a_bot2, vxfit_bot2, vxmeans_bot2, res_bot2 = balltrack_calibration(datafiles, fwhm, intsteps, fov_slices_bt)
    vxmeans_bt2, a_avg2, vxfit_avg2, bt_residuals2 = average_calibration(vxfit_top2, vxfit_bot2)
    ################################################
    ################ Plot results ##################
    ################################################
 

    label = 'DeepVel'
    # Default plot params
    fs = 16
    plt.rcParams.update({'font.size': fs})
    markers = 'd'
    colors = 'red'
    legend_loc = 'lower right'
    # Widths of bar plots
    widths = [0.02, 0.015, 0.01]


    #############################################################################################
    ######## Unfiltered. Indices are  [0, 1, 2, 3, 4, 5] ########################################
    #############################################################################################
    # List of indices to get LCT results from unfiltered
    # Max velocity in the axis of the linear fits.
    maxv = 0.3 * unit

    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 8))
    axs.set_title('FWHM = 7 px')
    # Consider only the unfiltered ones (the filtered ones do not work). range from 0 to 5 incl.
    axs.plot(vxmeansu, vx_ratesu, marker=markers, color=colors, ls='none', label=label)
    axs.plot(vxmeansu, vxfitsu, color=colors, label=r'$\alpha_{:s}$ ={:0.2f}'.format('D', a_dv))

    axs.plot(vxmeans_bt1 * unit, vx_ratesu, marker='o', markerfacecolor='none', ls='none', color='green', label='Balltracking data')
    axs.plot(vxfit_avg1 * unit, vx_ratesu, ls='-.', color='green', label=r'$\alpha_B = {:0.2f}$'.format(a_avg1))

    axs.axis([-maxv, maxv, -maxv, maxv])
    axs.set_xlabel('Measured velocity {:s}'.format(unit_str), fontsize=fs)
    axs.set_ylabel('True velocity {:s}'.format(unit_str), fontsize=fs)
    axs.legend(loc=legend_loc)
    axs.grid(True)
    axs.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(dv_dir, 'DeepVel_linear_fit_{:s}.png'.format(fig_suffix)), dpi=dpi)


    ##################################################################################################################
    ### Residuals uncorrected
    ##################################################################################################################
    max_resid_uncorrected = 1000#0.1 * unit

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    axs.set_title('FWHM = 7 px')
    axs.set_xlabel('True velocity {:s}'.format(unit_str))
    axs.set_ylabel('Residuals {:s}'.format(unit_str))
    bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
    colors = ['red']

    axs.bar(vx_ratesu, dv_residuals0u,width=widths[0] * unit, color=colors[0], tick_label=bar_labels, label='DeepVel')
    axs.bar(vx_ratesu, bt_residuals1 * unit, width=widths[0]*0.7 * unit, color='green', tick_label=None, label='balltracking')
    axs.set_ylim([0, max_resid_uncorrected])
    axs.legend()
    axs.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(dv_dir, 'DeepVel_residuals_{:s}.png'.format(fig_suffix)), dpi=dpi)

    ######################################
    ##### Residuals with linear correction
    ######################################
    max_resid = 50 #0.01 * unit / 2

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    axs.set_title('FWHM = 7 px (linearly corrected)')
    axs.set_xlabel('True velocity {:s}'.format(unit_str))
    axs.set_ylabel('Residuals {:s}'.format(unit_str))
    bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
    colors = ['black', 'red', 'blue']

    axs.bar(vx_ratesu, dv_residualsu,width=widths[0] * unit, color=colors[0], tick_label=bar_labels, label='DeepVel')
    axs.bar(vx_ratesu, bt_residuals1 * unit, width=widths[0] * 0.7 * unit, color='green', tick_label=bar_labels,label='balltracking')

    axs.set_ylim([0, max_resid])
    axs.legend()
    axs.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(dv_dir, 'DeepVel_residuals_corrected_{:s}.png'.format(fig_suffix)), dpi=dpi)





