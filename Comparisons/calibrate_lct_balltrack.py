import os, glob
import numpy as np
from scipy.io import readsav
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
import fitstools
import balltracking.balltrack as blt

DATADIR = '/Users/rattie/Data/karin/matrix_data/'


def lct_analysis(lctfiles):
    vxmeans = []
    for i in range(len(lctfiles)):
        idl_dict = readsav(lctfiles[i])
        vx = idl_dict['vx'].mean(axis=0)  # * 60/368
        time_factor = 1
        if '9min' in lctfiles[i]:
            time_factor = 9
        elif '6min' in lctfiles[i]:
            time_factor = 6
        elif '3min' in lctfiles[i]:
            time_factor = 3

        vx *= 1 / time_factor
        vxmeans.append(vx[fov_slices[1], fov_slices[0]].mean())
    vxmeans = np.array(vxmeans)# - vxmeans[4]
    ## Calibration parameters: y_measured = p0*x_true + p1 => true = (measured - p1)*1/p0
    p = np.polyfit(vx_rates, vxmeans, 1)
    a_lct = 1 / p[0]
    vxfit = a_lct * (vxmeans - p[1])
    # Calculate residuals
    lct_residuals0 = np.abs(vxmeans - vx_rates)
    lct_residuals = np.abs(vxfit - vx_rates)

    return vxmeans, a_lct, vxfit, lct_residuals0, lct_residuals


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



if __name__ == '__main__':

    reprocess_lct = False
    # Set if we balltrack again or use previous results
    reprocess_bt = False
    # Set if we use existing drifted images
    use_existing = True


    lctdirs = [os.path.join(DATADIR, 'test{:d}'.format(i)) for i in range(20,32)]
    lctfiles_list = [sorted(glob.glob(os.path.join(lctdir, 'flct-test*.save'))) for lctdir in lctdirs]
    # Here we need to exclude the region that got circularly shifted by the Fourier phase shit ~ 23 pixels in both horizontal directions (left-right)
    # Exclude also FWHM pixels in either side of the vertical axis.

    unit = 368000 / 60
    unit_str = '[m/s]'

    times = [9, 6, 3] # in minutes <=> nb of frames. Divide by it to get back to px/frame.
    fwhms = [15, 7]
    nframes = 80
    npts = 9
    imsize = 263 # actual size is 264 but original size was 263 then padded to 264 to make it even for the Fourier transform
    vx_rates = np.linspace(-0.2, 0.2, npts)
    drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()
    # Select a subfield excluding edge effects
    trim = int(vx_rates.max() * nframes + fwhms[0] + 2)
    fov_slices = np.s_[trim:imsize - trim, trim:imsize-trim]

    vx_ratesu = vx_rates * unit
    bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]

    if reprocess_lct:
        # Loop through the matrix data
        vxmeans_list, a_lcts, vxfits, lct_residuals0, lct_residuals = zip(*[lct_analysis(lctfiles) for lctfiles in lctfiles_list])
        vxmeans_arr = np.array(vxmeans_list)
        vxfits_arr =  np.array(vxfits)
        lct_residuals0_arr = np.array(lct_residuals0)
        lct_residuals_arr = np.array(lct_residuals)
        np.savez(os.path.join(DATADIR, 'lct_results.npz'), vxmeans_arr=vxmeans_arr, a_lcts=a_lcts, vxfits_arr=vxfits_arr,lct_residuals0_arr=lct_residuals0_arr, lct_residuals_arr=lct_residuals_arr)
    else:
        npzfile = np.load(os.path.join(DATADIR, 'lct_results.npz'))
        vxmeans_arr, a_lcts, vxfits_arr, lct_residuals0_arr, lct_residuals_arr = npzfile['vxmeans_arr'], npzfile['a_lcts'], npzfile['vxfits_arr'], npzfile['lct_residuals0_arr'], npzfile['lct_residuals_arr']


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
    plt.close('all')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(vx_ratesu, res_top1 * unit, width=0.02 * unit, color='orange', tick_label=bar_labels, label='top')
    plt.bar(vx_ratesu, res_bot1 * unit, width=0.015 * unit, color='blue', tick_label=bar_labels, label='bottom')
    plt.bar(vx_ratesu, bt_residuals1 * unit, width=0.01 * unit, color='green', tick_label=bar_labels, label='average')
    plt.ylim([0, 6.5])

    plt.title('fwhm: 7px - int. steps:{:d} dp:{:0.2f} sigma factor:{:0.2f}'.format(intsteps, dp, sigma_factor))
    plt.grid(True, axis='both')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(vx_ratesu, res_top2 * unit, width=0.02 * unit, color='orange', tick_label=bar_labels, label='top')
    plt.bar(vx_ratesu, res_bot2 * unit, width=0.015 * unit, color='blue', tick_label=bar_labels, label='bottom')
    plt.bar(vx_ratesu, bt_residuals2 * unit, width=0.01 * unit, color='green', tick_label=bar_labels, label='average')
    plt.ylim([0, 6.5])

    plt.title('fwhm: 15px')
    plt.grid(True, axis='both')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(DATADIR, 'bt_res_{:s}.png'.format(fig_suffix)), dpi=dpi)


    # convert units of lct
    vxmeansu = vxmeans_arr * unit
    vxfitsu = vxfits_arr * unit
    lct_residuals0u = lct_residuals0_arr * unit
    lct_residualsu = lct_residuals_arr * unit


    labels = ['dt = 9 min',
              'dt = 6 min',
              'dt = 3 min']

    # Default plot params
    fs = 16
    plt.rcParams.update({'font.size': fs})
    markers = ['d', '+', 'x']
    colors = ['black', 'red', 'blue']
    legend_loc = 'lower right'
    # Widths of bar plots
    widths = [0.02, 0.015, 0.01]




    #############################################################################################
    ######## Unfiltered. Indices are  [0, 1, 2, 3, 4, 5] ########################################
    #############################################################################################
    # List of indices to get LCT results from unfiltered
    plotseq = [0, 1, 2, 3, 4, 5]
    data_type = 'unfiltered'
    # Max velocity in the axis of the linear fits.
    maxv = 0.3 * unit

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8))
    axs[0].set_title('FWHM = 7 px ({:s})'.format(data_type))
    axs[1].set_title('FWHM = 15 px ({:s})'.format(data_type))

    # Consider only the unfiltered ones (the filtered ones do not work). range from 0 to 5 incl.
    for i in range(6):
        k = 1 - int(i/3) # LCT 7 px must be on the 0th axis (left-hand side)
        axs[k].plot(vxmeansu[plotseq[i], :], vx_ratesu, marker=markers[i % 3], color=colors[i % 3], ls='none',
                    label=labels[i%3])
        axs[k].plot(vxmeansu[plotseq[i]], vxfitsu[plotseq[i]], color=colors[i%3],
                    label=r'$\alpha_{:s}$ ={:0.2f}'.format(labels[i%3][5:6], a_lcts[plotseq[i]]))

    axs[0].plot(vxmeans_bt1 * unit, vx_ratesu, marker='o', markerfacecolor='none', ls='-.', color='green', label=r'$\alpha_B = {:0.2f}$'.format(a_avg1))
    axs[1].plot(vxmeans_bt2 * unit, vx_ratesu, marker='o', markerfacecolor='none', ls='-.', color='green', label=r'$\alpha_B = {:0.2f}$'.format(a_avg2))

    axs[0].axis([-maxv, maxv, -maxv, maxv])
    axs[1].axis([-maxv, maxv, -maxv, maxv])
    axs[0].set_xlabel('Measured velocity {:s}'.format(unit_str), fontsize=fs)
    axs[0].set_ylabel('True velocity {:s}'.format(unit_str), fontsize=fs)
    axs[1].set_xlabel('Measured velocity {:s}'.format(unit_str), fontsize=fs)
    axs[1].set_ylabel('True velocity {:s}'.format(unit_str), fontsize=fs)
    axs[0].legend(loc=legend_loc)
    axs[0].grid(True)
    axs[0].set_aspect('equal')
    axs[1].legend(loc=legend_loc)
    axs[1].grid(True)
    axs[1].set_aspect('equal')

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(DATADIR, '{:s}_linear_fit_{:s}.png'.format(data_type, fig_suffix)), dpi=dpi)


    ##################################################################################################################
    ### Residuals uncorrected
    ##################################################################################################################
    max_resid_uncorrected = 1000#0.1 * unit

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    axs[0].set_title('FWHM = 7 px ({:s})'.format(data_type))
    axs[1].set_title('FWHM = 15 px ({:s})'.format(data_type))
    axs[0].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[1].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[0].set_ylabel('Residuals {:s}'.format(unit_str))
    axs[1].set_ylabel('Residuals {:s}'.format(unit_str))
    bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
    colors = ['black', 'red', 'blue']


    for i in range(6):
        k = 1 - int(i / 3)  # LCT 7 px must be on the 0th axis (left-hand side)
        axs[k].bar(vx_ratesu, lct_residuals0u[plotseq[i], :],
                   width=widths[i%3] * unit, color=colors[i%3], tick_label=bar_labels, label=labels[i%3])

    axs[0].bar(vx_ratesu, bt_residuals1 * unit, width=widths[2]*0.7 * unit, color='green', tick_label=None, label='balltracking')
    axs[1].bar(vx_ratesu, bt_residuals2 * unit, width=widths[2]*0.7 * unit, color='green', tick_label=None, label='balltracking')

    axs[0].set_ylim([0, max_resid_uncorrected])
    axs[1].set_ylim([0, max_resid_uncorrected])
    axs[0].legend()
    axs[1].legend()
    axs[0].grid(True)
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(DATADIR, '{:s}_residuals_uncorrected_{:s}.png'.format(data_type, fig_suffix)), dpi=dpi)

    ######################################
    ##### Residuals with linear correction
    ######################################
    max_resid = 85 #0.01 * unit / 2

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    axs[0].set_title('FWHM = 7 px ({:s}, linearly corrected)'.format(data_type))
    axs[1].set_title('FWHM = 15 px ({:s}, linear corrected)'.format(data_type))
    axs[0].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[1].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[0].set_ylabel('Residuals {:s}'.format(unit_str))
    axs[1].set_ylabel('Residuals {:s}'.format(unit_str))
    bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
    colors = ['black', 'red', 'blue']


    for i in range(6): # Consider only the unfiltered ones (the filtered ones do not work). range from 0 to 5 incl.
        k = 1 - int(i / 3) # LCT 7 px must be on the 0th axis (left-hand side)
        axs[k].bar(vx_ratesu, lct_residualsu[plotseq[i], :],
                   width=widths[i%3] * unit, color=colors[i%3], tick_label=bar_labels, label=labels[i%3])

    axs[0].bar(vx_ratesu, bt_residuals1 * unit, width=widths[2] * 0.7 * unit, color='green', tick_label=bar_labels,
               label='balltracking')
    axs[1].bar(vx_ratesu, bt_residuals2 * unit, width=widths[2]*0.7 * unit, color='green', tick_label=bar_labels, label='balltracking')


    axs[0].set_ylim([0, max_resid])
    axs[1].set_ylim([0, max_resid])
    axs[0].legend()
    axs[1].legend()
    axs[0].grid(True, axis='y')
    axs[1].grid(True, axis='y')

    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(DATADIR, '{:s}_residuals_corrected_{:s}.png'.format(data_type, fig_suffix)), dpi=dpi)



    #############################################################################################
    ######## Filtered. Indices are  [6, 7, 8, 9, 10, 11] #############################################
    #############################################################################################

    plotseq = [6, 7, 8, 9, 10, 11]
    data_type = 'filtered'

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    axs[0].set_title('FWHM = 7 px ({:s})'.format(data_type))
    axs[1].set_title('FWHM = 15 px ({:s})'.format(data_type))

    # Consider only the unfiltered ones (the filtered ones do not work). range from 0 to 5 incl.
    for i in range(6):
        k = 1 - int(i / 3)  # LCT 7 px must be on the 0th axis (left-hand side)
        axs[k].plot(vxmeansu[plotseq[i], :], vx_ratesu, marker=markers[i % 3], color=colors[i % 3], ls='none',
                    label=labels[i % 3])
        axs[k].plot(vxmeansu[plotseq[i]], vxfitsu[plotseq[i]], color=colors[i % 3],
                    label=r'$\alpha_{:s}$ ={:0.2f}'.format(labels[i % 3][5:6], a_lcts[plotseq[i]]))

    axs[0].plot(vxmeans_bt1 * unit, vx_ratesu, marker='o', markerfacecolor='none', ls='-.', color='green',
                label=r'$\alpha_B = {:0.2f}$'.format(a_avg1))
    axs[1].plot(vxmeans_bt2 * unit, vx_ratesu, marker='o', markerfacecolor='none', ls='-.', color='green',
                label=r'$\alpha_B = {:0.2f}$'.format(a_avg2))

    axs[0].axis([-maxv, maxv, -maxv, maxv])
    axs[1].axis([-maxv, maxv, -maxv, maxv])
    axs[0].set_xlabel('Measured velocity {:s}'.format(unit_str), fontsize=fs)
    axs[0].set_ylabel('True velocity {:s}'.format(unit_str), fontsize=fs)
    axs[1].set_xlabel('Measured velocity {:s}'.format(unit_str), fontsize=fs)
    axs[1].set_ylabel('True velocity {:s}'.format(unit_str), fontsize=fs)
    axs[0].legend(loc=legend_loc)
    axs[0].grid(True)
    axs[0].set_aspect('equal')
    axs[1].legend(loc=legend_loc)
    axs[1].grid(True)
    axs[1].set_aspect('equal')

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(DATADIR, '{:s}_linear_fit_{:s}.png'.format(data_type,fig_suffix)), dpi=dpi)

    ##################################################################################################################
    ### Residuals uncorrected
    ##################################################################################################################
    max_resid_uncorrected = 1000  # 0.1 * unit

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    axs[0].set_title('FWHM = 7 px ({:s})'.format(data_type))
    axs[1].set_title('FWHM = 15 px ({:s})'.format(data_type))
    axs[0].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[1].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[0].set_ylabel('Residuals {:s}'.format(unit_str))
    axs[1].set_ylabel('Residuals {:s}'.format(unit_str))
    bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
    colors = ['black', 'red', 'blue']

    for i in range(6):
        k = 1 - int(i / 3)  # LCT 7 px must be on the 0th axis (left-hand side)
        axs[k].bar(vx_ratesu, lct_residuals0u[plotseq[i], :],
                   width=widths[i % 3] * unit, color=colors[i % 3], tick_label=bar_labels, label=labels[i % 3])

    axs[0].bar(vx_ratesu, bt_residuals1 * unit, width=widths[2] * 0.7 * unit, color='green', tick_label=None,
               label='balltracking')
    axs[1].bar(vx_ratesu, bt_residuals2 * unit, width=widths[2] * 0.7 * unit, color='green', tick_label=None,
               label='balltracking')

    axs[0].set_ylim([0, max_resid_uncorrected])
    axs[1].set_ylim([0, max_resid_uncorrected])
    axs[0].legend()
    axs[1].legend()
    axs[0].grid(True)
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(DATADIR, '{:s}_residuals_uncorrected_{:s}.png'.format(data_type, fig_suffix)), dpi=dpi)

    ######################################
    ##### Residuals with linear correction
    ######################################
    max_resid = 85  # 0.01 * unit / 2

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    axs[0].set_title('FWHM = 7 px ({:s}, linearly corrected)'.format(data_type))
    axs[1].set_title('FWHM = 15 px ({:s}, linearly corrected)'.format(data_type))
    axs[0].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[1].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[0].set_ylabel('Residuals {:s}'.format(unit_str))
    axs[1].set_ylabel('Residuals {:s}'.format(unit_str))
    bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
    colors = ['black', 'red', 'blue']

    for i in range(6):  # Consider only the unfiltered ones (the filtered ones do not work). range from 0 to 5 incl.
        k = 1 - int(i / 3)  # LCT 7 px must be on the 0th axis (left-hand side)
        axs[k].bar(vx_ratesu, lct_residualsu[plotseq[i], :],
                   width=widths[i % 3] * unit, color=colors[i % 3], tick_label=bar_labels, label=labels[i % 3])

    axs[0].bar(vx_ratesu, bt_residuals1 * unit, width=widths[2] * 0.7 * unit, color='green', tick_label=bar_labels,
               label='balltracking')
    axs[1].bar(vx_ratesu, bt_residuals2 * unit, width=widths[2] * 0.7 * unit, color='green', tick_label=bar_labels,
               label='balltracking')

    axs[0].set_ylim([0, max_resid])
    axs[1].set_ylim([0, max_resid])
    axs[0].legend()
    axs[1].legend()
    axs[0].grid(True, axis='y')
    axs[1].grid(True, axis='y')

    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(DATADIR, '{:s}_residuals_corrected_{:s}.png'.format(data_type, fig_suffix)), dpi=dpi)



