import matplotlib
matplotlib.use('agg')
import os, glob
import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt
import fitstools
import balltracking.balltrack as blt

DATADIR = '/Users/rattie/Data/karin/matrix_data/'

if __name__ == '__main__':

    def lct_analysis(lctfiles):

        vxmeans = []
        for i in range(len(lctfiles)):
            idl_dict = readsav(lctfiles[i])
            vx = idl_dict['vx'].mean(axis=0) #* 60/368
            time_factor = 1
            if '9min' in lctfiles[i]:
                time_factor = 9
            elif '6min' in lctfiles[i]:
                time_factor=6
            elif '3min' in lctfiles[i]:
                time_factor=3

            vx *= 1/time_factor
            vxmeans.append(vx[fov_slices[1], fov_slices[0]].mean())
        vxmeans = np.array(vxmeans) - vxmeans[4]
        ## Calibration parameters
        p = np.polyfit(vx_rates, vxmeans, 1)
        a_lct = 1 / p[0]
        vxfit = a_lct * (vxmeans - p[1])
        # Calculate residuals
        lct_residuals0 = np.abs(vxmeans - vx_rates)
        lct_residuals = np.abs(vxmeans * a_lct - vxfit)

        return vxmeans, a_lct, vxfit, lct_residuals0, lct_residuals

    def balltrack_calibration(fwhm, intsteps):

        trange = [0, nframes]

        if reprocess_bt:
            cal = blt.Calibrator(images2, drift_rates, nframes, rs, dp, sigma_factor, fwhm, outputdir,
                                 intsteps=intsteps,
                                 output_prep_data=False, use_existing=use_existing,
                                 nthreads=5)

            ballpos_top_list, ballpos_bottom_list = cal.balltrack_all_rates()
        else:
            print('Load existing tracked data at all rates')
            ballpos_top_list = np.load(os.path.join(outputdir, 'ballpos_top_list.npy'))
            ballpos_bottom_list = np.load(os.path.join(outputdir, 'ballpos_bottom_list.npy'))


        xrates = np.array(drift_rates)[:, 0]
        a_top, vxfit_top, vxmeans_top = blt.fit_calibration(ballpos_top_list, xrates, trange, fwhm,
                                                            images.shape[0:2], fov_slices,
                                                            return_flow_maps=False)
        a_bottom, vxfit_bottom, vxmeans_bottom = blt.fit_calibration(ballpos_bottom_list, xrates, trange, fwhm,
                                                                     images.shape[0:2], fov_slices,
                                                                     return_flow_maps=False)
        # Fit the averaged calibrated balltrack velocity
        vxmeans_bt = (vxmeans_top * a_top + vxmeans_bottom * a_bottom) / 2
        vxmeans_bt -= vxmeans_bt[4]
        p = np.polyfit(vx_rates, vxmeans_bt, 1)
        a_avg = 1 / p[0]
        vxfit_avg = a_avg * (vxmeans_bt - p[1])
        # Calculate residuals
        bt_residuals = np.abs(vxmeans_bt - vxfit_avg)

        return vxmeans_bt, a_avg, vxfit_avg, bt_residuals

    reprocess_lct = False
    # Set if we balltrack again or use previous results
    reprocess_bt = True
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
    trim = int(vx_rates.max() * nframes + fwhms[0] + 2)
    fov_slices = np.s_[trim:imsize - trim, fwhms[0]:imsize-fwhms[0]]

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
    dp = 0.2
    # Multiplier to the standard deviation.
    sigma_factor = 2

    # Select only a subset of nframes files
    selected_files = datafiles[0:nframes]
    # FOV
    # Here we need to exclude the region that got circularly shifted by the Fourier phase shit ~ 23 pixels in both horizontal directions (left-right)
    fov_slices = [np.s_[23:263-23, 0:263],]


    # Load the nt images
    images = fitstools.fitsread(selected_files)
    # Must make even dimensions for the fast fourier transform
    images2 = np.zeros([264, 264, images.shape[2]])
    images2[0:263, 0:263, :] = images.copy()
    images2[263, :] = images.mean()
    images2[:, 263] = images.mean()


    intsteps = 4
    ##########################################
    ######  Smoothing at FWHM = 7 px #########
    fwhm = 7
    vxmeans_bt1, a_bt1, vxfit_bt1, residuals_bt1 = balltrack_calibration(fwhm, intsteps)
    ##########################################
    ######  Smoothing at FWHM = 15 px #########
    fwhm = 15
    vxmeans_bt2, a_bt2, vxfit_bt2, residuals_bt2 = balltrack_calibration(fwhm, intsteps)

    ################################################
    ################ Plot results ##################
    ################################################

    vx_ratesu = vx_rates * unit

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
    dpi = 300
    fig_suffix = 'intsteps_{:d}'.format(intsteps)


    #############################################################################################
    ######## Unfiltered. Indices are  [0, 1, 2, 3, 4, 5] ########################################
    #############################################################################################

    plotseq = [0, 1, 2, 3, 4, 5]

    maxv = 0.3 * unit

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(18, 10))

    axs[0].plot(vx_ratesu, vx_ratesu, marker='.', ms=8, ls='--', color='gray', label='1:1')
    axs[1].plot(vx_ratesu, vx_ratesu, marker='.', ms=8, ls='--', color='gray', label='1:1')
    axs[0].set_title('FWHM = 15 px (unfiltered)')
    axs[1].set_title('FWHM = 7 px (unfiltered)')

    for i in range(6):
        k = int(i/3)
        axs[k].plot(vxmeansu[plotseq[i], :], vx_ratesu, marker=markers[i % 3], color=colors[i % 3], ls='none',
                    label=labels[i%3])
        axs[k].plot(vxmeansu[plotseq[i]], vxfitsu[plotseq[i]], color=colors[i%3],
                    label=r'$\alpha_{:s}$ ={:0.2f}'.format(labels[i%3][5:6], a_lcts[plotseq[i]]))

    axs[0].plot(vxmeans_bt1 * unit, vx_ratesu, marker='o', markerfacecolor='none', ls='-.', color='purple', label='balltracking')
    axs[1].plot(vxmeans_bt2 * unit, vx_ratesu, marker='o', markerfacecolor='none', ls='-.', color='purple', label='balltracking')

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
    plt.savefig(os.path.join(DATADIR, 'unfiltered_linear_fit_{:s}.png'.format(fig_suffix)), dpi=dpi)


    ##################################################################################################################
    ### Residuals
    ##################################################################################################################
    max_resid15 = 0.1 * unit
    max_resid7 = 0.17 * unit

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    axs[0].set_title('FWHM = 15 px (unfiltered)')
    axs[1].set_title('FWHM = 7 px (unfiltered)')
    axs[0].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[1].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[0].set_ylabel('Residuals {:s}'.format(unit_str))
    axs[1].set_ylabel('Residuals {:s}'.format(unit_str))
    bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
    colors = ['black', 'red', 'blue']


    alphas = [1, 0.8, 0.6]
    for i in range(6):
        k = int(i / 3)
        axs[k].bar(vx_ratesu, lct_residuals0u[plotseq[i], :],
                   width=widths[i%3] * unit, color=colors[i%3], tick_label=bar_labels, label=labels[i%3])

    axs[0].bar(vx_ratesu, residuals_bt1 * unit, width=widths[2]*0.7 * unit, color='green', tick_label=None, label='balltracking')
    axs[1].bar(vx_ratesu, residuals_bt2 * unit, width=widths[2]*0.7 * unit, color='green', tick_label=None, label='balltracking')

    axs[0].set_ylim([0, max_resid15])
    axs[1].set_ylim([0, max_resid7])
    axs[0].legend()
    axs[1].legend()
    axs[0].grid(True)
    axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(DATADIR, 'unfiltered_residuals_{:s}.png'.format(fig_suffix)), dpi=dpi)

    ######################################
    ##### Residuals with linear correction
    ######################################
    max_resid = 0.01 * unit / 2

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    axs[0].set_title('FWHM = 15 px (unfiltered, linear factor applied)')
    axs[1].set_title('FWHM = 7 px (unfiltered, linear factor applied)')
    axs[0].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[1].set_xlabel('True velocity {:s}'.format(unit_str))
    axs[0].set_ylabel('Residuals {:s}'.format(unit_str))
    axs[1].set_ylabel('Residuals {:s}'.format(unit_str))
    bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
    colors = ['black', 'red', 'blue']

    alphas = [1, 0.8, 0.6]
    for i in range(6):
        k = int(i / 3)
        axs[k].bar(vx_ratesu, lct_residualsu[plotseq[i], :],
                   width=widths[i%3] * unit, color=colors[i%3], tick_label=bar_labels, label=labels[i%3])

    axs[0].bar(vx_ratesu, residuals_bt1 * unit, width=widths[2]*0.7 * unit, color='green', tick_label=bar_labels, label='balltracking')
    axs[1].bar(vx_ratesu, residuals_bt2 * unit, width=widths[2]*0.7 * unit, color='green', tick_label=bar_labels, label='balltracking')

    axs[0].set_ylim([0, max_resid])
    axs[1].set_ylim([0, max_resid])
    axs[0].legend()
    axs[1].legend()
    axs[0].grid(True)
    axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(DATADIR, 'unfiltered_residuals_corrected_{:s}.png'.format(fig_suffix)), dpi=dpi)

    #############################################################################################
    ######## Filtered. Indices are  [6, 7, 8, 9, 10, 11] #############################################
    #############################################################################################

    # plotseq = [6, 7, 8, 9, 10, 11]
    #
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    #
    # axs[0].plot(vx_ratesu, vx_ratesu, ls='--', color='gray', label='1:1')
    # axs[1].plot(vx_ratesu, vx_ratesu, ls='--', color='gray', label='1:1')
    # axs[0].set_title('FWHM = 15 px (filtered)')
    # axs[1].set_title('FWHM = 7 px (filtered)')
    #
    # for i in range(6):
    #     k = int(i / 3)
    #     axs[k].plot(vxmeansu[plotseq[i], :], vx_ratesu, marker=markers[i % 3], color=colors[i % 3], ls='none',
    #                 label=labels[i % 3])
    #     axs[k].plot(vxmeansu[plotseq[i]], vxfitsu[plotseq[i]], color=colors[i % 3],
    #                 label=r'$\alpha_{:s}$ ={:0.2f}'.format(labels[i % 3][5:6], a_lcts[plotseq[i]]))
    #
    # axs[0].axis([-maxv, maxv, -maxv, maxv])
    # axs[1].axis([-maxv, maxv, -maxv, maxv])
    # axs[0].set_xlabel('Measured velocity {:s}'.format(unit_str), fontsize=fs)
    # axs[0].set_ylabel('True velocity {:s}'.format(unit_str), fontsize=fs)
    # axs[1].set_xlabel('Measured velocity {:s}'.format(unit_str), fontsize=fs)
    # axs[1].set_ylabel('True velocity {:s}'.format(unit_str), fontsize=fs)
    # axs[0].legend(loc=legend_loc)
    # axs[0].grid(True)
    # axs[0].set_aspect('equal')
    # axs[1].legend(loc=legend_loc)
    # axs[1].grid(True)
    # axs[1].set_aspect('equal')
    #
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(os.path.join(DATADIR, 'filtered_linear_fit_{:s}.png'.format(fig_suffix)), dpi=dpi)
    #
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    # axs[0].set_title('FWHM = 15 px (filtered)')
    # axs[1].set_title('FWHM = 7 px (filtered)')
    # axs[0].set_xlabel('True velocity {:s}'.format(unit_str))
    # axs[1].set_xlabel('True velocity {:s}'.format(unit_str))
    # axs[0].set_ylabel('Residuals {:s}'.format(unit_str))
    # axs[1].set_ylabel('Residuals {:s}'.format(unit_str))
    # bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
    # colors = ['black', 'red', 'blue']
    # widths = [0.02, 0.015, 0.01]
    # alphas = [1, 0.8, 0.6]
    # for i in range(6):
    #     k = int(i / 3)
    #     axs[k].bar(vx_ratesu, lct_residuals0u[plotseq[i], :],
    #                width=widths[i % 3] * unit, color=colors[i % 3], tick_label=bar_labels, label=labels[i % 3])
    #     # axs[k].bar(vx_ratesu, lct_residualsu[plotseq[i], :], width=0.01*unit, color='black', tick_label=bar_labels, label='LCT (corrected)')
    #     # axs[k].bar(vx_ratesu, bt_residualsu, width=0.015*unit, color='red', tick_label=bar_labels, alpha=0.7, label='Balltracking (calibrated)')
    #
    # axs[0].set_ylim([0, max_resid])
    # axs[1].set_ylim([0, max_resid])
    # axs[0].legend()
    # axs[1].legend()
    # axs[0].grid(True)
    # axs[1].grid(True)
    # plt.tight_layout()
    #
    # plt.savefig(os.path.join(DATADIR, 'filtered_residuals_{:s}.png'.format(fig_suffix)), dpi=dpi)
    #
    # ######################################
    # ##### Residuals with linear correction
    # ######################################
    #
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    # axs[0].set_title('FWHM = 15 px (filtered, linear factor applied)')
    # axs[1].set_title('FWHM = 7 px (filtered,  linear factor applied)' )
    # axs[0].set_xlabel('True velocity {:s}'.format(unit_str))
    # axs[1].set_xlabel('True velocity {:s}'.format(unit_str))
    # axs[0].set_ylabel('Residuals {:s}'.format(unit_str))
    # axs[1].set_ylabel('Residuals {:s}'.format(unit_str))
    # bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
    # colors = ['black', 'red', 'blue']
    # widths = [0.02, 0.015, 0.01]
    # alphas = [1, 0.8, 0.6]
    # for i in range(6):
    #     k = int(i / 3)
    #     axs[k].bar(vx_ratesu, lct_residualsu[plotseq[i], :],
    #                width=widths[i % 3] * unit, color=colors[i % 3], tick_label=bar_labels, label=labels[i % 3])
    #     # axs[k].bar(vx_ratesu, lct_residualsu[plotseq[i], :], width=0.01*unit, color='black', tick_label=bar_labels, label='LCT (corrected)')
    #     # axs[k].bar(vx_ratesu, bt_residualsu, width=0.015*unit, color='red', tick_label=bar_labels, alpha=0.7, label='Balltracking (calibrated)')
    #
    # axs[0].set_ylim([0, max_resid])
    # axs[1].set_ylim([0, max_resid])
    # axs[0].legend()
    # axs[1].legend()
    # axs[0].grid(True)
    # axs[1].grid(True)
    # plt.tight_layout()
    #
    # plt.savefig(os.path.join(DATADIR, 'filtered_residuals_corrected_{:s}.png'.format(fig_suffix)), dpi=dpi)
