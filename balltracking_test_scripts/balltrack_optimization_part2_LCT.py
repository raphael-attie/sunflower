import os, glob
import numpy as np
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from scipy.io import readsav


def make_args_list(intsteps, dps, sigmas):
    # Build argument list for parallelization of run_balltrack()
    mesh_intsteps, mesh_dp, mesh_sigma_factor = np.meshgrid(intsteps, dps, sigmas, indexing='ij')
    intsteps_ravel = np.ravel(mesh_intsteps)
    dp_ravel = np.ravel(mesh_dp)
    sigmaf_ravel = np.ravel(mesh_sigma_factor)
    args_list = [list(a) for a in zip(intsteps_ravel, dp_ravel, sigmaf_ravel)]
    return args_list


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
        # The FOV slices here could be 2D due to the balltrack option to have more than 1 FOV.
        # There is only one FOV so we must select the index [0].
        vxmeans.append(vx[fov_slices[0][1], fov_slices[0][0]].mean())
    vxmeans = np.array(vxmeans)# - vxmeans[4]
    ## Calibration parameters: y_measured = p0*x_true + p1 => true = (measured - p1)*1/p0
    p = np.polyfit(vx_rates, vxmeans, 1)
    a_lct = 1 / p[0]
    vxfit = a_lct * (vxmeans - p[1])
    # Calculate residuals
    lct_residuals0 = np.abs(vxmeans - vx_rates)
    lct_residuals = np.abs(vxfit - vx_rates)

    return vxmeans, a_lct, vxfit, lct_residuals0, lct_residuals



DATADIR = '/Users/rattie/Data/Ben/SteinSDO/calibration/'

npts = 9
nframes = 80
trange = [0, nframes]
vx_rates = np.linspace(-0.2, 0.2, npts)

unit = 368000 / 60
unit_str = '[m/s]'
vx_ratesu = vx_rates * unit

imshape = [264, 264]
imsize = 263  # actual size is imshape = 264 but original size was 263 then padded to 264 to make it even for the Fourier transform

fwhms = [7, 15]
trim = int(vx_rates.max() * nframes + max(fwhms) + 2)
# FOV
# Here we need to exclude the region that got circularly shifted by the Fourier phase shit ~ 23 pixels in both horizontal directions (left-right)
fov_slices = [np.s_[trim:imsize - trim, trim:imsize - trim], ]
# At FWHM 7
fwhm = 7

npz = np.load(os.path.join(DATADIR, 'all_results_fwhm_{:d}.npz'.format(fwhm)))
vxfit_avg_la, residuals_la, a_avg_la, vxmeans_la = [npz[key] for key in npz.keys()]


intsteps_l = [3, 4, 5]
dp_l = [0.1, 0.2, 0.3, 0.4, 0.5]
sigma_factor_l = [1, 1.25, 1.5, 1.75, 2]
args_l = np.array(make_args_list(intsteps_l, dp_l, sigma_factor_l))

residuals_means_u = np.array([res.mean() for res in residuals_la]) * unit
residuals_sigma_u = np.array([res.std() for res in residuals_la]) * unit
# average residuals for intsteps = 3, sigma_factor_l, at all dp_l


######  Get LCT results #####
lctdirs = [os.path.join('/Users/rattie/Data/karin/matrix_data/', 'test{:d}'.format(i)) for i in range(20, 32)]
lctfiles_list = [sorted(glob.glob(os.path.join(lctdir, 'flct-test*.save'))) for lctdir in lctdirs]
# Loop through the matrix data
vxmeans_list, a_lcts, vxfits, lct_residuals0, lct_residuals = zip(*[lct_analysis(lctfiles) for lctfiles in lctfiles_list])
vxmeans_arr = np.array(vxmeans_list)
vxfits_arr = np.array(vxfits)
lct_residuals0_arr = np.array(lct_residuals0)
lct_residuals_arr = np.array(lct_residuals)
lct_residualsu = lct_residuals_arr * unit
# Uncorrected
lct_residuals_means0_u = np.array([res.mean() for res in lct_residuals0]) * unit
lct_residuals_sigma0_u = np.array([res.std() for res in lct_residuals0]) * unit
# Corrected (linearly)
lct_residuals_means_u = np.array([res.mean() for res in lct_residuals]) * unit
lct_residuals_sigma_u = np.array([res.std() for res in lct_residuals]) * unit
# Let's select the best LCT case for FHWM = 7 px => dt = 6 min => index [5]
lct_index = 5
lct_selected_res0_u = lct_residuals_means0_u[lct_index]
lct_selected_sigma0_u = lct_residuals_sigma0_u[lct_index]
lct_selected_res_u = lct_residuals_means_u[lct_index]
lct_selected_sigma_u = lct_residuals_sigma_u[lct_index]

# Give the 2D properly labeled axes.
extent = [sigma_factor_l[0]-0.25/2, sigma_factor_l[-1]+0.25/2, dp_l[0]-0.05, dp_l[-1]+0.05]

dpi = 300
vmin = 0
vmax = 10

# Make 3 slices, each slice for each int. steps parameter (3,4,5)
slices = [slice(i*25,(i+1)*25) for i in range(3)]

res_bt_lct = residuals_means_u[slices[0]].reshape([5, 5]) - lct_selected_res_u
vmin2 = - np.abs(res_bt_lct).max()
vmax2 = np.abs(res_bt_lct).max()


ims = []
ims2 = []
ims3 = []
plt.close('all')
fig, axs = plt.subplots(3,3, figsize=(16,10))
for i in range(3):
    ims.append(
        axs[0, i].imshow(residuals_means_u[slices[i]].reshape([5, 5]), origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='magma'))
    axs[0, i].set_xlabel('sigma_factor')
    axs[0, i].set_xticks(sigma_factor_l)
    axs[0, i].set_yticks(dp_l)
    axs[0, i].set_title('int. steps: {:d}'.format(intsteps_l[i]))

    # Difference with LCT
    res_bt_lct = residuals_means_u[slices[i]].reshape([5, 5]) - lct_selected_res_u
    ims2.append(
        axs[1, i].imshow(res_bt_lct, origin='lower', extent=extent, vmin=vmin2, vmax=vmax2, cmap='RdGy'))
    axs[1, i].set_xlabel('sigma_factor')
    axs[1, i].set_xticks(sigma_factor_l)
    axs[1, i].set_yticks(dp_l)
    axs[1, i].set_title('int. steps: {:d}'.format(intsteps_l[i]))

    # Sigma (standard deviation)
    diff_sigma = residuals_sigma_u[slices[i]].reshape([5, 5]) - lct_selected_sigma_u
    ims3.append(
        axs[2, i].imshow(diff_sigma, origin='lower', extent=extent, vmin=vmin2, vmax=vmax2, cmap='RdGy'))
    axs[2, i].set_xlabel('sigma_factor')
    axs[2, i].set_xticks(sigma_factor_l)
    axs[2, i].set_yticks(dp_l)
    axs[2, i].set_title('int. steps: {:d}'.format(intsteps_l[i]))

axs[0, 0].set_ylabel('dp')
axs[1, 0].set_ylabel('dp')
axs[2, 0].set_ylabel('dp')

fig.subplots_adjust(left=0.05, bottom=0, top=0.95, right=0.98, hspace=0.1, wspace=0.1)
cbar_ax = fig.add_axes([0.1, 1, 0.7, 0.05])
ip = InsetPosition(axs[0, 0], [0, 1.3, 3.2, 0.05])
cbar_ax.set_axes_locator(ip)
cb = fig.colorbar(ims[0], cax=cbar_ax, ax=[axs[0, 0], axs[0, 1], axs[0, 2]], orientation='horizontal')
cb.ax.tick_params(labelsize=10)
cbar_ax.set_title('Balltrack residuals {:s}'.format(unit_str), fontsize=10)

# LCT colorbar
cbar_ax2 = fig.add_axes([0.1, 0.6, 0.7, 0.05])
ip2 = InsetPosition(axs[1, 0], [0, 1.3, 3.2, 0.05])
cbar_ax2.set_axes_locator(ip2)
cb2 = fig.colorbar(ims2[0], cax=cbar_ax2, ax=[axs[1, 0], axs[1, 1], axs[1, 2]], orientation='horizontal')
cb2.ax.tick_params(labelsize=10)
cbar_ax2.set_title('BT - LCT residuals {:s}'.format(unit_str), fontsize=10)

# # sigma colorbar
cbar_ax3 = fig.add_axes([0.1, 0.3, 0.7, 0.05])
ip3 = InsetPosition(axs[2, 0], [0, 1.3, 3.2, 0.05])
cbar_ax3.set_axes_locator(ip3)
cb3 = fig.colorbar(ims3[0], cax=cbar_ax3, ax=[axs[2, 0], axs[2, 1], axs[2, 2]], orientation='horizontal')
cb3.ax.tick_params(labelsize=10)
cbar_ax3.set_title(r'$\sigma_B - \sigma_L$ {:s}'.format(unit_str), fontsize=10)

plt.show()

plt.savefig(os.path.join(DATADIR, 'balltrack_lct_residuals_corrected.png'), dpi=dpi)

#TODO: Compare LCT without linear correction with BT to show how LCT performs "as is", i.e. as everyone have been doing so far.

res_bt_lct0 = residuals_means_u[slices[0]].reshape([5, 5]) - lct_selected_res0_u
vmin3 = - np.abs(res_bt_lct0).max()
vmax3 = np.abs(res_bt_lct0).max()


ims2 = []
ims3 = []
plt.close('all')
fig, axs = plt.subplots(2,3, figsize=(16,7))
for i in range(3):
    # Difference with LCT
    res_bt_lct = residuals_means_u[slices[i]].reshape([5, 5]) - lct_selected_res0_u
    ims2.append(
        axs[0, i].imshow(res_bt_lct, origin='lower', extent=extent, vmin=vmin3, vmax=vmax3, cmap='RdGy'))
    axs[0, i].set_xlabel('sigma_factor')
    axs[0, i].set_xticks(sigma_factor_l)
    axs[0, i].set_yticks(dp_l)
    axs[0, i].set_title('int. steps: {:d}'.format(intsteps_l[i]))

    # Sigma (standard deviation)
    diff_sigma = residuals_sigma_u[slices[i]].reshape([5, 5]) - lct_selected_sigma0_u
    ims3.append(
        axs[1, i].imshow(diff_sigma, origin='lower', extent=extent, vmin=vmin3, vmax=vmax3, cmap='RdGy'))
    axs[1, i].set_xlabel('sigma_factor')
    axs[1, i].set_xticks(sigma_factor_l)
    axs[1, i].set_yticks(dp_l)
    axs[1, i].set_title('int. steps: {:d}'.format(intsteps_l[i]))

axs[0, 0].set_ylabel('dp')
axs[1, 0].set_ylabel('dp')

fig.subplots_adjust(left=0.05, bottom=0, top=0.95, right=0.98, hspace=0.1, wspace=0.1)
# LCT colorbar
cbar_ax2 = fig.add_axes([0.1, 0.6, 0.7, 0.05])
ip2 = InsetPosition(axs[0, 0], [0, 1.3, 3.2, 0.05])
cbar_ax2.set_axes_locator(ip2)
cb2 = fig.colorbar(ims2[0], cax=cbar_ax2, ax=[axs[0, 0], axs[0, 1], axs[0, 2]], orientation='horizontal')
cb2.ax.tick_params(labelsize=10)
cbar_ax2.set_title('BT - LCT residuals {:s}'.format(unit_str), fontsize=10)

# # sigma colorbar
cbar_ax3 = fig.add_axes([0.1, 0.3, 0.7, 0.05])
ip3 = InsetPosition(axs[1, 0], [0, 1.3, 3.2, 0.05])
cbar_ax3.set_axes_locator(ip3)
cb3 = fig.colorbar(ims3[0], cax=cbar_ax3, ax=[axs[1, 0], axs[1, 1], axs[1, 2]], orientation='horizontal')
cb3.ax.tick_params(labelsize=10)
cbar_ax3.set_title(r'$\sigma_B - \sigma_L$ {:s}'.format(unit_str), fontsize=10)

plt.show()

plt.savefig(os.path.join(DATADIR, 'balltrack_lct_residuals_uncorrected.png'), dpi=dpi)


# # Unfold the residuals for detailed view over all tests
# bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
# plt.figure(figsize=(6,6))
# # for i in range(len(args_l)):
# for i in range(2):
#     plt.bar(vx_ratesu, lct_residualsu)
#     plt.bar(vx_ratesu, residuals_la[i] * unit, width=0.01 * unit, color='gray', tick_label=bar_labels, label='average')
#     plt.xlabel('velocity [m/s]')
#     plt.ylabel('Absolute residual error [m/s]')
#     plt.ylim([0, 10])
#     plt.grid(True, axis='both')
#     plt.title('fwhm: 7px - int. steps:{:0.0f} dp:{:0.2f} sigma factor:{:0.2f}'.format(*args_l[i]))
#     #plt.show()
#     fig_suffix = 'intsteps_{:0.0f}_dp_{:0.2f}_sigma_factor_{:0.2f}'.format(*args_l[i])
#     plt.savefig(os.path.join(DATADIR, 'plots_average_residuals_BT_LCT', 'bt_avg_res_{:s}.png'.format(fig_suffix)), dpi=dpi)
#     #plt.tight_layout()
#     plt.cla()
#
# plt.close('all')
