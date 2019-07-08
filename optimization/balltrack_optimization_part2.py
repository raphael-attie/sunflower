"""
plot quality matrix of Balltracking.
Using Mean Absolute Error (MAE) instead of root-mean-square error (RMSE)
https://en.wikipedia.org/wiki/Mean_absolute_error
"""

import os, glob
import numpy as np
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition




def make_args_list(intsteps, dps, sigmas):
    # Build argument list for parallelization of run_balltrack()
    mesh_intsteps, mesh_dp, mesh_sigma_factor = np.meshgrid(intsteps, dps, sigmas, indexing='ij')
    intsteps_ravel = np.ravel(mesh_intsteps)
    dp_ravel = np.ravel(mesh_dp)
    sigmaf_ravel = np.ravel(mesh_sigma_factor)
    args_list = [list(a) for a in zip(intsteps_ravel, dp_ravel, sigmaf_ravel)]
    return args_list

plt.rcParams.update({'font.size': 11})

DATADIR = '/Users/rattie/Data/Ben/SteinSDO/calibration/'
# At FWHM 7
fwhm = 7
###### Load all results from Balltracking
npz = np.load(os.path.join(DATADIR, 'all_results_fwhm_{:d}_simulation.npz'.format(fwhm)))
vxfit_avg_la, residuals_la, a_avg_la, vxmeans_la = [npz[key] for key in npz.keys()]


npts = 9
nframes = 80
trange = [0, nframes]
vx_rates = np.linspace(-0.2, 0.2, npts)
unit = 368000 / 60
unit_str = '[m/s]'
vx_ratesu = vx_rates * unit


intsteps_l = [3, 4, 5]
dp_l = [0.1, 0.2, 0.3, 0.4, 0.5]
sigma_factor_l = [1, 1.25, 1.5, 1.75, 2]
args_l = np.array(make_args_list(intsteps_l, dp_l, sigma_factor_l))
residuals_means = np.array([res.mean() for res in residuals_la])
residuals_sigma = np.array([res.std() for res in residuals_la])
# Axis range in balltracking parameters coordinate space
extent = [sigma_factor_l[0]-0.25/2, sigma_factor_l[-1]+0.25/2, dp_l[0]-0.05, dp_l[-1]+0.05]



slices = [slice(i*25,(i+1)*25) for i in range(3)]
vmin = 0
vmax = 10

# Fontsize
dpi = 300
plt.close('all')
ims = []
fig, ax = plt.subplots(2,3, figsize=(16,7.5))
for i in range(3):
    # Mean residuals
    ims.append(ax[0,i].imshow(residuals_means[slices[i]].reshape([5, 5])*unit, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='magma'))
    ax[0,i].set_xlabel(r'${\sigma}_{N_{MOD}}$')
    ax[0,i].set_xticks(sigma_factor_l)
    ax[0,i].set_yticks(dp_l)
    ax[0,i].set_title(r'$N_{int}$ = '+ '{:d}'.format(intsteps_l[i]))
    # Standard deviations
    ims.append(ax[1,i].imshow(residuals_sigma[slices[i]].reshape([5, 5])*unit, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='viridis'))
    ax[1,i].set_xlabel(r'${\sigma}_{N_{MOD}}$')
    ax[1,i].set_xticks(sigma_factor_l)
    ax[1,i].set_yticks(dp_l)
    ax[1,i].set_title(r'$N_{int}$ = '+ '{:d}'.format(intsteps_l[i]))

ax[0,0].set_ylabel('Dp')
ax[1,0].set_ylabel('Dp')
# divider = make_axes_locatable(ax[2])
# cax = divider.append_axes("right", size="3%", pad=0.3)
# fig.colorbar(im3, cax=cax, ax=[ax[0], ax[1], ax[2]], orientation='vertical')
fig.subplots_adjust(left=0.05, bottom=0, top=0.98, right=0.98, hspace=0, wspace=0.1)
cbar_ax = fig.add_axes([0.1, 1, 0.7, 0.05])
ip = InsetPosition(ax[0,0], [0, 1.3, 3.2, 0.05])
cbar_ax.set_axes_locator(ip)
cb = fig.colorbar(ims[0], cax=cbar_ax, ax=[ax[0,0], ax[0,1], ax[0,2]], orientation='horizontal')
#cb.ax.tick_params(labelsize=FS)
cbar_ax.set_title('MAE {:s}'.format(unit_str))

cbar_ax2 = fig.add_axes([0.1, 1, 0.5, 0.05])
ip2 = InsetPosition(ax[1,0], [0, 1.3, 3.2, 0.05])
cbar_ax2.set_axes_locator(ip2)
cb2 = fig.colorbar(ims[1], cax=cbar_ax2, ax=[ax[1,0], ax[1,1], ax[1,2]], orientation='horizontal')
cbar_ax2.set_title(r'$\sigma_{MAE}$ ' +'{:s}'.format(unit_str))

plt.show()

plt.savefig(os.path.join(DATADIR, 'balltrack_mean_sigma_residuals.png'), dpi=dpi)

#### Unfold residuals for each parameters-tuplet for detailed view over all tests
bar_labels = ['{:0.0f}'.format(vxrate) for vxrate in vx_ratesu]
plt.figure(figsize=(6,6))
for i in range(len(args_l)):
    plt.bar(vx_ratesu, residuals_la[i] * unit, width=0.01 * unit, color='gray', tick_label=bar_labels, label='average')
    plt.xlabel('velocity [m/s]')
    plt.ylabel('Absolute residual error [m/s]')
    plt.ylim([0, 10])
    plt.grid(True, axis='both')
    plt.title('fwhm: 7px - int. steps:{:0.0f} dp:{:0.2f} sigma factor:{:0.2f}'.format(*args_l[i]))
    #plt.show()
    fig_suffix = 'intsteps_{:0.0f}_dp_{:0.2f}_sigma_factor_{:0.2f}'.format(*args_l[i])
    plt.savefig(os.path.join(DATADIR, 'plots_average_residuals', 'bt_avg_res_{:s}.png'.format(fig_suffix)), dpi=dpi)
    #plt.tight_layout()
    plt.cla()

plt.close('all')
