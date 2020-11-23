##############################
# Properties
##############################
# Parameters/properties
nx_ref = 960
ny_ref = 960
dx_ref = 70.0*0.001
dy_ref = 70.0*0.001
xi_ref = 0
xf_ref = xi_ref + nx_ref
yi_ref = 0
yf_ref = yi_ref + ny_ref
x_ref = np.linspace(xi_ref, xf_ref, nx_ref)*dx_ref
y_ref = np.linspace(yi_ref, yf_ref, ny_ref)*dy_ref
nx_crop = 100
ny_crop = 100
xi_crop = 400
xf_crop = xi_crop + nx_crop
yi_crop = 400
yf_crop = yi_crop + ny_crop
x_crop = np.linspace(xi_crop, xf_crop, nx_crop)*dx_ref
y_crop = np.linspace(yi_crop, yf_crop, ny_crop)*dy_ref
#################################
# Figure
#################################
# QUIVER
spl_nb = 1
scale = 7.0
qk_length = 2.0
width=0.006
headwidth=3
headlength=3
vunit=1
# Figure
frame = 0
font_size = 10
figsize_x = 7.8
figsize_y = 3.5
x_label = 'x (Mm)'
y_label = 'y (Mm)'
colorbar_cmap = 'GnBu_r'
colorbar_label = r'Intensity ($I/\langle I \rangle$)'
# title_label = 'IBIS'
plt.rcParams["font.family"] = "Times New Roman"
fig = plt.figure(figsize=(figsize_x, figsize_y))
spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1.00, 1.05], height_ratios=[1], wspace=0.05,
                         hspace=0.0)
ax1 = fig.add_subplot(spec[0, 0])
I = ax1.imshow(np.flipud(ic_ibis[frame, :, :]), extent=[min(x_ref), max(x_ref), min(y_ref), max(y_ref)], cmap=colorbar_cmap, aspect=1, interpolation='none', vmin=min_cb, vmax=max_cb)
ax1.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size, left=True, right=True)
ax1.set_ylabel(y_label, fontsize=font_size, labelpad=5.0)
ax1.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size, bottom=True, top=True)
ax1.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
ax1.get_xaxis().set_major_locator(plt.MultipleLocator(20))
ax1.get_yaxis().set_major_locator(plt.MultipleLocator(20))
rect = patches.Rectangle((400*dx_ref, 400*dy_ref), 100*dx_ref, 100*dy_ref, linewidth=1, edgecolor='r', facecolor='none')
ax1.add_patch(rect)
# Title
ax1.set_title(r'(a) IBIS 7200Å $I(\tau \approx 1)$, $t$={0:.0f} s'.format(frame*12.0), fontsize=font_size, y=1.02, wrap=True)
ax = fig.add_subplot(spec[0, 1])
I2 = ax.imshow(np.flipud(ic_ibis[frame, :, :]), extent=[min(x_crop), max(x_crop), min(y_crop), max(y_crop)], cmap=colorbar_cmap, aspect=1, interpolation='none', vmin=min_cb, vmax=max_cb)
Q = ax.quiver(x_crop[::spl_nb], y_crop[::spl_nb], vx_ibis[frame, ::spl_nb, ::spl_nb] * vunit,
              vy_ibis[frame, ::spl_nb, ::spl_nb] * vunit, units='xy', scale=scale,
              width=width, headwidth=headwidth, headlength=headlength)
qk_label = str(np.around(qk_length, decimals=1))
qk = ax.quiverkey(Q, 0.878, 0.02, qk_length, qk_label + r' km s$^{-1}$', labelpos='E',
                  coordinates='figure', fontproperties={'size': '10'},
                  labelsep=0.05)
ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size, left=True, right=True, labelleft=True, labelright=False)
ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size, bottom=True, top=True)
ax.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
ax.get_xaxis().set_major_locator(plt.MultipleLocator(1))
ax.get_yaxis().set_major_locator(plt.MultipleLocator(1))
ax.set_title(r'(b) DeepVelU $\textbf{{v}}_{{\small{{\textrm{{D}}, \, t}}}}$($\tau \approx 1$)', fontsize=font_size, y=1.02, wrap=True)
# Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0)
cb = colorbar(I, extend='neither', cax=cax)
cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
cb.set_label(colorbar_label, labelpad=20.0, rotation=270, size=font_size)