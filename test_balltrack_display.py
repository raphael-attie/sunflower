#%gui qt
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import balltracking.balltrack as blt
import fitstools
import fitsio
import filters


# Path to fits file (fits cube)
#file    = '/Users/rattie/Data/SDO/HMI/EARs/AR11130_2010_11_27/mtrack_20101126_170034_TAI_20101127_170034_TAI_LambertCylindrical_continuum.fits'
file = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/series_continuum/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum_00000.fits'
#file = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/series_continuum_calibration/calibration/drift0.20/drift_0001.fits'
# Get the header
h       = fitstools.fitsheader(file)
# Get the 1st image
image   = fitsio.read(file)
image   = image.astype(np.float32)

# Filter image
ffilter_hpf = filters.han2d_bandpass(image.shape[0], 0, 5)
fdata_hpf = filters.ffilter_image(image, ffilter_hpf)
sigma0 = image[1:200, 1:200].std()
sigma_hpf = fdata_hpf[1:200, 1:200].std()

surface0 = blt.rescale_frame(image, sigma0)
surface_hpf = blt.rescale_frame(fdata_hpf, 2*sigma_hpf)

#surface = np.zeros([30, 60])
nt = 10
rs = 2
dp = 0.2
bt = blt.BT(surface0.shape, nt, rs, dp)
# Initialize ball positions with height
bt.initialize_ballpos(surface_hpf)





bsize = np.ones([bt.xstart.ravel().shape[0]], dtype=float)*4
skip = 24

# Background colors
blue=(0.16, 0.28, 0.46)
gray=(0.5, 0.5, 0.5)


# Display the original data as a surface.
f0 = mlab.figure(size=(900, 900), bgcolor=gray)
s0 = mlab.imshow(bt.xcoords, bt.ycoords, image, vmin=image.mean() - 3*sigma0, vmax=image.mean() + 3*sigma0, colormap='afmhot')


# Display the filtered data surface as a surface and with the 3d wireframe mesh.
f1 = mlab.figure(size=(900, 900), bgcolor=gray)
s1 = mlab.surf(bt.xcoords, bt.ycoords, surface_hpf, warp_scale=1.0, vmin=-2, vmax=2, colormap='afmhot', line_width=1, opacity=1)
s1.actor.mapper.interpolate_scalars_before_mapping = True
s1.actor.property.edge_visibility = True
# Add the balls as spherical glyphs with proper radius
pts1 = mlab.quiver3d(bt.pos_t[0, ::skip], bt.pos_t[1, ::skip], bt.pos_t[2, ::skip], bsize[::skip], bsize[::skip], bsize[::skip], scale_factor=1.0, scalars=bt.pos_t[2, ::skip], opacity=0.8, mode ='sphere', vmin=-3, vmax=3, resolution=16, colormap='rainbow', line_width=1)
pts1.glyph.color_mode = 'color_by_scalar'
pts1.glyph.glyph_source.glyph_source.center = [0, 0, 0]
# Edge visibility of the balls
# pts0.actor.property.edge_visibility = True
mlab.sync_camera(f0, f1)
mlab.show()





size = 50
sigma = 2
surface = blt.gauss2d(size, sigma) * 3

rs = 2
dp = 0.2
nt = 50

bt = blt.BT(surface.shape, nt, rs, dp)

# Initialize 1 ball
xstart = [20, 23]
ystart = [24, 27]
zstart = blt.put_balls_on_surface(surface, xstart, ystart, rs, dp)

pos, vel = blt.initialize_ball_vector(xstart, ystart, zstart)
pos, vel, force = [np.array(v).squeeze() for v in zip(*[blt.integrate_motion(pos, vel, bt, surface) for i in range(nt)])]
pos2, vel2 = blt.initialize_ball_vector(xstart, ystart, zstart)
pos2, vel2, force2 = [np.array(v).squeeze() for v in zip(*[blt.integrate_motion2(pos2, vel2, bt, surface) for i in range(nt)])]
# Display and compare results
f1 = plt.figure(figsize=(10, 10))
plt.imshow(surface, origin='lower', cmap='gray')
plt.plot(xstart, ystart, 'r+', markersize=10)

plt.plot(pos[:,0, 0], pos[:,1, 0], 'go', markerfacecolor='none')
plt.plot(pos2[:, 0, 0], pos2[:,1, 0], 'b+', markerfacecolor='none')

plt.plot(pos[:,0, 1], pos[:,1, 1], 'go', markerfacecolor='none')
plt.plot(pos2[:, 0, 1], pos2[:,1, 1], 'b+', markerfacecolor='none')



### Simple benchmark of the integration
xstart = np.full([16129], 20)
ystart = np.full([16129], 30)
zstart = blt.put_balls_on_surface(surface, xstart, ystart, rs, dp)

pos, vel = blt.initialize_ball_vector(xstart, ystart, zstart)
mywrap = wrapper(blt.integrate_motion, pos, vel, bt, surface)
timeit(mywrap, number = 100)

pos2, vel2 = blt.initialize_ball_vector(xstart, ystart, zstart)
mywrap2 = wrapper(blt.integrate_motion2, pos2, vel2, bt, surface)
timeit(mywrap2, number = 100)
### End of benchmark





# fig = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(bt.meshx, bt.meshy, surface)
# z = blt.bilin_interp(surface, bt.xstart.ravel(), bt.ystart.ravel())


