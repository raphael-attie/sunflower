# Need to run %gui qt to make mayavi work here.
#%gui qt
from mayavi import mlab
import numpy as np
import balltracking.balltrack as blt


# Path to fits file (fits cube)
file = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'

#surface = np.zeros([30, 60])
nt = 10
rs = 2
dp = 0.2
ballspacing = 4 * rs
sigma_factor = 1.3
bt = blt.BT(nt, rs, dp, ballspacing=ballspacing, sigma_factor=sigma_factor, datafiles=file)

bt.initialize()
smin = bt.surface.min()
smax = bt.surface.max()




# Background colors
blue=(0.16, 0.28, 0.46)
gray=(0.5, 0.5, 0.5)


# Display the original data as a surface.
# f0 = mlab.figure(size=(900, 900), bgcolor=gray)
# s0 = mlab.imshow(bt.xcoords, bt.ycoords, bt.surface, vmin=smin, vmax=smax, colormap='afmhot')

xmin, xmax = 0, 50
ymin, ymax = 0, 50
skip = 1

xslice = slice(xmin,xmax)
yslice = slice(ymin,ymax)

xcoords = bt.xcoords[xslice]
ycoords = bt.ycoords[yslice]
surface = bt.surface[yslice, xslice]
xmask = (bt.pos[0, :] < xmax-bt.rs) & (bt.pos[0, :] > xmin)
ymask = (bt.pos[1, :] < ymax-bt.rs) & (bt.pos[1, :] > ymin)
bmask = xmask & ymask
xpos = bt.pos[0, bmask][::skip]
ypos = bt.pos[1, bmask][::skip]
zpos = bt.pos[2, bmask][::skip]
#bsize = np.ones([bt.xstart.ravel().shape[0]], dtype=float)*4
bsize = np.ones(np.count_nonzero(bmask), dtype=float)[::skip]*2


mlab.options.offscreen = False
# Display the filtered data surface as a surface and with the 3d wireframe mesh.
f1 = mlab.figure(size=(1300, 1100), bgcolor=gray)
s1 = mlab.surf(xcoords, ycoords, surface, warp_scale=1.0, vmin=-2, vmax=2, colormap='hot',
               line_width=1, opacity=1)

s1.actor.mapper.interpolate_scalars_before_mapping = True
s1.actor.property.edge_visibility = True
# Add the balls as spherical glyphs with proper radius
# pts1 = mlab.quiver3d(0, 0, 0, 2, 2, 2,
#                      scale_factor=1.0, scalars=1, opacity=0.5, mode ='sphere', vmin=-3, vmax=3,
#                      resolution=16, colormap='rainbow', line_width=1)
pts1 = mlab.quiver3d(xpos, ypos, zpos, bsize, bsize, bsize,
                     scale_factor=1.1, scalars=zpos+1, opacity=1, mode ='sphere', vmin=-2, vmax=2,
                     resolution=12, colormap='hot', line_width=1)

# pts1 = mlab.points3d(xpos, ypos, zpos-1, zpos,
#                      scale_factor=2.0, opacity=0.5, mode ='sphere', vmin=-2, vmax=2,
#                      resolution=12, colormap='hot', line_width=1, scale_mode='none')

pts1.glyph.color_mode = 'color_by_scalar'
#pts1.glyph.glyph_source.glyph_source.center = [0, 0, 0]
pts1.glyph.glyph_source.glyph_position = 'center'
# Edge visibility of the balls
pts1.actor.property.edge_visibility = True

mlab.colorbar(s1, title='intensity <-> height', orientation='vertical')

scene = s1.scene
scene.camera.position = [73.95775697560468, 127.68336925468316, 71.49539073525987]
scene.camera.focal_point = [24.5, 24.5, -0.041573286056518555]
scene.camera.view_angle = 30.0
scene.camera.view_up = [-0.26394010350999647, -0.46091215285079035, 0.8472872058007145]
scene.camera.clipping_range = [73.62458996197505, 211.5042604946076]
scene.camera.compute_view_plane_normal()
scene.render()

cam = s1.scene.camera
cam.zoom(1.2)
mlab.draw()

#mlab.show()
#mlab.sync_camera(f0, f1)

mlab.savefig('/Users/rattie/Desktop/test.png')





# size = 50
# sigma = 2
# surface = blt.gauss2d(size, sigma) * 3
#
# rs = 2
# dp = 0.2
# nt = 50
#
# bt = blt.BT(surface.shape, nt, rs, dp)
#
# # Initialize 1 ball
# xstart = [20, 23]
# ystart = [24, 27]
# zstart = blt.put_balls_on_surface(surface, xstart, ystart, rs, dp)
#
# pos, vel = blt.initialize_ball_vector(xstart, ystart, zstart)
# pos, vel, force = [np.array(v).squeeze() for v in zip(*[blt.integrate_motion(pos, vel, bt, surface) for i in range(nt)])]
# pos2, vel2 = blt.initialize_ball_vector(xstart, ystart, zstart)
# pos2, vel2, force2 = [np.array(v).squeeze() for v in zip(*[blt.integrate_motion2(pos2, vel2, bt, surface) for i in range(nt)])]
# # Display and compare results
# f1 = plt.figure(figsize=(10, 10))
# plt.imshow(surface, origin='lower', cmap='gray')
# plt.plot(xstart, ystart, 'r+', markersize=10)
#
# plt.plot(pos[:,0, 0], pos[:,1, 0], 'go', markerfacecolor='none')
# plt.plot(pos2[:, 0, 0], pos2[:,1, 0], 'b+', markerfacecolor='none')
#
# plt.plot(pos[:,0, 1], pos[:,1, 1], 'go', markerfacecolor='none')
# plt.plot(pos2[:, 0, 1], pos2[:,1, 1], 'b+', markerfacecolor='none')
#
#
#
