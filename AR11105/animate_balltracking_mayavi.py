# Need to run %gui qt to make mayavi work here.
#%gui qt
from mayavi import mlab
import numpy as np
import balltracking.balltrack as blt
import os
import fitstools
# Background colors
blue=(0.16, 0.28, 0.46)
gray=(0.5, 0.5, 0.5)

# Path to fits file (fits cube)
file = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/mtrack_20100901_120034_TAI_20100902_120034_TAI_LambertCylindrical_continuum.fits'
datadir = '/Users/rattie/Data/SDO/HMI/EARs/AR11105_2010_09_02/python_balltracking/'
ballpos_top = np.load(os.path.join(datadir,'ballpos_top.npy'))
#surface = np.zeros([30, 60])
nt = 50
rs = 2
dp = 0.2
intsteps = 5
ballspacing = 4 * rs
sigma_factor = 1.3
bt = blt.BT(nt, rs, dp, intsteps=5, ballspacing=ballspacing, sigma_factor=sigma_factor, datafiles=file)
bt.track()



xmin, xmax = 0, 50
ymin, ymax = 0, 50


xslice = slice(xmin,xmax)
yslice = slice(ymin,ymax)

xcoords = bt.xcoords[xslice]
ycoords = bt.ycoords[yslice]

mlab.options.offscreen = False

sample = fitstools.fitsread(file, tslice=0).astype(np.float32)
surface, mean, sigma = blt.prep_data2(sample, sigma_factor=sigma_factor)
smin = surface.min()
smax = surface.max()

for n in range(nt):
    image = fitstools.fitsread(file, tslice=n).astype(np.float32)
    surface, mean, sigma = blt.prep_data2(image, sigma_factor=sigma_factor)

    surface = surface[yslice, xslice]

    xmask = (bt.ballpos[0, :, n] < xmax - bt.rs) & (bt.ballpos[0, :, n] > xmin)
    ymask = (bt.ballpos[1, :, n] < ymax - bt.rs) & (bt.ballpos[1, :, n] > ymin)
    bmask = xmask & ymask
    xpos = bt.ballpos[0, bmask, n]
    ypos = bt.ballpos[1, bmask, n]
    zpos = bt.ballpos[2, bmask, n]
    bsize = np.ones(np.count_nonzero(bmask), dtype=float) * 2

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
    cam.zoom(1.3)
    mlab.draw()

    #mlab.show()
    #mlab.sync_camera(f0, f1)

    mlab.savefig(os.path.join(datadir, 'figures_3D/test_{:02d}.png'.format(n)))
    mlab.close()


