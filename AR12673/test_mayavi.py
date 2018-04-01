#%gui qt
import glob, os
import numpy as np
import fitstools
import fitsio
import datetime
import balltracking.balltrack as blt
from mayavi import mlab
from matplotlib.cm import get_cmap

def get_avg_data(file, tslice):
    samples = fitstools.fitsread(file, tslice=tslice).astype(np.float32)
    avg_data = samples.mean(axis=2)
    return avg_data


def get_vel(frame_number):
    vx = fitsio.read(vx_files[frame_number]).astype(np.float32)
    vy = fitsio.read(vy_files[frame_number]).astype(np.float32)
    v = [vx, vy]
    return v


def get_data(frame_number):

    mag = get_avg_data(datafilem, tslices[frame_number])
    v = get_vel(frame_number)
    lanes = blt.make_lanes(*v, nsteps, maxstep)

    return mag, lanes, v



datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'
datafilem = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
tracking_dir ='/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/python_balltracking'
vx_files = glob.glob(os.path.join(tracking_dir,'vx_[0-9]*.fits'))
vy_files = glob.glob(os.path.join(tracking_dir,'vy_[0-9]*.fits'))
lanes_files = glob.glob(os.path.join(tracking_dir,'lanes_[0-9]*.fits'))

nlanes = len(lanes_files)
### Lanes parameters
nsteps = 50
maxstep = 4


sample = fitstools.fitsread(datafile, tslice=0).astype(np.float32)
header = fitstools.fitsheader(datafile)
### time windows of the flow maps
nframes = int((3600*24*2 + 18*3600)/45) # 5280 frames
tspan = 80
tstep = 40
tcenters = np.arange(0, nframes-tstep, tstep)
tranges = [[tcenters[i], tcenters[i]+ tspan] for i in range(tcenters.size)]
# Build list of slices for extracting the corresponding magnetograms
tslices = [slice(trange[0], trange[1]) for trange in tranges]

### Build a list of datetime centered on each flow map
# Middle date of first map
dtime = datetime.datetime(year=2017, month=9, day=1, hour=0, minute=30, second=0)
dstep = datetime.timedelta(minutes=30)
dtimes = [dtime + i*dstep for i in range(len(tranges))]

# Grab data
frame_numbers = [20, 27, 31, 50]

mag, lanes, v = get_data(frame_numbers[0])
cont = get_avg_data(datafile, tslices[frame_numbers[0]])
cont=cont.T

lanes = lanes.T
mag = mag.T
vx, vy = v
vx = vx.T
vy = vy.T
vz = np.zeros(vx.shape)
vnorm = np.sqrt(vx**2 + vy**2)
# Grid
x, y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))
z = np.zeros(x.shape)

u = np.zeros(vx.shape+(2,))
v = np.zeros(vx.shape+(2,))
w = np.zeros(vx.shape+(2,))
u[...,0] = vx
u[...,1] = 0
v[...,0] = vy
v[...,1] = 0
w[...,0] = vz
w[...,1] = 0

# Visualization

#mlab.quiver3d(vx, vy, vz)

# data = np.repeat(np.linspace(0,1,512)[:,np.newaxis], 512, axis=1 )
#
# mlab.figure(bgcolor=(1,1,1))
# im = mlab.imshow(data, colormap='cool', transparent=True)
# cmap = im.module_manager.scalar_lut_manager.lut.table.to_array()
# cmap[:, -1] = np.linspace(0, 255, 256)
# im.module_manager.scalar_lut_manager.lut.table = cmap
# im.update_pipeline()
# mlab.view(0, 0)
#
# mlab.draw()
# mlab.view(0, 0)
# mlab.show()

# lanes3d = np.zeros(lanes.shape+(2,))
# lanes3d[...,0] = lanes.copy()
#
# fig = mlab.figure(size=(1000,1000))
# s1 = mlab.imshow(cont, colormap='Greys', extent = [0, x.shape[0]-1, 0, y.shape[0]-1, 0, 0])
# s1.module_manager.scalar_lut_manager.reverse_lut = True
# s1.update_pipeline()

# s2 = mlab.imshow(lanes, colormap='Blues', extent = [0, x.shape[0]-1, 0, y.shape[0]-1, 1, 0])
# cmap = s2.module_manager.scalar_lut_manager.lut.table.to_array()
# cmap[:, -1] = np.linspace(0, 255, 256)
# s2.module_manager.scalar_lut_manager.lut.table = cmap
# mlab.view(0, 0)
# s2.update_pipeline()


fig = mlab.figure(size=(1000,1000))

s1 = mlab.imshow(cont, colormap='Greys', extent = [0, x.shape[0]-1, 0, y.shape[0]-1, 0, 0])
s1.module_manager.scalar_lut_manager.reverse_lut = True
s1.update_pipeline()

s2 = mlab.imshow(lanes, colormap='Blues', extent = [0, x.shape[0]-1, 0, y.shape[0]-1, 1, 0])
cmap = s2.module_manager.scalar_lut_manager.lut.table.to_array()
cmap[:, -1] = np.linspace(0, 255, 256)
s2.module_manager.scalar_lut_manager.lut.table = cmap
mlab.view(0, 0)
s2.update_pipeline()

src = mlab.pipeline.vector_field(u, v, w, figure=fig)
magnitude = mlab.pipeline.extract_vector_norm(src)
vec = mlab.pipeline.vectors(magnitude, mask_points=40, scale_factor=10., line_width=4, colormap='Oranges')
vec.module_manager.vector_lut_manager.reverse_lut=True

vec.update_pipeline()
mlab.view(0, 0)
# cut = mlab.pipeline.vector_cut_plane(src, scale_factor=3, plane_orientation='z_axes')
# cut.implicit_plane.widget.normal_to_z_axis_ = 1
# cut.implicit_plane.widget.normal_to_z_axis = 1
# cut.implicit_plane.widget.tubing = 0
# cut.implicit_plane.plane.origin = np.array([256.5, 256.5, 0.0])

flow = mlab.pipeline.streamline(magnitude, seedtype='plane', seed_visible=False, seed_scale=0.5, seed_resolution=40, linetype='line', line_width=2, colormap='Oranges')
flow.seed.widget.normal_to_z_axis_ = 1
flow.seed.widget.normal_to_z_axis = 1
flow.seed.widget.center = np.array([256.5, 256.5, 1])
flow.seed.widget.origin = np.array([0, 0, 1])
flow.seed.widget.point1 = np.array([511, 0, 1])
flow.seed.widget.point2 = np.array([0, 511, 1])
flow.seed.widget.enabled = 1
#flow.seed.widget.resolution = 50
flow.stream_tracer.maximum_propagation = 30
flow.stream_tracer.integration_direction = 'both'
flow.module_manager.scalar_lut_manager.reverse_lut=True

flow.module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([ 0.85,  0.14 ])
flow.module_manager.scalar_lut_manager.scalar_bar_representation.position2 = np.array([ 0.1,  0.725])
flow.module_manager.scalar_lut_manager.show_scalar_bar = True
flow.module_manager.scalar_lut_manager.show_legend = True

flow.seed.update_pipeline()
flow.update_pipeline()
flow.seed.widget.enabled = 0
flow.seed.update_pipeline()
#flow.update_pipeline()


mlab.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/maya_fig_recent.png', magnification=4)
#mlab.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/maya_fig_recent.png')


mlab.draw()
mlab.show()


