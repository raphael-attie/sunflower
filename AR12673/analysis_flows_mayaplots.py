#%gui qt
from importlib import reload
import glob, os
import numpy as np
import fitstools
import fitsio
import datetime
import balltracking.balltrack as blt
from mayavi import mlab
from AR12673 import mayaplots

def get_avg_data(file, tslice):
    samples = fitstools.fitsread(file, tslice=tslice).astype(np.float32)
    avg_data = samples.mean(axis=2)
    return avg_data


def get_vel(frame_number):
    vx = fitsio.read(vx_files[frame_number]).astype(np.float32)
    vy = fitsio.read(vy_files[frame_number]).astype(np.float32)
    return vx, vy


def get_data(frame_number):

    mag = get_avg_data(datafilem, tslices[frame_number])
    vx, vy = get_vel(frame_number)
    lanes = blt.make_lanes(vx, vy, nsteps, maxstep)

    return mag, lanes, vx, vy


def lanes_plot(frame_number, file_name, fov=None, do_print=False):

    mag, lanes, vx, vy = get_data(frame_numbers[0])
    cont = get_avg_data(datafile, tslices[frame_numbers[0]])

    offset = 1
    if fov is not None:
        lanes = lanes[fov[2]:fov[3], fov[0]:fov[1]]
        cont = cont[fov[2]:fov[3], fov[0]:fov[1]]
        mag = mag[fov[2]:fov[3], fov[0]:fov[1]]
        vx = vx[fov[2]+offset:fov[3]-offset, fov[0]+offset:fov[1]-offset]
        vy = vy[fov[2]+offset:fov[3]-offset, fov[0]+offset:fov[1]-offset]

    fig = mlab.figure(size=(500, 500), bgcolor=(1,1,1), fgcolor=(0.5, 0.5, 0.5))

    mag_plot = mayaplots.plot_im(mag, vmin=-200, vmax=200)
    lanes_plot = mayaplots.plot_lanes(lanes)

    #ax = mayaplots.add_axes_labels(fig, mag.T.shape, ranges=[0, 500, 0, 500, 0, 1])

    magnitude, vec = mayaplots.plot_flow_vectors(vx, vy, fig, offset=offset, reverse = False)

    flow = mayaplots.plot_streamlines(magnitude, vx, vy, reverse=False)


    #mayaplots.add_vector_colorbar(vec, reverse=True)
    mayaplots.add_scalar_colorbar(flow, 0, 600, 5)
    mlab.move(10, 0, 0)

    mlab.draw()
    if do_print:
        mlab.savefig(file_name + '_%d' % frame_number + '.png', magnification = 4)

    mlab.show()





datafile = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_continuum.fits'
datafilem = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mtrack_20170901_000000_TAI20170905_235959_LambertCylindrical_magnetogram.fits'
tracking_dir ='/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/python_balltracking'
fig_dir = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/'
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

# Velocity unit
# 1 px = 0.03 deg  solar radius = 6.957 x 10^8 m
# 1 px = 364.257 km
# 1 px / frame = 364257 / 45 m/s
px_meter = 0.03 * 3.1415/180 * 6.957e8
ms_unit = px_meter / 45

# Grab data
frame_numbers = [20, 27, 31, 50]

# for frame_nb in frame_numbers:
fov=(225-50, 225+50, 176-50, 176+50)
lanes_plot(frame_numbers[0], os.path.join(fig_dir, 'lanes_streamlines_res12'), fov=fov, do_print=True)