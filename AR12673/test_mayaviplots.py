#%gui qt
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

mag, lanes, vx, vy = get_data(frame_numbers[0])
cont = get_avg_data(datafile, tslices[frame_numbers[0]])


fig = mlab.figure(size=(1000,1000))

mayaplots.plot_cont(cont)
mayaplots.plot_lanes(lanes)
#mayaplots.plot_flow_vectors(vx, vy, fig)

#mlab.savefig('/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/figures/maya_fig_recent.png', magnification=4)


mlab.draw()
mlab.show()


