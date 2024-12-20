import sys, os
from pathlib import Path
sys.path.append(Path(os.environ['DEV'], 'sunflower').as_posix())
sys.path.append(Path(os.environ['DEV'], 'sunflower/balltracking').as_posix())
sys.path.append(Path(os.environ['DEV'], 'sunflower/balltracking/cython_modules').as_posix())
import fitstools
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from balltracking import mballtrack as mblt


plt.rcParams.update({'font.size': 12})
dpi = 168

DTYPE = np.float32

gamma = 0.75
def prep_data(image, gamma=0.5, norm_value=None, min_value=None):
    image2 = np.sqrt(abs(image))#**gamma
    image3 = image2.max() - image2
    if norm_value is None:
        norm_value = image3.std()
    surface = (image3 - image3.mean())/norm_value
    if min_value is None:
        min_value = surface.min()
    surface = surface - min_value
    return surface.copy(order='C').astype(DTYPE)


# Data prepped by Julia Clark in Heliocloud at /efs/jrclark/Saved FITS/Avg Maps/2021_03/
datadir = Path(os.environ['DATA'], 'HMI', 'Polar_Faculae', '2021_03')
datafiles = sorted(datadir.glob('*.fits'))
nfiles = len(datafiles)
print(nfiles, datafiles[0])

sample = fitstools.fitsread(datafiles[0].as_posix(), cube=False)
sampleNaN = sample.copy()
sampleNaN[sampleNaN == -1] = np.NaN
surfaceNaN = np.sqrt(np.abs(sampleNaN))
surfaceNaN = np.nanmax(surfaceNaN) - surfaceNaN
xlims = (500, 3500)
ylims = (100, 800)
norm_value = np.nanstd(surfaceNaN[ylims[0]:ylims[1], xlims[0]:xlims[1]])
surfaceNaN = (surfaceNaN - np.nanmean(surfaceNaN)) / norm_value
min_value = np.nanmin(surfaceNaN[ylims[0]:ylims[1], xlims[0]:xlims[1]])

print(norm_value, min_value)
prep_data_partial = partial(prep_data, gamma=gamma, norm_value=norm_value, min_value=min_value)

# Constrain initial positions to either Northern or Southern PFe

ballspacing = 10
threshold = 1.04

# To get the initial positions, we are working with the surface height, where the faculae will have negative intensities
# From the perspective of Magnetic Balltracking, that is a negative polarity,
# even though we are using intensity continuum images
polarity = 1
init_pos = mblt.get_local_extrema(sample, polarity, ballspacing, threshold, local_min=False, xlims=xlims, ylims=ylims)

tdx = 0.1
zdamping = 1.7
intsteps = 80
noise_level = 1.03

outputdir = Path(datadir, 'balltracking',
                 f'gamma{gamma}_noise{noise_level}_thresh{threshold}_tdxy{tdx}_z{zdamping}_intsteps{intsteps}')

mbt_dict = {"nt": 10,
            "rs": 2,
            "am": 1,
            "dp": 0.3,
            "tdx": tdx,
            "tdy": tdx,
            "zdamping": zdamping,
            "ballspacing": ballspacing,
            "intsteps": intsteps,
            "mag_thresh": 2,
            "noise_level": noise_level,
            "track_emergence": False,  # Tracking only the faculae already there
            "prep_function": prep_data_partial,
            "datafiles": datafiles,
            "init_pos": np.array(init_pos),
            "do_plots": 1,
            "axlims": [1800, 2400, 100, 350],
            "figsize": (12, 5),
            "fig_vmin_vmax": (1.01, 1.06),  # (1.02, 1.06),
            "astropy": False,
            "outputdir": outputdir,
            "verbose": True,
            "fig_dir": Path(outputdir, f'figures')}

# mbt_p, mbt_n = mblt.mballtrack_main(**mbt_dict)

# For the balltracking proper, we use polarity = 1 as the surface is mapped to the original image
# during bad-ball checking, which is definite positive on the surface, and excluding off-disk -1 flag value.
mbt = mblt.MBT(polarity=1, **mbt_dict)

mbt.track_all_frames()

