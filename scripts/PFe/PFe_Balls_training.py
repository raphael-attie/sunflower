import sys, os
from pathlib import Path
sys.path.append(Path(os.environ['DEV'], 'sunflower').as_posix())
sys.path.append(Path(os.environ['DEV'], 'sunflower/balltracking').as_posix())
sys.path.append(Path(os.environ['DEV'], 'sunflower/balltracking/cython_modules').as_posix())
import fitstools
from functools import partial
import numpy as np
from balltracking import mballtrack as mblt
import pandas as pd

DTYPE = np.float32

def prep_data(image, gamma=0.5, norm_value=None, min_value=None):
    image2 = abs(image)**gamma
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

outputdir = Path(datadir, 'balltracking')
sample = fitstools.fitsread(datafiles[0].as_posix(), cube=False)

noise_level = 1.03
ballspacing = 10
intsteps = 80
xlims = (500, 3500)
ylims = (100, 800)

polarity = 1
threshold = 1.04
init_pos = mblt.get_local_extrema(sample, polarity, ballspacing, threshold, local_min=False, xlims=xlims, ylims=ylims)


gammas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
tdxs = np.arange(0, 2.1, 0.1)[1:]
zdampings = np.arange(0, 2.1, 0.1)[1:]

df = pd.DataFrame(columns=["gamma", "tdxy", "zdamping", "nbadballs"])

for g, gamma in enumerate(gammas):
    for t, tdx in enumerate(tdxs):
        for z, zdamping in enumerate(zdampings):
            print(f'g={g}/{len(gammas)}, t={t}/{len(tdxs)}, z={z}/{len(zdampings)}')

            prep_data_partial = partial(prep_data, gamma=gamma)
            # To get the initial positions, we are working with the surface height, where the faculae will have negative intensities
            # From the perspective of Magnetic Balltracking, that is a negative polarity,
            # even though we are using intensity continuum images

            mbt_dict = {"nt": 3,
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
                        "do_plots": 0,
                        "axlims": [1800, 2400, 100, 350],
                        "figsize": (12, 5),
                        "fig_vmin_vmax": (1.01, 1.06),  # (1.02, 1.06),
                        "astropy": False,
                        "verbose": False,
                        "fig_dir": Path(datadir, f'figures/with_surface_i_gamma{gamma}_noise{noise_level}_thresh{threshold}_tdxy{tdx}_z{zdamping}_intsteps{intsteps}')}

            # mbt_p, mbt_n = mblt.mballtrack_main(**mbt_dict)

            # For the balltracking proper, we use polarity = 1 as the surface is mapped to the original image
            # during bad-ball checking, which is definite positive on the surface, and excluding off-disk -1 flag value.
            mbt = mblt.MBT(polarity=1, **mbt_dict)

            mbt.track_all_frames()

            opt_params = {
                "gamma": gamma,
                "tdxy": tdx,
                "zdamping": zdamping,
                "nbadballs": mbt.nbadballs
            }
            df.loc[len(df)] = opt_params

df.to_csv('opt_params.csv', index=False)


