import os
import glob
from pathlib import Path
import numpy as np
import fitstools
from balltracking import balltrack as blt

outputdir = Path(os.environ['DATA3'], 'sanity_check/stein_series/calibration3')
dv = 0.04
vx_rates = np.arange(-0.2, 0.21, dv)
vx_rates[int(len(vx_rates) / 2)] = 0
drift_rates = np.stack((vx_rates, np.zeros(len(vx_rates))), axis=1).tolist()

nframes = 60
# Read the fits files
imfiles = sorted(glob.glob(os.path.join(os.environ['DATA'], 'Ben/SteinSDO/SDO_int*.fits')))[0:nframes]
images = fitstools.fitsread(imfiles)

for i, vx_r in enumerate(vx_rates):
    drift_rate = [vx_r, 0]
    subdir = Path(outputdir, f'drift_{i:02d}')
    os.makedirs(subdir, exist_ok=True)
    filepaths = [str(Path(subdir, f'im_drifted_{k:02d}.fits')) for k in range(60)]
    _ = blt.create_drift_series(images, drift_rate, filepaths=filepaths)
