import balltracking.balltrack as blt
import calibration_config_template as config
import numpy as np
from pathlib import Path

cal = blt.Calibrator(config.images,
                     config.vx_rates,
                     config.trange,
                     config.bt_params_top,
                     config.bt_params_bottom,
                     config.outputdir,
                     drift_dirs=config.drift_dirs,
                     read_drift_images=config.read_drift_images,
                     save_ballpos_list=config.save_ballpos_list,
                     nthreads=config.nthreads,
                     reprocess_existing=config.reprocess_existing,
                     verbose=config.verbose)

ballpos_top_list, ballpos_bottom_list = cal.balltrack_all_rates()

ftop = cal.fit(ballpos_top_list)
fbot = cal.fit(ballpos_bottom_list)

print('linear fit parameters (top): ', ftop[0])
print('Bottom-side linear fit parameters (bottom): ', fbot[0])

np.savez(Path(config.outputdir, 'calibration.npz'),
         ptop=ftop[0],
         pbot=fbot[0])
