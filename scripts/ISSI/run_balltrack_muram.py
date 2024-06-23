import balltracking.balltrack as blt
from scripts.ISSI import inputs
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from os import environ
matplotlib.use('agg')


def read_iout(filepath, tslice=None, **kwargs):
    if isinstance(filepath, list):
        filepath = filepath[tslice]
    data = np.fromfile(filepath, dtype=np.float32)
    shape = tuple(data[1:3].astype(int))
    # time = data[3]
    intensitygram = data[4:].reshape([shape[1], shape[0]]).swapaxes(0, 1)
    return intensitygram


if __name__ == "__main__":

    # set the number of threads for many common libraries
    # N_THREADS = '1'
    # environ['OMP_NUM_THREADS'] = N_THREADS
    # environ['OPENBLAS_NUM_THREADS'] = N_THREADS
    # environ['MKL_NUM_THREADS'] = N_THREADS
    # environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
    # environ['NUMEXPR_NUM_THREADS'] = N_THREADS
    # the multiprocessing start method can only bet set once
    if inputs.use_multiprocessing:
        import multiprocessing
        try:
            multiprocessing.set_start_method('spawn', force=True)
            print('spawned')
        except RuntimeError:
            print('could not set the multiprocessing start method')
            pass

    if inputs.run_calibration:
        print('Running calibration...')
        _ = blt.full_calibration(inputs.datafiles, inputs.bt_params, inputs.cal_args, inputs.cal_opt_args,
                                 make_drift_images=inputs.make_drift_images, reprocess_bt=inputs.reprocess_bt,
                                 verbose=True, image_reader=read_iout)
        print('Calibration done.')

    if inputs.run_balltracking:
        print('Running balltracking...')
        _, _ = blt.balltrack_main_hmi(inputs.bt_params, inputs.outputdir, datafiles=inputs.datafiles, ncores=1,
                                      image_reader=read_iout)
        print('Balltracking done')

    # Load the file created during the calibration
    calibration_file = Path(inputs.cal_args['outputdir_cal'], 'param_sweep_00000.csv')
    # Make calibrated euler flows
    v_series, v_avg = blt.calibrate_flows(inputs.datafiles, calibration_file, inputs.outputdir, inputs.maps_params)

    # Quick look on the flow maps
    plt.figure()
    plt.imshow(v_avg['vx_avg'], origin='lower', cmap='gray')
    plt.title('vx_avg')
    plt.colorbar()
    plt.savefig(Path(inputs.outputdir, 'quicklook_vx_avg.png'))

    plt.figure()
    plt.imshow(v_avg['vy_avg'], origin='lower', cmap='gray')
    plt.title('vy_avg')
    plt.colorbar()
    plt.savefig(Path(inputs.outputdir, 'quicklook_vy_avg.png'))

    plt.figure()
    plt.imshow(v_avg['lanes_avg'], origin='lower', cmap='afmhot', vmin=0, vmax=12)
    plt.title('lanes_avg')
    plt.colorbar()
    plt.savefig(Path(inputs.outputdir, 'quicklook_lanes_avg.png'))

    plt.figure()
    plt.imshow(v_series['run_avg_lanes'], origin='lower', cmap='afmhot', vmin=0, vmax=12)
    plt.title('run_avg_lanes')
    plt.colorbar()
    plt.savefig(Path(inputs.outputdir, 'quicklook_run_lanes_avg.png'))