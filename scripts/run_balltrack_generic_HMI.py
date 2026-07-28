import balltracking.balltrack as blt
from pathlib import Path
from time import time
import matplotlib
import matplotlib.pyplot as plt
import sys

from scripts import inputs_generic_HMI as inputs

import faulthandler

faulthandler.enable()
# Optionally dump traceback on SIGSEGV to a file:
faulthandler.enable(file=open('faulthandler1.log', 'w'))

matplotlib.use('agg')

if __name__ == "__main__":
    # Owarn when no data directory argument was passed (the default will be used)
    if len(sys.argv) < 2:
        print(f'No data directory argument provided. Using default: {inputs.datadir}')
    print(f'Using data directory: {inputs.datadir}')

    # the multiprocessing start method can only bet set once
    if inputs.use_multiprocessing:
        import multiprocessing
        try:
            multiprocessing.set_start_method('spawn', force=True)
            print('spawned')
        except RuntimeError:
            print('could not set the multiprocessing start method')
            pass

    timings = {}  # store wall-clock times (in seconds) per stage
    # Total wall clock for the whole script
    t_total_start = time()

    if inputs.run_calibration:
        t0 = time()
        _ = blt.full_calibration(inputs.datafiles, inputs.bt_params, inputs.cal_args, inputs.cal_opt_args,
                                 make_drift_images=inputs.make_drift_images,
                                 reprocess_bt=inputs.run_balltracking,
                                 verbose=True)

        timings['full_calibration'] = time() - t0
        print(f'calibration finished. Elapsed: {timings["full_calibration"] / 60:0.2f} min '
              f'({timings["full_calibration"]:0.2f} s)')

        print('calibration finished.')

    if inputs.run_balltracking:
        if inputs.use_multiprocessing:
            ncores = 4
        else:
            ncores = 1

        t0 = time()
        _, _ = blt.balltrack_main_hmi(inputs.bt_params, inputs.outputdir, datafiles=inputs.datafiles, ncores=ncores)
        timings['balltrack_main_hmi'] = time() - t0
        print(f'balltracking finished. Elapsed: {timings["balltrack_main_hmi"] / 60:0.2f} min '
              f'({timings["balltrack_main_hmi"]:0.2f} s)')

    elif not Path(inputs.outputdir, 'ballpos.npz').exists():
        raise FileNotFoundError(
            f'{inputs.outputdir}/ballpos.npz is missing: run_balltracking is False but no prior ballpos.npz was found.'
        )

    # Load the file created during the calibration
    calibration_file = Path(inputs.cal_args['outputdir_cal'], 'param_sweep_00000.csv')
    if not calibration_file.exists():
        raise FileNotFoundError(f'Calibration CSV not found: {calibration_file}. Did run_calibration run?')
    # Make calibrated euler flows
    t0 = time()
    v_series, v_avg = blt.calibrate_flows(inputs.datafiles, calibration_file, inputs.outputdir, inputs.maps_params)
    timings['calibrate_flows'] = time() - t0
    print(f'calibrate_flows finished. Elapsed: {timings["calibrate_flows"] / 60:0.2f} min '
          f'({timings["calibrate_flows"]:0.2f} s)')

    # Quick look on the flow maps
    plt.figure()
    plt.imshow(v_avg['vx_avg'], origin='lower', cmap='gray')
    plt.savefig(Path(inputs.outputdir, 'quicklook_vx_avg.png'))

    plt.figure()
    plt.imshow(v_avg['vy_avg'], origin='lower', cmap='gray')
    plt.savefig(Path(inputs.outputdir, 'quicklook_vy_avg.png'))

    plt.figure()
    plt.imshow(v_avg['lanes_avg'], origin='lower', cmap='Blues')
    plt.savefig(Path(inputs.outputdir, 'quicklook_lanes_avg.png'))

    plt.figure()
    plt.imshow(v_series['run_avg_lanes'], origin='lower', cmap='Blues')
    plt.savefig(Path(inputs.outputdir, 'quicklook_run_lanes_avg.png'))


    # Summary of wall-clock times
    total_elapsed = time() - t_total_start
    timings['total'] = total_elapsed

    print('====================================================')
    print('Wall-clock timing summary:')
    for stage, secs in timings.items():
        print(f'  {stage:20s}: {secs / 60:8.2f} min  ({secs:10.2f} s)')
    print('====================================================')

    # Persist the timings to a text file next to the outputs for later reference
    timings_file = Path(inputs.outputdir, 'timings.txt')
    with open(timings_file, 'w') as f:
        f.write('Wall-clock timing summary (seconds and minutes):\n')
        for stage, secs in timings.items():
            f.write(f'{stage:20s}: {secs:10.2f} s   ({secs / 60:8.2f} min)\n')
    print(f'Timings saved to {timings_file}')
