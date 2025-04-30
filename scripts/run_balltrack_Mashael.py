import balltracking.balltrack as blt
from scripts import inputs_Mashael as inputs
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('agg')

if __name__ == "__main__":
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
        _ = blt.full_calibration(inputs.datafiles, inputs.bt_params, inputs.cal_args, inputs.cal_opt_args,
                                 datacube=True, verbose=True)

    if inputs.run_balltracking:
        _, _ = blt.balltrack_main_hmi(inputs.bt_params, inputs.outputdir, datafiles=inputs.datafiles, ncores=4)

    # Load the file created during the calibration
    calibration_file = Path(inputs.cal_args['outputdir_cal'], 'param_sweep_00000.csv')
    # Make calibrated euler flows
    v_series, v_avg = blt.calibrate_flows(inputs.datafiles, calibration_file, inputs.outputdir, inputs.maps_params)

    # Quick look on the flow maps
    plt.figure()
    plt.imshow(v_avg['vx_avg'], origin='lower', cmap='gray')
    plt.savefig(Path(inputs.outputdir, 'quicklook_vx_avg.png'))

    plt.figure()
    plt.imshow(v_avg['vy_avg'], origin='lower', cmap='gray')
    plt.savefig(Path(inputs.outputdir, 'quicklook_vy_avg.png'))

    plt.figure()
    plt.imshow(v_avg['lanes_avg'], origin='lower', cmap='gray')
    plt.savefig(Path(inputs.outputdir, 'quicklook_lanes_avg.png'))

    plt.figure()
    plt.imshow(v_series['run_avg_lanes'], origin='lower', cmap='gray')
    plt.savefig(Path(inputs.outputdir, 'quicklook_run_lanes_avg.png'))