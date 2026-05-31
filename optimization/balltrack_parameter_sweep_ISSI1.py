import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path    
import balltracking.balltrack as blt
from functools import partial
from time import time
from optimization import inputs_ISSI1 as inputs
import shutil


def run_worker_calibration(params):
    """
    Wrapper function to execute calibration for a single parameter set
    while redirecting stdout and stderr to a parameter-index-specific log file.
    """
    import sys
    import traceback
    
    index = params.get('index', 0)
    logs_dir = inputs.outputdir / 'worker_logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / f'worker_{index:05d}.log'

    with open(log_file_path, 'w', buffering=1) as f:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = f
        sys.stderr = f
        try:
            print(f"==================================================")
            print(f" Starting Calibration for Parameter Index: {index}")
            print(f"==================================================")
            print(f"Parameters: {params}\n")
            
            res = blt.full_calibration(
                inputs.datafiles,
                params,
                cal_args=inputs.cal_args,
                cal_opt_args=inputs.cal_opt_args,
                make_drift_images=False,
                reprocess_bt=inputs.run_balltracking,
                verbose=True
            )
            
            print(f"\n==================================================")
            print(f" Calibration Completed Successfully for Index: {index}")
            print(f"==================================================")
            return res
        except Exception as e:
            print(f"\n==================================================")
            print(f" ERROR: Calibration Failed for Index: {index}")
            print(f"==================================================")
            traceback.print_exc()
            raise e
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == '__main__':

    # the multiprocessing start method can only bet set once
    if inputs.use_multiprocessing:
        import multiprocessing
        multiprocessing.set_start_method('spawn')
    # TODO: check directory content to not overwrite files that will have the same index
    inputs.outputdir.mkdir(parents=True, exist_ok=True)

    if inputs.make_drift_images:
        print('Pre-generating drift images synchronously to avoid race conditions...')
        from astropy.io import fits
        dfiles = inputs.datafiles
        tr = inputs.bt_params['trange']
        if isinstance(dfiles, str) or isinstance(dfiles, Path):
            data = fits.getdata(dfiles)[tr[0]:tr[1]+1]
        else:
            data = [fits.getdata(f) for f in dfiles[tr[0]:tr[1]+1]]
            
        outdir_cal = inputs.cal_args['outputdir_cal']
        if outdir_cal is not None:
            for i, (drx, dry) in enumerate(zip(inputs.cal_args['vx_rates'], inputs.cal_args['vy_rates'])):
                blt.create_drift_series(data, drx, dry,
                                        outputdir=Path(outdir_cal, f'drift_{i:02d}'))

    start = time()

    # Check if the DEBUG env var is set manually or if a debugger (e.g. PyCharm, VSCode) is attached
    # If so, run sequentially
    is_debug_env = os.getenv('DEBUG', '0').lower() in ('1', 'true', 'yes')
    is_debugger_attached = sys.gettrace() is not None or 'debugpy' in sys.modules
    if is_debug_env or is_debugger_attached:
        print("Running in DEBUG mode (sequential)")
        calibrate_partial = partial(blt.full_calibration,
                                    inputs.datafiles,
                                    cal_args=inputs.cal_args,
                                    cal_opt_args=inputs.cal_opt_args,
                                    make_drift_images=False,
                                    reprocess_bt=inputs.run_balltracking,
                                    verbose=True)
        results = []
        for params in inputs.bt_params_list[0:2]:
            res = calibrate_partial(params)
            results.append(res)

        end = time()
        print(f'Elapsed time: {(end - start) / 60:0.2f} min')
        sys.exit(0)

    if inputs.use_multiprocessing:
        # For local execution on one node, tested with up to 32 cpus.
        from concurrent.futures import ProcessPoolExecutor as PoolExec
        with PoolExec(max_workers=inputs.ncpus) as executor:
            results = list(executor.map(run_worker_calibration, inputs.bt_params_list))
    else:
        # For cluster execution, multiple nodes, with MPI
        from mpi4py.futures import MPIPoolExecutor as PoolExec
        with PoolExec() as executor:  # Let mpi4py.futures automatically manage the pre-spawned worker pool
            results = executor.map(run_worker_calibration, inputs.bt_params_list)

    end = time()
    etime = (end - start)/60
    print(f'Elapsed time: {(end - start) / 60:0.2f} min')
    print('Moving output files to subdirectories...')
    mean_vel_dir = inputs.outputdir / 'mean_velocity_files'
    param_sweep_dir = inputs.outputdir / 'param_sweep_files'
    
    mean_vel_dir.mkdir(exist_ok=True)
    param_sweep_dir.mkdir(exist_ok=True)
    
    for f in inputs.outputdir.glob('mean_velocity*.npz'):
        shutil.move(str(f), str(mean_vel_dir / f.name))
        
    for f in inputs.outputdir.glob('param_sweep_*.csv'):
        shutil.move(str(f), str(param_sweep_dir / f.name))


