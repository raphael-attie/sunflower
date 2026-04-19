import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path    
import balltracking.balltrack as blt
from functools import partial
from time import time
from optimization import inputs


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
        tr = inputs.cal_args['trange']
        if isinstance(dfiles, str) or isinstance(dfiles, Path):
            data = fits.getdata(dfiles)[tr[0]:tr[1]+1]
        else:
            data = [fits.getdata(f) for f in dfiles[tr[0]:tr[1]+1]]
            
        outdir_cal = inputs.cal_args['outputdir_cal']
        if outdir_cal is not None:
            for i, (drx, dry) in enumerate(zip(inputs.cal_args['vx_rates'], inputs.cal_args['vy_rates'])):
                blt.create_drift_series(data, drx, dry,
                                        outputdir=Path(outdir_cal, f'drift_{i:02d}'))

    calibrate_partial = partial(blt.full_calibration,
                                inputs.datafiles,
                                cal_args=inputs.cal_args,
                                cal_opt_args=inputs.cal_opt_args,
                                make_drift_images=False,
                                reprocess_bt=inputs.run_balltracking,
                                verbose=True)

    start = time()

    # Check if the DEBUG env var is set manually or if a debugger (e.g. PyCharm, VSCode) is attached
    # If so, run sequentially
    is_debug_env = os.getenv('DEBUG', '0').lower() in ('1', 'true', 'yes')
    is_debugger_attached = sys.gettrace() is not None or 'debugpy' in sys.modules
    if is_debug_env or is_debugger_attached:
        print("Running in DEBUG mode (sequential)")
        results = []
        for params in inputs.bt_params_list[0:2]:
            res = calibrate_partial(params)
            results.append(res)

        end = time()
        print(f'Elapsed time: {(end - start) / 60:0.2f} min')
        sys.exit(0)

    if inputs.use_multiprocessing:
        # For local execution on one node, tested with to 32 cpus.
        from concurrent.futures import ProcessPoolExecutor as PoolExec
    else:
        # For cluster execution, multiple nodes, with MPI
        from mpi4py.futures import MPIPoolExecutor as PoolExec

    with PoolExec(max_workers=inputs.ncpus) as executor:
        results = executor.map(calibrate_partial, inputs.bt_params_list)

    end = time()
    etime = (end - start)/60
    print(f'Elapsed time: {(end - start) / 60:0.2f} min')

# At the end of this parallel job, use "parameter_sweep_aggregation.py" to aggregate everything

