import balltracking.balltrack as blt
from functools import partial
from time import time
import inputs

if __name__ == '__main__':

    # the multiprocessing start method can only bet set once
    if inputs.use_multiprocessing:
        import multiprocessing
        multiprocessing.set_start_method('spawn')
    # ray.init(num_cpus=32)
    # TODO: check directory content to not overwrite files that will have the same index
    inputs.outputdir.mkdir(parents=True, exist_ok=True)

    # Prepare partial function for parallel pool & map, which can accept only the list of parameters as its argument
    calibrate_partial = partial(blt.full_calibration,
                                drift_rates=inputs.drift_rates,
                                drift_dirs=inputs.drift_dirs,
                                read_drift_images=inputs.read_drift_images,
                                trange=inputs.trange,
                                fov_slices=inputs.fov_slices,
                                reprocess_bt=inputs.reprocess_bt,
                                outputdir=inputs.outputdir,
                                fwhm=inputs.fwhm,
                                dims=inputs.dims,
                                outputdir2=inputs.outputdir_cal,
                                save_ballpos_list=inputs.save_ballpos_list,
                                verbose=inputs.verbose)

    start = time()

    if inputs.use_multiprocessing:
        from concurrent.futures import ProcessPoolExecutor as PoolExec
    else:
        from mpi4py.futures import MPIPoolExecutor as PoolExec

    with PoolExec() as executor:
        results = executor.map(calibrate_partial, inputs.bt_params_list)

    end = time()
    etime = (end - start)/60
    print(f'Elapsed time: {etime:0.2f} min')

# At the end of this parallel job, use "parameter_sweep_velocity_calibration.py" to aggregate everything

