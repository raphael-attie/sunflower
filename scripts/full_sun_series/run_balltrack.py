import balltracking.balltrack as blt
import fitstools
from scripts.full_sun_series import inputs
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
matplotlib.use('agg')

if __name__ == "__main__":
    # the multiprocessing start method can only bet set once
    if inputs.use_multiprocessing:
        import multiprocessing
        try:
            multiprocessing.set_start_method('spawn', force=True)
            print('spawned')
        except RuntimeError:
            pass


    if inputs.run_calibration:
        if inputs.make_drift_images:
            # Create the drift images but do not keep them in memory to avoid running out of memory for big series
            cal_datafiles = inputs.datafiles[inputs.cal_args['trange'][0]:inputs.cal_args['trange'][1]]
            images_select = fitstools.fitsread(cal_datafiles)
            if inputs.cal_args['outputdir_cal'] is not None:
                _, drift_dirs = zip(*[blt.create_drift_series(images_select, drx, dry,
                                                              outputdir=Path(inputs.cal_args['outputdir_cal'],
                                                                             f'drift_{i:02d}'))
                                      for i, (drx, dry) in
                                      enumerate(zip(inputs.cal_args['vx_rates'], inputs.cal_args['vy_rates']))])

        _ = blt.full_calibration(inputs.bt_params, inputs.cal_args, inputs.cal_opt_args,
                                 reprocess_bt=inputs.reprocess_bt, verbose=True)

    if inputs.run_balltracking:
        print('Running Balltracking...')
        _, _ = blt.balltrack_main_hmi(inputs.bt_params, inputs.outputdir,
                                      datafiles=inputs.datafiles, ncores=4)

    # Make euler flows
    df_fit = (pd.read_csv(Path(inputs.cal_args['outputdir_cal'], 'param_sweep_00000.csv')).
              query('kernel=="gaussian"'))
    cal_top = df_fit['p_top_0'].values[0]
    cal_bottom = df_fit['p_bot_0'].values[0]

    # Load ballpos arrays
    arr = np.load(Path(inputs.outputdir, 'ballpos.npz'))
    ballpos_top, ballpos_bottom = [arr['ballpos_top'], arr['ballpos_bottom']]

    sample = fits.getdata(inputs.datafiles[0])
    dims = sample.shape
    fwhm = inputs.cal_args['fwhm']
    tranges = [[i, i + inputs.navg] for i in
               range(inputs.bt_params['trange'][0], inputs.bt_params['trange'][1]-inputs.navg, inputs.dt)]
    headers = []
    for tr in tranges:
        print(tr)
        # If RICE compressed, getheader() should have the optional argument "1" for the right hduC
        headers.append(fits.getheader(inputs.datafiles[tr[0] + inputs.navg//2], 1))



    vxs, vys, lanesl, lanes_sup = blt.make_euler_vel_lanes_series(ballpos_top, ballpos_bottom, cal_top, cal_bottom,
                                                                  dims, fwhm, tranges,
                                                                  nsteps=inputs.nsteps,
                                                                  outputdir=inputs.outputdir,
                                                                  headers=headers)

    vx_avg, vy_avg, lanes_avg = blt.make_euler_vel_lanes(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, fwhm,
                                                         nsteps=inputs.nsteps,
                                                         outputdir=inputs.outputdir,
                                                         header=headers[len(headers) // 2])

    for i, (vx, vy) in enumerate(zip(vxs, vys)):

        lanes = lanesl[i]
        plt.figure(figsize=(16, 10))
        plt.imshow(lanes, origin='lower', cmap='gray_r')
        plt.xlabel('x [px] 1 px ~ 0.0301 deg.')
        plt.ylabel('y [px] 1 px ~ 0.0301 deg.')
        plt.title(f'convection boundary lanes (nsteps={inputs.nsteps})')
        plt.tight_layout()
        plt.savefig(Path(inputs.outputdir, f'lanes_navg{inputs.navg}_nsteps{inputs.nsteps}_{i:03d}'), dpi=150)
        plt.close('all')

    plt.figure(figsize=(16, 10))
    plt.imshow(lanes_sup, origin='lower', cmap='gray_r')
    plt.xlabel('x [px] 1 px ~ 0.0301 deg.')
    plt.ylabel('y [px] 1 px ~ 0.0301 deg.')
    plt.title(f'running average boundary lanes (nsteps={inputs.nsteps})')
    plt.tight_layout()
    plt.savefig(Path(inputs.outputdir, f'lanes_navg{inputs.navg}_nsteps{inputs.nsteps}_run_avg'), dpi=150)
    plt.close('all')

    plt.figure(figsize=(16, 10))
    plt.imshow(lanes_avg, origin='lower', cmap='gray_r')
    plt.xlabel('x [px] 1 px ~ 0.0301 deg.')
    plt.ylabel('y [px] 1 px ~ 0.0301 deg.')
    plt.title(f'Averaged convection boundary lanes (nsteps={inputs.nsteps})')
    plt.tight_layout()
    plt.savefig(Path(inputs.outputdir, f'lanes_nsteps{inputs.nsteps}_tALL'), dpi=150)
    plt.close('all')
