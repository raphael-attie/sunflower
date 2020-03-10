from balltracking.balltrack import make_lanes


filename_read='directory/deepvel_output_simulation_series_00-31.fits'
img = fits.open(filename_read)
vv_ar_dvu = img[0].data
vx1_dvu_all = vv_ar_dvu[:,:,:,1]
vy1_dvu_all = vv_ar_dvu[:,:,:,0]
vx1_dvu = np.mean(vx1_dvu_all, axis=0)
vy1_dvu = np.mean(vy1_dvu_all, axis=0)


### Lanes parameters
nsteps = 50
maxstep = 4

lanes = make_lanes(vx, vy, nsteps, maxstep)