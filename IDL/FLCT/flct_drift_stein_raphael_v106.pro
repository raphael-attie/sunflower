; ==============================================================================
; Date: 20-Feb-2018
; Author: Benoit Tremblay (Université de Montréal)
; Project: DeepVel
; ------------------------------------------------------------------------------
; Description: Fourier-based Local correlation tracking for the reconstruction 
;              of plasma motions.
; Input: sigma (in pixels)
; Output: Transverse velocity fields in 
; 
; ==============================================================================
PRO flct_drift_stein_raphael_v106, t1, t2, sigma=sigma, BC=bc
; ==============================================================================

; //////////////////////////////////////////////////////////////////////////////
; (0) Original simulation parameters (DO NOT CHANGE !!!)
; //////////////////////////////////////////////////////////////////////////////

; Dimensions of the Nordlund & Stein (2012) simulation
nx_stein=1008L
ny_stein=1008L
; km per pixel
dx_stein=96D0
dy_stein=96D0
; Lengths
lx_stein=FLOAT(nx_stein)*dx_stein
ly_stein=FLOAT(ny_stein)*dy_stein
; Timestep
dt_stein=60D0
; Nb. of optical depths
nz=3L

; Dimensions of the simulation once it has been resampled to the SDO/HMI resolution
deepvel_dimsdo, nx_stein, ny_stein, dx_stein, dy_stein, nx=nx, ny=ny
nx_sdo=nx ; should be 263 px
ny_sdo=ny
; km per pixel
dx_sdo=lx_stein/FLOAT(nx_sdo)
dy_sdo=ly_stein/FLOAT(ny_sdo)


; SDO resolution
; Arcsec/pix
cdelt1=0.504365D0
cdelt2=0.504365D0
; R -> m
rsun_ref=6.96000D8
; R -> km
rkm=rsun_ref/1D3
; R -> arcsec
rsun_obs=953.288D0
; R -> pixel
rsun=round(rsun_obs/cdelt1)
; Nb. m / pix
deltas=rkm*1E3/rsun_obs*cdelt1

; ------------------------------------------------------------------------------
; (1) Input data properties
; ------------------------------------------------------------------------------
; Dimensions once padding is removed
nx=263L
ny=263L
; m per pixel
dx=dx_sdo*1E3
dy=dy_sdo*1E3
; Number of drifts to test
nb_tests=11
nb_images=30

; ------------------------------------------------------------------------------
; (2) Fourier-based Local Correlation Tracking
; ------------------------------------------------------------------------------

; Velocity fields
vx_flct_all=FLTARR(nx,ny,nb_images-1) ; nb_images-1 because consecutive images are used
vy_flct_all=FLTARR(nx,ny,nb_images-1)
vx_flct=FLTARR(nx,ny)
vy_flct=FLTARR(nx,ny)

; Parent data directory
data_dir = '/Users/rattie/Data/sanity_check/stein_series/'
; Output

output_dir = '/Users/rattie/Data/sanity_check/stein_series_FLCT_106/output_FLCT_sigma'+STRTRIM(sigma,1)
IF KEYWORD_SET(BC) THEN output_dir = output_dir + '_bias_correction'

FILE_MKDIR, output_dir

; Loop over all drifts to test
;FOR t=0, nb_tests-1 DO BEGIN
FOR t=t1, t2 DO BEGIN
  drift_label=string(t, FORMAT='(I02)')
  filenames=FILE_SEARCH(data_dir + '/drift_'+drift_label+'/im_shifted_*')
  subdir = output_dir+'/drift_unfiltered_'+drift_label
  FILE_MKDIR, subdir
  ; Loop over all images used to compute the flow field
  FOR i=0, nb_images-2 DO BEGIN
    ; Continuum image #1
    img1=DOUBLE(READFITS(filenames(i)))
    ; Continuum image #2
    img2=DOUBLE(READFITS(filenames(i+1)))
    ; Input file
    vcimage2out, img1, img2, 'TemporaryInputFile.dat'
    ; FLCT computations
    IF KEYWORD_SET(BC) THEN BEGIN
    	PRINT, 'calling FLCT with bias correction'
    	spawn,'./flct106 TemporaryInputFile.dat TemporaryOutputFile.dat 1. 1. ' + strtrim(sigma,1) + ' -bc' ;+' -k 0.35' 
    ENDIF ELSE BEGIN
    	PRINT, 'calling FLCT without bias correction'
    	spawn,'./flct106 TemporaryInputFile.dat TemporaryOutputFile.dat 1. 1. ' + strtrim(sigma,1)  ;+' -k 0.35' 
    ENDELSE
    	
    
    ; Velocity fields
    vcimage3in,vxtest,vytest,vmtest,'TemporaryOutputFile.dat'
    vx_flct_all(*,*,i)=vxtest
    vy_flct_all(*,*,i)=vytest
    ; Output
    file_index=string(i, FORMAT='(I03)')
    WRITEFITS, subdir+'/FLCT_vx1_'+file_index+'.fits', vx_flct_all(*,*,i)
    WRITEFITS, subdir+'/FLCT_vy1_'+file_index+'.fits', vy_flct_all(*,*,i)
  ENDFOR
  vx_flct=MEAN(vx_flct_all, DIMENSION=3)
  vy_flct=MEAN(vy_flct_all, DIMENSION=3)
  ; Output (average)
  WRITEFITS, subdir+'/FLCT_vx1_000-' + file_index + '.fits', vx_flct
  WRITEFITS, subdir+'/FLCT_vy1_000-' + file_index + '.fits', vy_flct
ENDFOR

spawn,'rm TemporaryInputFile.dat'
spawn,'rm TemporaryOutputFile.dat'

end  
