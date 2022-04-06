
PRO flct_drift_hmi_raphael, d1, d2, nb_images, sigma=sigma, BC=bc
; ==============================================================================
; Date: 1-Nov-2019
; Author: Raphael Attie (NASA GSFC)
; ------------------------------------------------------------------------------
; Description: Fourier-based Local correlation tracking for the reconstruction 
;              of plasma motions.
; Input: 
;    t1: drift directory start index (t1)
;    t2: drift directory end index (t2)
;    nb_images: number of images to consider in each series
;    sigma: size of gaussian kernel (in pixels)
; Optional keyword:
;    BC: enable FLCT internal bias correction
; Output: Transverse velocity fields written as fits files. 
; 
; ==============================================================================

; Parent data directory
data_dir = '/Users/rattie/Data/sanity_check/hmi_series/'
; Output
output_dir = '/Users/rattie/Data/sanity_check/hmi_series/output_FLCT_sigma'+STRTRIM(sigma,1)
; ------------------------------------------------------------------------------
; (1) Input data properties
; ------------------------------------------------------------------------------
; Dimensions of the images
nx=512L
ny=512L
; ------------------------------------------------------------------------------
; (2) Fourier-based Local Correlation Tracking
; ------------------------------------------------------------------------------

; Velocity fields
vx_flct_all=FLTARR(nx,ny,nb_images-1) ; nb_images-1 because consecutive images are used
vy_flct_all=FLTARR(nx,ny,nb_images-1)
vx_flct=FLTARR(nx,ny)
vy_flct=FLTARR(nx,ny)


IF KEYWORD_SET(BC) THEN output_dir = output_dir + '_bias_correction'

FILE_MKDIR, output_dir

; Loop over all drifts to test
FOR t=d1, d2 DO BEGIN
  drift_label=string(t, FORMAT='(I02)')
  filenames=FILE_SEARCH(data_dir + '/drift_'+drift_label+'/im_shifted_*')
  subdir = output_dir+'/drift_unfiltered_'+drift_label
  FILE_MKDIR, subdir
  ; Loop over all images used to compute the flow field
  FOR i=0, nb_images-2 DO BEGIN
    ; Continuum image #1
    img1=FLOAT(READFITS(filenames(i)))
    ; Continuum image #2
    img2=FLOAT(READFITS(filenames(i+1)))
    ; Input file
    vcimage2out, img1, img2, 'TemporaryInputFile.dat'
    ; FLCT computations
    IF KEYWORD_SET(BC) THEN BEGIN
    	PRINT, 'calling FLCT with bias correction'
    	spawn,'./flct TemporaryInputFile.dat TemporaryOutputFile.dat 1. 1. ' + strtrim(sigma,1) + ' -bc' ;+' -k 0.35' 
    ENDIF ELSE BEGIN
    	PRINT, 'calling FLCT without bias correction'
    	spawn,'./flct TemporaryInputFile.dat TemporaryOutputFile.dat 1. 1. ' + strtrim(sigma,1)  ;+' -k 0.35' 
    ENDELSE
    	 
    ; Velocity fields
    vcimage3in,vxtest,vytest,vmtest,'TemporaryOutputFile.dat'
    ; Output
    file_index=string(i, FORMAT='(I03)')
    WRITEFITS, subdir+'/FLCT_vx1_'+file_index+'.fits', vxtest
    WRITEFITS, subdir+'/FLCT_vy1_'+file_index+'.fits', vytest
  ENDFOR
ENDFOR

spawn,'rm TemporaryInputFile.dat'
spawn,'rm TemporaryOutputFile.dat'

end  
