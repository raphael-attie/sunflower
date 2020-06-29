; Parent data directory
data_dir = '/Users/rattie/Data/Ben/SteinSDO/'
; Output
output_dir = data_dir + 'FLCT_Raphael/output_FLCT_sigma'+STRTRIM(sigma,1)
;padding size
pad = 10
nx = 263
ny = 263
fwhm = 7
; get FLCT velocities
vxf= '/Users/rattie/Data/Ben/SteinSDO/FLCT_Raphael/output_FLCT_sigma7/FLCT_vx1_000-028.fits'
vyf= '/Users/rattie/Data/Ben/SteinSDO/FLCT_Raphael/output_FLCT_sigma7/FLCT_vy1_000-028.fits'

vx_flct = READFITS(vxf)
vy_flct = READFITS(vyf)

; get Stein velocities
vsxf=FILE_SEARCH(data_dir + 'SDO_vx*.fits')
vsyf=FILE_SEARCH(data_dir + 'SDO_vy*.fits')
nb_fields=29
vsxf = vsxf(0:nb_fields-1)
vsyf = vsyf(0:nb_fields-1)

; Number of files to track

; ------------------------------------------------------------------------------
; (2) Fourier-based Local Correlation Tracking
; ------------------------------------------------------------------------------
; Velocity fields
vxs=FLTARR(nx,ny,nb_fields) 
vys=FLTARR(nx,ny,nb_fields)

FOR i=0, nb_fields-1 DO BEGIN &$
	vxs(*, *, i) = READFITS(vsxf(i)) &$
	vys(*, *, i) = READFITS(vsyf(i)) &$
ENDFOR
; Take the average
vxsm=SMOOTH(MEAN(vxs, DIMENSION=3), fwhm)
vysm=SMOOTH(MEAN(vys, DIMENSION=3), fwhm)

field1x = vxsm(pad:nx-1-pad, pad:ny-1-pad)
field1y = vysm(pad:nx-1-pad, pad:ny-1-pad)

field2x = vx_flct(pad:nx-1-pad, pad:ny-1-pad)
field2y = vy_flct(pad:nx-1-pad, pad:ny-1-pad)

c = TOTAL(field1x*field2x + field1y*field2y)/SQRT(TOTAL(field1x^2 + field1y^2)*TOTAL(field2x^2 + field2y^2))


