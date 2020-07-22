; Need to .compile write_slices_raphael, wherever it is. 
.compile write_slices_raphael

; Postel

filename = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/mtrack_20101126_220034_TAI_20101127_100034_TAI_Postel_continuum.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/idl_save_files/Postel_Intensity'
write_slices_raphael, filename, outputdir

filename = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/mtrack_20101126_220034_TAI_20101127_100034_TAI_Postel_Dopplergram.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/idl_save_files/Postel_Dopplergrams'
write_slices_raphael, filename, outputdir

filename = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/mtrack_20101126_220034_TAI_20101127_100034_TAI_Postel_magnetogram.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/idl_save_files/Postel_magnetograms'
write_slices_raphael, filename, outputdir

; Lambert Cylindrical

filename = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/mtrack_20101126_220034_TAI_20101127_100034_TAI_LambertCylindrical_continuum.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/idl_save_files/LC_Intensity'
write_slices_raphael, filename, outputdir

filename = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/mtrack_20101126_220034_TAI_20101127_100034_TAI_LambertCylindrical_Dopplergram.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/idl_save_files/LC_Dopplergrams'
write_slices_raphael, filename, outputdir

filename = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/mtrack_20101126_220034_TAI_20101127_100034_TAI_LambertCylindrical_magnetogram.fits'
outputdir = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/idl_save_files/LC_magnetograms'
write_slices_raphael, filename, outputdir
