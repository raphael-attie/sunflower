PRO write_slices_raphael, filename, outputdir
;+
; Name: write_slices_raphael
; 
; Purpose: Write frame for each time step of a 3D fits files (data cube). Mac / Linux only
;
; Input Parameters:
;
; 	filename - Path to file, can be relative to current directory or absolute path. 
; 	outputdir - Directory where each fits file will be written. Trailing or no trailing '/' will work. 
;
; Calling Sequence:
;	write_slices_raphael, filename, outputdir
;
;-

;Make sure the outputdir has a trailing /
if not strmatch(strmid(outputdir, strlen(outputdir)-1, 1), '/') then outputdir = strcompress(outputdir + '/', /remove_all)

; Get path to solarsoft
lib_path = getenv('SSW')+'/vobs/ontology/binaries/darwin_x86_64/

; Load the data and convert header to an IDL structure
data = fitsio_read_image(filename, header, so_path= lib_path)
head_struct = fitshead2struct(header)

; Make sure we have the basename in case of absolute file path, for string processing later
basename = file_basename(filename)
; Extract starttime (the one in the header require more string processing)
start_time = strmid(basename, 7, 15)

; Prepare the file names depending on the nature of the data: intensity, magnetograms, dopplergrams and projection: Postel or LamberCylindrical
segment = 'unknown'
if strmatch(filename, '*continuum*') then segment = 'i' else $
if strmatch(filename, '*magnetogram*') then segment = 'm' else $
if strmatch(filename, '*Dopplergram*') then segment = 'v'

projection = 'unknown'
if strmatch(head_struct.MAPPROJ, '*Postel*') then projection = 'po' else $
if strmatch(head_struct.MAPPROJ, '*LambertCylindrical*') then projection = 'lc' 

sz=size(data)

for i=0,sz(3)-1 do begin &$
frame = data(*,*,i) &$
new_filename = strcompress(outputdir + start_time + '-' + segment + '-' + projection +'-'+ string(i, FORMAT='(I6.5)'),/remove_all) +'.save'
save,file= new_filename, frame, header &$
endfor

end