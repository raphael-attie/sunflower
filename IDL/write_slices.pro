

filename='/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/mtrack_20101126_220034_TAI_20101127_100034_TAI_Postel_continuum.fits'

data = fitsio_read_image(filename, header, so_path='/Users/rattie/ssw/vobs/ontology/binaries/darwin_x86_64/')

outputdir = '/Users/rattie/Data/SDO/HMI/Nov_27_2010_45s/mtracked/idl_save_files'

sz=size(data)

for i=0,sz(3)-1 do begin &$
;mlc=data(*,*,i) &$
;ilc=data(*,*,i) &$
mpo=data(*,*,i) &$
save,file= outputdir + '/Postel-m/20100901_120034-m-po-000'+ string(i, FORMAT='(I7.6)') +'.save',mpo,header &$
endfor

end


