pro vcimage2out, a,b, filename
;
; vcimage2out: 
; http://solarmuri.ssl.berkeley.edu/overview/publicdownloads/software.html
; Copyright (C) 2007,2008 Regents of the University of California
;
; This program is free software; you can redistribute it and/or modify
; it under the terms of the GNU General Public License as published by
; the Free Software Foundation; either version 2 of the License, or
; (at your option) any later version.
;
; This program is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
; GNU General Public License for more details.
;
; You should have received a copy of the GNU General Public License
; along with this program; if not, write to the Free Software
; Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
;
;
; - - takes two images a and b, assumed floating point, and finds the
; - - 2D size, and writes out the dimensions and array in an
; - - unformatted C-style write.  
;
;+
; vcimage2out,a,b,filename
; vcimage2out.pro writes out the 2 2-d arrays a,b to the file filename.
; The arrays a,b are converted to single precision floats before output.
; This IDL procedure is used to create the input file consisting of
; two images for the flct local correlation tracking program.
;-
np=n_params()
if (np ne 3) then begin
   message, 'Usage:  vcimage2out, a,b, fname',/info
   return
endif
asize=size(a)
bsize=size(b)
size_ck=total(abs((asize-bsize)))
if(size_ck ne 0) then begin
  print,'vcsizeimage2out: dimensions or ranks of a,b do not match'
  return
endif
if((asize[0] ne 2) and (n_elements(a) gt 1)) then begin
   message, 'in vcimage2out, input array a is not 2d',/info
   return
endif

vcid=2136967593L ; initial integer ID for a "vel_ccor" i/o file
if(n_elements(a) gt 1) then begin
  nx=long(asize[1])
  ny=long(asize[2])
endif else begin
  nx=1L
  ny=1L
endelse

get_lun, unit
openw, unit, filename,/swap_if_little_endian
; - - note that data will be byte-swapped if this is a small endian machine:

writeu, unit, vcid, nx, ny
; - - flct expects 32-bit floats for the arrays, so convert them if necessary:
aa=float(a)
bb=float(b)
writeu, unit, aa
writeu, unit, bb

close, unit
free_lun,unit
return
end
