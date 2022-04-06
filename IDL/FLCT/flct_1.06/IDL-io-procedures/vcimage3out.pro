pro vcimage3out, a,b,c, filename
;
; vcimage3out: 
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
; - - takes 3 images a, b, and c, assumed floating point, and finds the
; - - 2D size, and writes out the dimensions and arrays in an
; - - unformatted C-style write.  
;
;+
; vcimage3out,a,b,c, filename
; vcimage3out.pro writes out the 3 2-d arrays a,b,c to the file filename.
; a,b,c are converted to single precision floating arrays before output.
;-
np=n_params()
if (np ne 4) then begin
   message, 'Usage:  vcimage3out, a,b,c, fname',/info
   return
endif
asize=size(a)
bsize=size(b)
csize=size(c)
size_cka=total(abs((asize-bsize)))
size_ckb=total(abs((bsize-csize)))
if((size_cka ne 0) or (size_ckb ne 0)) then begin
  print,'vcsizeimage3out: dimensions or ranks of a,b,c do not match'
  return
endif

asize=size(a)
if((asize[0] ne 2) and (n_elements(a) gt 1)) then begin
   message, 'in vcimage3out, input array a is not 2d',/info
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

writeu, unit, vcid, nx, ny
; now ensure that output arrays are converted to float for compat w/flct
aa=float(a)
bb=float(b)
cc=float(c)
writeu, unit, aa
writeu, unit, bb
writeu, unit, cc

close, unit
free_lun,unit
return
end
