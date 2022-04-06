pro vcimage1out, a, filename
;
; vcimage1out: 
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
; - - takes an image a, and finds its
; - - 2D size, and writes out the dimensions and array in an
; - - unformatted C-style write.  a is converted to float before output.
;
;+
; vcimage1out,a,filename
; vcimage1out.pro writes out the dimensions and values of the 2-d array a
; to file filename.  It converts a to single precision float before writing.
;-
np=n_params()
if (np ne 2) then begin
   message, 'Usage:  vcimage1out, a, fname',/info
   return
endif

asize=size(a)
if((asize[0] ne 2) and (n_elements(a) gt 1)) then begin
   message, 'in vcimage1out, input array not 2d',/info
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
; Ensure a is converted to float
aa=float(a)
writeu, unit, aa

close, unit
free_lun,unit
return
end
