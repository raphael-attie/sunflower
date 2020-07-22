pro vcimage2in, a,b, filename
;
; vcimage2in: 
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
; - - reads an input file "filename", finds the array dimensions, and returns
; - - the floating point array "a" to the calling program.
;
;
;+
; vcimage2in,a,b,filename
; vcimage2in.pro reads in the two 2-d arrays a,b from filename
;-
np=n_params()
if (np ne 3) then begin
   message, 'Usage:  vcimage2in, a, b, filename',/info
   return
endif
get_lun, unit
openr, unit, filename,/swap_if_little_endian
; - - data will be byteswapped if on small endian platform
nx=0L
ny=0L
vcid=0L
readu, unit, vcid, nx, ny

if(vcid ne 2136967593L) then begin
   message, 'Input file is not a vel_ccor i/o file',/info
   close,unit
   free_lun,unit
   return
endif

a=fltarr(nx,ny)
b=fltarr(nx,ny)
readu,unit,a
readu,unit,b
if(n_elements(a) eq 1) then begin
   a=a[0]
endif
if(n_elements(b) eq 1) then begin
   b=b[0]
endif
close,unit
free_lun,unit
return
end
