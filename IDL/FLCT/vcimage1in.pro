pro vcimage1in, a, filename
;
; vcimage1in: 
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
; vcimage1in,a,filename
; vcimage1in reads in a single 2-d array a from file filename.
;-

np=n_params()
if (np ne 2) then begin
   message, 'Usage:  vcimage1in, a, filename',/info
   return
endif
get_lun, unit
openr, unit, filename,/swap_if_little_endian

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
readu,unit,a
if(n_elements(a) eq 1) then begin
  a=a[0]
endif
close,unit
free_lun,unit
return
end
