function make_freq, n
;
; make_freq: 
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
; - - returns the frequencies that IDL expects corresponding to a dimension
; - - of length n returned from an FFT.
;+
; k=make_freq(n)
; make_freq returns a 1-d array of wavenumbers consistent with the wave
; numbers used by the fft function in IDL.
;-
;
nlong=long(n)
kfreq=fltarr(nlong)
n21=nlong/2L-1L
kfreq[0:n21]=findgen(n21+1L)
nnext=long(n21+1L)
if((n/2)*2 ne n) then begin
   kfreq[nnext]=float(n/2)
   kfreq[nnext+1]=-(float(n/2))
   nnext=nnext+2
endif else begin
   kfreq[nnext]=float(n/2)
   nnext=nnext+1
endelse
kfreq[nnext:n-1]=-(float(n21)-findgen(n21))
return, kfreq
end
