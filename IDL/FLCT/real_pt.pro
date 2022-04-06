function real_pt,f
;
; real_pt: 
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

;+
; a=real_pt(f)
; real_pt returns the real part of f if f is single or double precision complex.
; For any other type of argument, real_pt just returns f without change.
;-


; The real_pt function returns real part of single or double precision
; complex numbers.  It otherwise just returns the argument without change.

; get variable type

fsize=size(f)
typecode=fsize[fsize[0]+1]

; for single precision complex, return float(f), for double precision complex
; return double(f), otherwise just return f

if (typecode eq 6) then begin
   return,float(f)
endif else if(typecode eq 9) then begin
   return,double(f)
endif else begin
   return,f
endelse

end
