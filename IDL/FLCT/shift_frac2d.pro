function shift_frac2d,f,deltax,deltay,progress=progress,err_stop=err_stop
;
; shift_frac2d: 
; http://solarmuri.ssl.berkeley.edu/overview/publicdownloads/software.html
; Copyright (C) 2007,2008 Regents of the University of California
; Authors G.H. Fisher and B.T. Welsch, Space Sciences Lab, UC Berkeley 
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
; f2=shift_frac2d(f,deltax,deltay,progress=progress,err_stop=err_stop)
;
; The purpose of shift_frac2d is to perform shifts on a 2-d array f,
; with the shifts as floating point numbers instead of integers.  The syntax
; is otherwise similar to the use of IDL's shift function.  The units of
; the applied shifts are in pixels.  Fractional pixels
; work fine.  deltax and deltay can also be 2-d arrays representing non-uniform
; shifts (warping) provided their dimensions are the same as those of f.
; The progress keyword toggles the output of progress information for
; shift_frac2d while it is running.  This is sometimes useful for non-uniform
; shifts (warping) when the array sizes are large.  The keyword err_stop
; toggles a stop within shift_frac2d on error, useful for debugging purposes.
;-

np=n_params()
if (np ne 3) then begin
   message,'Usage:  f2=shift_frac2d(f,deltax,deltay, /progress,/err_stop)',/info
   message,'Input:  f - a 2-d array (image) to be shifted',/info
   message,'Input:  deltax - scalar or 2-d array of desired shift(s) in x',/info
   message,'Input:  deltay - scalar or 2-d array of desired shift(s) in y',/info
   message,'Output:  f2 - resulting 2-d shifted array',/info
   if(keyword_set(err_stop)) then begin
      stop
   endif else begin
      return, -1
   endelse
endif

; get frequencies:

fsize=size(f)
if(fsize[0] ne 2) then begin
   print,'shift_frac2d: input array f is not 2D'
   if(keyword_set(err_stop)) then begin
      stop
   endif else begin
      return, -1
   endelse
endif
nx=fsize[1]
ny=fsize[2]

kx=make_freq(nx)
ky=make_freq(ny)

dsizex=size(deltax)
dsizey=size(deltay)
dsize_ck=total(abs((dsizex-dsizey)))
if(dsize_ck ne 0) then begin
  print,'shift_frac2d: type, dimensions, or rank of deltax,deltay do not match'
  if(keyword_set(err_stop)) then begin
     stop
  endif else begin
     return, -1
  endelse
endif else begin
  if(dsizex[0] eq 2) then begin
     dxnx=dsizex[1]
     dxny=dsizex[2]
     dynx=dsizey[1]
     dyny=dsizey[2]
     if((dxnx ne nx) or (dxny ne ny)) then begin
        print,'shift_frac2d: 2d dims of deltax,deltay do not match those of f'
        if(keyword_set(err_stop)) then begin
           stop
        endif else begin
           return, -1
        endelse
     endif
     scalar=0
     twodee=1
  endif else begin
     if(dsizex[0] eq 0) then begin
        scalar=1
        twodee=0
     endif else begin
        print,'shift_frac2d: rank or dims of deltax,deltay do not match f'
        if(keyword_set(err_stop)) then begin
           stop
        endif else begin
           return, -1
        endelse
     endelse
  endelse
endelse

; minus sign applied to make behavior consistent with IDL's shift function

if(scalar eq 1) then begin

  ; This branch when deltax, deltay are scalars, and a uniform shift is applied:

  ; Now, compute fft of original function:

  ff=fft(f,-1)
  fsize=size(ff)
  typecode=fsize[fsize[0]+1]

  ; shifts normalized to nx,ny
  if(typecode eq 6) then begin

     dxarg=-float(deltax)/float(nx)
     dyarg=-float(deltay)/float(ny)

     ; Compute fft of delta function at deltax,deltay:

     ffdeltx=complex(cos(!pi*2.*kx*dxarg),sin(!pi*2.*kx*dxarg))
     ffdelty=complex(cos(!pi*2.*ky*dyarg),sin(!pi*2.*ky*dyarg))

  endif else if(typecode eq 9) then begin

     dxarg=-double(deltax)/double(nx)
     dyarg=-double(deltay)/double(ny)

     ; compute double precision fft of delta function at deltax,deltay:

     ffdeltx=dcomplex(cos(!dpi*2.*kx*dxarg),sin(!dpi*2.*kx*dxarg))
     ffdelty=dcomplex(cos(!dpi*2.*ky*dyarg),sin(!dpi*2.*ky*dyarg))

  endif else begin

     print,'shift_frac2d: scalar branch - fouled up typecode = ',typecode
     if(keyword_set(err_stop)) then begin
        stop
     endif else begin
        return,-1
     endelse
  endelse

  ffdelt=ffdeltx#ffdelty


  ; compute fft of shifted fn, using convolution theorem:

  ffshift=ffdelt*ff

  ; get shifted fn with inverse transform:

  fshift=fft(ffshift,1) 

  ; Here, we use real_pt instead of real_part so this can be used in GDL too

  return,real_pt(fshift)

endif

if(twodee eq 1) then begin

  ; This branch when deltax, deltay are 2d arrays, non-uniform shifts applied

  ; Initialize f2 to have the same type, dimensions as f:

  f2=f
  ; Compute fft of original function f:

  ff=fft(f,-1)
  fsize=size(ff)
  typecode=fsize[fsize[0]+1]
  if((typecode ne 6) and (typecode ne 9)) then begin
    print,'shift_fracd: twodee branch - typecode = ',typecode,' is fouled up'
     if(keyword_set(err_stop)) then begin
         stop
     endif else begin
         return,-1
     endelse
  endif

  ; Now loop over all pixels, computing inverse FFT pixel-by-pixel:

  for ii=0L,nx*ny-1L do begin
    j=ii/nx
    i=ii mod nx
    if(keyword_set(progress)) then begin
       if(i eq 0) then begin
           print,'shift_frac2d: j = ',j,' out of ',ny-1
       endif
    endif

;   xarg, yarg are made up of two parts: The value of x,y, and the applied
;   shifts, deltax, deltay.  The - sign on the shifts is used to make the
;   output of shift_frac2d consistent with the IDL shift function.  The
;   deltax,deltay contributions come from the fft of the delta function,
;   and the i,j contributions come from expanding
;   the basis functions exp(ik dot r) evaluated at x_i,y_j.
    
    if(typecode eq 6) then begin

       xarg=!pi*2*(float(i) - float(deltax[ii]))/float(nx)
       yarg=!pi*2*(float(j) - float(deltay[ii]))/float(ny)

       ; 

       ; Compute exp[i(kx*(x-deltax)+ky*(y-deltay)] in single precision

       ffdeltx=complex(cos(kx*xarg), sin(kx*xarg))
       ffdelty=complex(cos(ky*yarg), sin(ky*yarg))

    endif 

    if(typecode eq 9) then begin

       xarg=!dpi*2*(double(i) - double(deltax[ii]))/double(nx)
       yarg=!dpi*2*(double(j) - double(deltay[ii]))/double(ny)

       ; Compute exp[i(kx*(x-deltax)+ky*(y-deltay)] in double precision

       ffdeltx=dcomplex(cos(kx*xarg),sin(kx*xarg))
       ffdelty=dcomplex(cos(ky*yarg),sin(ky*yarg))

       
    endif

    ffdelt=ffdeltx#ffdelty

    ;

    ffshift=ffdelt*ff

    ; get shifted fn with inverse transform at the single point x_i,y_j:
    ; Note use of real_pt instead of real_part so can be used with GDL too

    f2[ii]=real_pt(total(ffshift))

  endfor
  
  ; Loop over all points done, return the f2 array

  return,f2
endif
end
