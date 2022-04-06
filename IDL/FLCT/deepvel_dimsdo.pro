; ==============================================================================
; Date: 27-Jul-2017
; Auteur: Benoit Tremblay (Université de Montréal)
; Projet: DeepVel
; ------------------------------------------------------------------------------
; Description: Conversion des dimensions nx & ny aux valeurs correspondantes 
;              à la résolution de SDO/HMI.
;
; Fonctionnement: 
;
; PRO deepvel_dimsdo, nxi, nyi, dxi, dyi, nx=nx, ny=ny
; 
; Input: 
;        nxi: Integer. Dimensions originales.
;        nyi: Integer. Dimensions originales.
;        dxi: Float. Dimensions originales.
;        dyi: Float. Dimensions originales.
; Output: 
;        nx : Integer. Dimensions de l'image à la résolution de SDO.
;        ny : Integer. Dimensions de l'image à la résolution de SDO.
; ==============================================================================
PRO deepvel_dimsdo, nxi, nyi, dxi, dyi, nx=nx, ny=ny
; ==============================================================================

; ------------------------------------------------------------------------------
; (0) Paramètres SDO (à modifier !)
; ------------------------------------------------------------------------------

; Arcsec/pix
cdelt1=0.504365D0
cdelt2=0.504365D0
; R -> m
rsun_ref=6.96000D8
; R -> km
rkm=rsun_ref/1D3
; R -> arcsec
rsun_obs=953.288D0
; R -> pixel
rsun=round(rsun_obs/cdelt1)
; Nb. km / pix
pixel=rkm/rsun_obs*cdelt1

; ------------------------------------------------------------------------------
; (1) Conversion des dimensions
; ------------------------------------------------------------------------------

nx=ROUND(nxi*dxi/pixel)
ny=ROUND(nyi*dyi/pixel)

RETURN

END

; ==============================================================================
; END OF FILE
; ==============================================================================