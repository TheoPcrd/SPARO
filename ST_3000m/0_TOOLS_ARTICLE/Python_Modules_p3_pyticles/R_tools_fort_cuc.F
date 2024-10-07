# 1 "R_tools_fort_cuc.F"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "R_tools_fort_cuc.F"

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! ROMS ROUTINES
!!
!! copied from actual ROMS scripts
!!
!! compile with:
!! "cpp R_tools_fort.F R_tools_fort.f"
!! "f2py -DF2PY_REPORT_ON_ARRAY_COPY=1 -c -m R_tools_fort R_tools_fort.f" for python use
!!
!! print R_tools_fort.rho_eos.__doc__
!!
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




# 1 "R_tools_fort_routines_gula/cppdefs_cuc.h" 1
# 51 "R_tools_fort_routines_gula/cppdefs_cuc.h"
# 1 "R_tools_fort_routines_gula/set_global_definitions.h" 1
# 16 "R_tools_fort_routines_gula/set_global_definitions.h"
c--#define ALLOW_SINGLE_BLOCK_MODE
# 260 "R_tools_fort_routines_gula/set_global_definitions.h"
c-#ifdef
c-# define float dfloat
c-# define FLoaT dfloat
c-# define FLOAT dfloat
c-# define sqrt dsqrt
c-# define SQRT dsqrt
c-# define exp dexp
c-# define EXP dexp
c-# define dtanh dtanh
c-# define TANH dtanh
c-#endif
# 52 "R_tools_fort_routines_gula/cppdefs_cuc.h" 2
# 20 "R_tools_fort_cuc.F" 2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# 1 "R_tools_fort_routines_gula/get_akv_cuc.F" 1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subpart of the lmd_kpp.F routine (myroms version)
! used to compute the Kv
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




c----#define WND_AT_RHO_POINTS



      subroutine get_akv_cuc (Lm,Mm,N,alpha,beta, z_r,z_w
     & , stflx, srflx, swr_frac, sustr, svstr ,Ricr, hbls, f
     & , u, v, bvf, rmask, r_D
     & , Kv)

      implicit none

      integer Lm,Mm,N,NT, i,j,k
     & ,itemp,isalt

      integer imin,imax,jmin,jmax

      real epsil

# 1 "R_tools_fort_routines_gula/scalars.h" 1
! This is include file "scalars.h"
!----- -- ------- ---- -----------
! The following common block contains time variables and indices
! for 2D (k-indices) and 3D (n-indices) computational engines. Since
! they are changed together, they are placed into the same cache line
! despite their mixed type, so that only one cachene is being
! invalidated and has to be propagated accross the cluster.
! Additionally, variables proc and CPU_time are to hold process ID
! numbers of individual threads and to measure CPU time consumed by
! each of them during the whole model run (these are for purely
! diagnostic/performance measurements and do not affect the model
! results.)
!
! Note that real values are placed first into the common block before
! integers. This is done to prevent misallignment of the 8-byte
! objects in the case when an uneven number of 4-byte integers is
! placed before a 8-byte real (in the case when default real size is
! set to 8 Bytes). Although misallignment is not formally a violation
! of fortran standard, it may cause performance degradation and/or
! make compiler issue a warning message (Sun, DEC Alpha) or even
! crash (Alpha).
!

!
! Physical constants: Earth radius [m]; Aceleration of gravity
!--------- ---------- duration of the day in seconds; Specific
! heat [Joules/kg/degC] for seawater (it is approximately 4000,
! and varies only slightly, see Gill, 1982, Appendix 3); von
! Karman constant.
!
      real pi, Eradius,g, Cp,vonKar, deg2rad,rad2deg,day2sec
      parameter (pi=3.14159265358979323, Eradius=6371315.,
     & deg2rad=pi/180., rad2deg=180./pi, day2sec=86400.,
     & Cp=3985., vonKar=0.41)
      parameter (g=9.81)

      real ,parameter :: sec2day=1./86400.
# 31 "R_tools_fort_routines_gula/get_akv_cuc.F" 2

      parameter (NT=2)
      parameter (itemp=1,isalt=2)
      parameter (epsil=1.E-16)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      real ustar3,
     & Bfsfc, zscale,
     & zetahat, ws,wm


      real*8 Bo(0:Lm+1,0:Mm+1), Bosol(0:Lm+1,0:Mm+1)
     & ,Bfsfc_bl(0:Lm+1)
     & ,z_bl
     & ,ustar(0:Lm+1,0:Mm+1)
     & ,Cr(0:Lm+1,0:N)
     & ,FC(0:Lm+1,0:N)
     & ,wrk1(0:Lm+1,0:N)
     & ,wrk2(0:Lm+1,0:N)
     & ,Hz(0:Lm+1,0:Mm+1,N)

     & ,FX(0:Lm+1,0:Mm+1)
     & ,FE(0:Lm+1,0:Mm+1)
     & ,FE1(0:Lm+1,0:Mm+1)



     & ,Gm1(0:Lm+1), Av_bl,
     & dGm1dS(0:Lm+1), dAv_bl, f1,
     & Gt1(0:Lm+1), At_bl, a1,
     & dGt1dS(0:Lm+1), dAt_bl, a2,
     & Gs1(0:Lm+1), As_bl, a3,
     & dGs1dS(0:Lm+1), dAs_bl

      integer kbl(0:Lm+1)

      real Kern, Vtc, Vtsq, sigma, cff,cff1, cff_up,cff_dn







      real nubl, nu0c, Cv, Ricr, Ri_inv, betaT, epssfc, C_Ek, C_MO,
     & Cstar, Cg, eps, zeta_m, a_m, c_m, zeta_s, a_s, c_s,
     & r2,r3,r4

      parameter (nubl=0.01,
     & nu0c=0.5,Cv=1.8,
     & betaT=-0.2,epssfc=0.1,C_MO=1.,C_Ek=258.,
     & Cstar=10.,eps=1.E-20,zeta_m=-0.2,a_m=1.257,
     & c_m=8.360,zeta_s=-1.0,a_s=-28.86,c_s=98.96,
     & r2=0.5, r3=1./3., r4=0.25)




      real*8 hbl(0:Lm+1,0:Mm+1)
     & ,umask(0:Lm+1,0:Mm+1),vmask(0:Lm+1,0:Mm+1)


      real*8 rdrg, Zob


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      real*8 ghat(0:Lm+1,0:Mm+1,N)
     & ,r_D(0:Lm+1,0:Mm+1)



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      ! Variables IN
      real*8 stflx(0:Lm+1,0:Mm+1,NT)
     & ,srflx(0:Lm+1,0:Mm+1)
     & ,swr_frac(0:Lm+1,0:Mm+1,0:N)
     & ,alpha(0:Lm+1,0:Mm+1), beta(0:Lm+1,0:Mm+1)
     & ,z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N)
     & ,sustr(1:Lm+1,0:Mm+1), svstr(0:Lm+1,1:Mm+1)
     & ,hbls(0:Lm+1,0:Mm+1), f(0:Lm+1,0:Mm+1)
     & ,u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N)
     & ,bvf(0:Lm+1,0:Mm+1,0:N),rmask(0:Lm+1,0:Mm+1)


      ! Variables OUT
      real*8 Kv(0:Lm+1,0:Mm+1,0:N)
     & ,Kt(0:Lm+1,0:Mm+1,0:N)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!# include "compute_tile_bounds.h"
# 1 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h" 1
! Auxiliary module "compute_auxiliary_bounds.h":
!---------- ------ -----------------------------
! Compute derived bounds for the loop indices over a subdomain
! "tile". The extended bounds [labelled by suffix R] are designed to
! cover also the outer ghost points, if the subdomain "tile" is
! adjacent to a PHYSICAL boundary. (NOTE: istrR,iendR,jstrR,jendR
! computed by this module DO NOT COVER ghost points associated with
! periodic boundaries (if any) or with 2-point computational marhins
! of MPI subdomains.
!
! This module also computes loop-bounds for U- and V-type variables
! which belong to the interior of the computational domain. These are
! labelled by suffixes U,V and they step one grid point inward from
! the side of the subdomain adjacent to the physical boundary.
! Conversely, for an internal subdomain [which does not have segments
! of physical boundary] all variables with suffixes R,U,V are set to
! the same values are the corresponding non-suffixed variables.
!
! Because this module also contains type declarations for these
! bounds, it must be included just after the last type declaration
! inside a subroutine, but before the first executable statement.
!
# 35 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h"
      integer istrU, istrR, iendR
# 46 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h"
      integer jstrV, jstrR, jendR


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      integer istr,iend,jstr,jend

        istr=1
        iend=Lm
        jstr=1
        jend=Mm

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




      if (istr.eq.1) then
        istrR=istr-1
        istrU=istr+1
      else
        istrR=istr
        istrU=istr
      endif
      if (iend.eq.Lm) then
        iendR=iend+1
      else
        iendR=iend
      endif



      if (jstr.eq.1) then
        jstrR=jstr-1
        jstrV=jstr+1
      else
        jstrR=jstr
        jstrV=jstr
      endif
      if (jend.eq.Mm) then
        jendR=jend+1
      else
        jendR=jend
      endif
# 130 "R_tools_fort_routines_gula/get_akv_cuc.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,alpha,beta ,z_r,z_w,stflx,srflx, swr_frac, sustr, svstr ,Ricr,hbls, f, u, v, bvf, rmask, r_D
Cf2py intent(out) Kv

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jstrR,jendR
        do i=istr,iendR
            umask(i,j) = rmask(i,j)*rmask(i-1,j)
          enddo
      enddo



      do j=jstr,jendR
        do i=istrR,iendR
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
          enddo
      enddo

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      Ri_inv=1./Ricr


      Cg=Cstar * vonKar * (c_s*vonKar*epssfc)**(1./3.)
      Vtc=Cv * sqrt(-betaT/(c_s*epssfc)) / (Ricr*vonKar**2)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1






      if (istr.eq.1) then
        imin=istr
      else
        imin=istr-1
      endif
      if (iend.eq.Lm) then
        imax=iend
      else
        imax=iend+1
      endif





      if (jstr.eq.1) then
        jmin=jstr
      else
        jmin=jstr-1
      endif
      if (jend.eq.Mm) then
        jmax=jend
      else
        jmax=jend+1
      endif
# 209 "R_tools_fort_routines_gula/get_akv_cuc.F"
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

       CALL lmd_vmix (Lm,Mm,N,u,v,z_r
     & ,bvf,Kv,Kt)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1


! Compute thermal expansion coefficient "alpha" [kg/m^3/decC] and
! saline contraction coefficient "beta" [kg/m^3/PSU] at surface, then
! compute surface turbulent buoyancy forcing "Bo" [m^2/s^3] (in doing
! so remove incoming solar shortwave radiation component and save it
! separately as "Bosol"). Also get an approximation for ssurface
! layer depth using "epssfc" and boundary layer depth from previous
! time step (this is needed to estimate turbulent velocity scale
! in computation of "Vterm" in "Cr", before new hbl is found). Also
! compute turbulent friction velocity "ustar" from wind stress at
! RHO-points. Finally, initialize boundary layer depth "hbl" and
! index "kbl" to the maximum (bottomed out) values.
!

! call alfabeta_tile (istr,iend,jstr,jend, imin,imax,
! & jmin,jmax, alpha,beta)
      do j=jmin,jmax
        do i=imin,imax

          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo


          Bo(i,j)=g*( alpha(i,j)*(stflx(i,j,itemp)-srflx(i,j))

     & -beta(i,j)*stflx(i,j,isalt)

     & )
          Bosol(i,j)=g*alpha(i,j)*srflx(i,j)


          ustar(i,j)=sqrt(0.5*sqrt( (sustr(i,j)+sustr(i+1,j))**2
     & +(svstr(i,j)+svstr(i,j+1))**2))
          hbl(i,j)=hbls(i,j) !<-- use old value as initial guess

          kbl(i)=0





          FC(i,N)=0.
          Cr(i,N)=0.
          Cr(i,0)=0.
        enddo !--> discard alpha,beta; keep Bo,Bosol to the very end.


!
!======================================
! Integral at W-points
!======================================
!
# 390 "R_tools_fort_routines_gula/get_akv_cuc.F"
!
!/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/!\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

!/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/!\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

!
        do k=N-1,1,-1
          do i=imin,imax
            zscale=z_w(i,j,N)-z_r(i,j,k)
            Kern=zscale/(zscale+epssfc*hbl(i,j))

            FC(i,k)=FC(i,k+1) + Kern*(
     & 0.5*( ( u(i ,j,k+1)-u(i ,j,k)
     & +u(i+1,j,k+1)-u(i+1,j,k) )**2
     & +( v(i,j ,k+1)-v(i,j ,k)
     & +v(i,j+1,k+1)-v(i,j+1,k) )**2
     & )/(Hz(i,j,k)+Hz(i,j,k+1))

     & -0.5*(Hz(i,j,k)+Hz(i,j,k+1))*( Ri_inv*bvf(i,j,k)

     & +C_Ek*f(i,j)*f(i,j)
     & ))
          enddo
        enddo

        do k=N,1,-1
          do i=imin,imax
            wrk1(i,k)=sqrt(swr_frac(i,j,k)*swr_frac(i,j,k-1))

            zscale=z_w(i,j,N)-z_r(i,j,k)
            Bfsfc=Bo(i,j)+Bosol(i,j)*(1.-wrk1(i,k))

# 1 "R_tools_fort_routines_gula/lmd_wscale_ws_only.h" 1
! Compute turbulent velocity scales for tracers, ws=ws(zscale,Bfsfc),
! where where zscale is distance from the surface and Bfsfc buoyancy
! forcing. The procedure of computation follows Eqs. (B1a)-(B1e) of
! LMD1994 paper with constants zeta_s, a_s, c_s specified in Eq.(B2).
! Mixing length scale "zscale" is limited by a specified fraction of
! boundary layer thickness in the case of unstable buoyancy forcing
! (as in the original 1994 code); or in both stable and unstable
! forcing (modification of Gokhan and Bill Large in 2003).
! Adapted from Bill Large 1995/2003 code.
!
! input variables: zscale, Bfsfc (both are volatile scalars),
! ustar, hbl (hbl is only for limiting of zscale)
! zeta_s, a_s, c_s are constants
!
! output: ws (volatile scalar)
!
!

          if (Bfsfc .lt. 0.) zscale=min(zscale, hbl(i,j)*epssfc)





          zetahat=vonKar*zscale*Bfsfc
          ustar3=ustar(i,j)**3

          if ((k.eq.N-1) .and. (j.eq.1) .and. (i.eq.181))
     & write(*,*)'zetahat',zetahat


!
! Stable regime.
!
          if (zetahat .ge. 0.) then
            ws=vonKar*ustar(i,j)*ustar3/max(ustar3+5.*zetahat,
     & 1.E-20)


!
! Unstable regime: note that zetahat is always negative here, also
! negative is the constant "zeta_s", hence "ustar" must be positive
! and bounded away from zero for this condition to be held.
!
          elseif (zetahat .gt. zeta_s*ustar3) then
            ws=vonKar*( (ustar3-16.*zetahat)/ustar(i,j) )**r2
!
! Convective regime: note that unlike the two cases above, this
! results in non-zero "ws" even in the case when ustar==0.
!
          else
            ws=vonKar*(a_s*ustar3-c_s*zetahat)**r3
          endif
                     !--> discard zetahat, ustar3
# 423 "R_tools_fort_routines_gula/get_akv_cuc.F" 2

c-- Vtsq=Vtc*ws*sqrt(max(0., 0.5*(bvf(i,j,k)+bvf(i,j,k-1)) ))

c fix by Gokhan

            Vtsq=Vtc*ws*sqrt(max(0., bvf(i,j,k-1) ))

            Cr(i,k)=FC(i,k)+Vtsq

            if (kbl(i).eq.0 .and. Cr(i,k).lt.0.) kbl(i)=k




          enddo
        enddo !<-- k, discard

        do i=imin,imax
          hbl(i,j)=z_w(i,j,N)-z_w(i,j,0) +eps
          if (kbl(i).gt.0) then
            k=kbl(i)
            if (k.eq.N) then
              hbl(i,j)=z_w(i,j,N)-z_r(i,j,N)
            else
              hbl(i,j)=z_w(i,j,N)-( z_r(i,j,k)*Cr(i,k+1)
     & -z_r(i,j,k+1)*Cr(i,k)
     & )/(Cr(i,k+1)-Cr(i,k))
            endif
          endif
# 472 "R_tools_fort_routines_gula/get_akv_cuc.F"
          hbl(i,j)=hbl(i,j)*rmask(i,j)

        enddo !--> discard kbl



!======================================



!================================
! Smoothing hbl/hbbl
!================================
!

      enddo !<-- j terminate j-loop only if smothing takes place.


      if (istr.eq.1) then
        do j=jmin,jmax
          hbl(istr-1,j)=hbl(istr,j)
        enddo
      endif
      if (iend.eq.Lm) then
        do j=jmin,jmax
          hbl(iend+1,j)=hbl(iend,j)
        enddo
      endif


      if (jstr.eq.1) then
        do i=imin,imax
          hbl(i,jstr-1)=hbl(i,jstr)
        enddo
      endif
      if (jend.eq.Mm) then
        do i=imin,imax
          hbl(i,jend+1)=hbl(i,jend)
        enddo
      endif

      if (istr.eq.1 .and. jstr.eq.1) then
        hbl(istr-1,jstr-1)=hbl(istr,jstr)
      endif
      if (istr.eq.1 .and. jend.eq.Mm) then
        hbl(istr-1,jend+1)=hbl(istr,jend)
      endif
      if (iend.eq.Lm .and. jstr.eq.1) then
        hbl(iend+1,jstr-1)=hbl(iend,jstr)
      endif
      if (iend.eq.Lm .and. jend.eq.Mm) then
        hbl(iend+1,jend+1)=hbl(iend,jend)
      endif


!
! Apply horizontal smoothing operator to hbl, while avoiding land-
! masked values which is accomplished by explessing everything in
! terms of elementary differences, subject to masking by U,V-rules.
! In the absense of masking the stencil of smoothing operator has
! the following weights, depending on coefficient settings in the
! code segment below:
!
! cff = 1/8, 1/4 cff = 1/12, 3/16 cff = 0, 1/8
!
! 1/16 1/8 1/16 1/32 1/8 1/32 1/8
!
! 1/8 1/4 1/8 1/8 3/8 1/8 1/8 1/4 1/8
!
! 1/16 1/8 1/16 1/32 1/8 1/32 1/8
!
! 2D 1-2-1-Hanning isotropic 5-point
! window smoother Laplacian Laplacian
!
! in all three cases the smoothing operator suppresses cheque-board
! mode in just one iteration; however, only the first one eliminates
! the 1D (flat-front) 2dx-modes in one iteration; the two others
! attenuate 1D 2dx-mode by factors of 1/4 and 1/2 per iteration
! respectively.
!
      do j=jstr-1,jend+1
        do i=istr,iend+1
          FX(i,j)=(hbl(i,j)-hbl(i-1,j))

     & *umask(i,j)

        enddo
      enddo
      do j=jstr,jend+1
        do i=istr-1,iend+1
          FE(i,j)=(hbl(i,j)-hbl(i,j-1))

     & *vmask(i,j)

        enddo
      enddo
      cff=1.D0/12.D0
      cff1=3.D0/16.D0
      do j=jstr,jend+1
        do i=istr,iend
          FE1(i,j)=FE(i,j)+cff*( FX(i+1,j)+FX(i,j-1)
     & -FX(i,j)-FX(i+1,j-1))
        enddo
      enddo
      do j=jstr,jend
        do i=istr,iend+1
          FX(i,j)=FX(i,j)+cff*( FE(i,j+1)+FE(i-1,j)
     & -FE(i,j)-FE(i-1,j+1))
        enddo
      enddo
      do j=jstr,jend
        do i=istr,iend
          hbl(i,j)=hbl(i,j)+cff1*( FX(i+1,j)-FX(i,j)
     & +FE1(i,j+1)-FE1(i,j))

          hbl(i,j)=hbl(i,j)*rmask(i,j)

        enddo !--> discard FX,FE,FE1
      enddo

      do j=jstr,jend !--> restart j-loop



!================================
! Surface KPP
!================================
!
        do i=istr,iend
          kbl(i)=N !<-- initialize search
        enddo
        do k=N-1,1,-1 ! find new boundary layer index "kbl".
          do i=istr,iend
            if (z_w(i,j,k) .gt. z_w(i,j,N)-hbl(i,j)) kbl(i)=k
          enddo
        enddo
!
! Find buoyancy forcing for final "hbl" values, and compute
! tubulent velocity scales (wm,ws) at "hbl".
! Then compute nondimensional shape function coefficients Gx( ) by
! matching values and vertical derivatives of interior mixing
! coefficients at hbl (sigma=1).
!
        do i=istr,iend
          k=kbl(i)
          z_bl=z_w(i,j,N)-hbl(i,j)
          zscale=hbl(i,j)

          if (swr_frac(i,j,k-1).gt. 0.) then
            Bfsfc=Bo(i,j) +Bosol(i,j)*( 1. -swr_frac(i,j,k-1)
     & *swr_frac(i,j,k)*(z_w(i,j,k)-z_w(i,j,k-1))
     & /( swr_frac(i,j,k )*(z_w(i,j,k) -z_bl)
     & +swr_frac(i,j,k-1)*(z_bl -z_w(i,j,k-1))
     & ))
          else
            Bfsfc=Bo(i,j)+Bosol(i,j)
          endif

# 1 "R_tools_fort_routines_gula/lmd_wscale_wm_and_ws.h" 1

            if (Bfsfc.lt.0.) zscale=min(zscale, hbl(i,j)*epssfc)




            zscale=zscale*rmask(i,j)

            zetahat=vonKar*zscale*Bfsfc
            ustar3=ustar(i,j)**3
!
! Stable regime.
!
            if (zetahat.ge.0.) then
              wm=vonKar*ustar(i,j)*ustar3/max( ustar3+5.*zetahat,
     & 1.E-20)
              ws=wm
!
! Unstable regime: note that zetahat is always negative here, also
! negative are constants "zeta_m" and "zeta_s".
!
            else
              if (zetahat .gt. zeta_m*ustar3) then
                wm=vonKar*( ustar(i,j)*(ustar3-16.*zetahat) )**r4
              else
                wm=vonKar*(a_m*ustar3-c_m*zetahat)**r3
              endif
              if (zetahat .gt. zeta_s*ustar3) then
                ws=vonKar*( (ustar3-16.*zetahat)/ustar(i,j) )**r2
              else
                ws=vonKar*(a_s*ustar3-c_s*zetahat)**r3
              endif
            endif
# 631 "R_tools_fort_routines_gula/get_akv_cuc.F" 2


          f1=5.0 * max(0., Bfsfc) * vonKar/(ustar(i,j)**4+eps)



          cff=1./(z_w(i,j,k)-z_w(i,j,k-1))
          cff_up=cff*(z_bl -z_w(i,j,k-1))
          cff_dn=cff*(z_w(i,j,k) -z_bl)

          Av_bl=cff_up*Kv(i,j,k)+cff_dn*Kv(i,j,k-1)
          dAv_bl=cff * (Kv(i,j,k) - Kv(i,j,k-1))
          Gm1(i)=Av_bl/(hbl(i,j)*wm+eps)
          dGm1dS(i)=min(0., Av_bl*f1-dAv_bl/(wm+eps))
!
! At_bl=cff_up*Kt(i,j,k)+cff_dn*Kt(i,j,k-1)
! dAt_bl=cff * (Kt(i,j,k) - Kt(i,j,k-1))
! Gt1(i)=At_bl/(hbl(i,j)*ws+eps)
! dGt1dS(i)=min(0., At_bl*f1-dAt_bl/(ws+eps))

! # ifdef
! As_bl=cff_up*Ks(i,j,k)+cff_dn*Ks(i,j,k-1)
! dAs_bl=cff * (Ks(i,j,k) - Ks(i,j,k-1))
! Gs1(i)=As_bl/(hbl(i,j)*ws+eps)
! dGs1dS(i)=min(0., As_bl*f1-dAs_bl/(ws+eps))
! # endif
          Bfsfc_bl(i)=Bfsfc
        enddo

!
! Compute boundary layer mixing coefficients.
!--------- -------- ----- ------ -------------
! Compute turbulent velocity scales at vertical W-points.
!
        do i=istr,iend
          do k=N-1,kbl(i),-1
            Bfsfc=Bfsfc_bl(i)
            zscale=z_w(i,j,N)-z_w(i,j,k)

# 1 "R_tools_fort_routines_gula/lmd_wscale_wm_and_ws.h" 1

            if (Bfsfc.lt.0.) zscale=min(zscale, hbl(i,j)*epssfc)




            zscale=zscale*rmask(i,j)

            zetahat=vonKar*zscale*Bfsfc
            ustar3=ustar(i,j)**3
!
! Stable regime.
!
            if (zetahat.ge.0.) then
              wm=vonKar*ustar(i,j)*ustar3/max( ustar3+5.*zetahat,
     & 1.E-20)
              ws=wm
!
! Unstable regime: note that zetahat is always negative here, also
! negative are constants "zeta_m" and "zeta_s".
!
            else
              if (zetahat .gt. zeta_m*ustar3) then
                wm=vonKar*( ustar(i,j)*(ustar3-16.*zetahat) )**r4
              else
                wm=vonKar*(a_m*ustar3-c_m*zetahat)**r3
              endif
              if (zetahat .gt. zeta_s*ustar3) then
                ws=vonKar*( (ustar3-16.*zetahat)/ustar(i,j) )**r2
              else
                ws=vonKar*(a_s*ustar3-c_s*zetahat)**r3
              endif
            endif
# 671 "R_tools_fort_routines_gula/get_akv_cuc.F" 2
!
! Compute vertical mixing coefficients
!
            sigma=(z_w(i,j,N)-z_w(i,j,k))/max(hbl(i,j),eps)

            a1=sigma-2.
            a2=3.-2.*sigma
            a3=sigma-1.

            if (sigma.lt.0.07D0) then
              cff=0.5*(sigma-0.07D0)**2/0.07D0
            else
              cff=0.D0
            endif

            Kv(i,j,k)=wm*hbl(i,j)*( cff + sigma*( 1.+sigma*(
     & a1+a2*Gm1(i)+a3*dGm1dS(i) )))

! Kt(i,j,k)=ws*hbl(i,j)*( cff + sigma*( 1.+sigma*(
! & a1+a2*Gt1(i)+a3*dGt1dS(i) )))
! # ifdef
! Ks(i,j,k)=ws*hbl(i,j)*( cff + sigma*( 1.+sigma*(
! & a1+a2*Gs1(i)+a3*dGs1dS(i) )))
! # endif

            if (Bfsfc .lt. 0.) then
              ghat(i,j,k)=Cg * sigma*(1.-sigma)**2
            else
              ghat(i,j,k)=0.
            endif

          enddo

! ccc CHANGE cccccccccccccc
! # ifdef
! do k=N-1,kbl(i),-1
! # ifdef
! if (bvf(i,j,k).lt.0.) then ! Add convective
! Kv(i,j,k)=Kv(i,j,k) + nu0c ! adjustment
! ! Kt(i,j,k)=Kt(i,j,k) + nu0c
! ! # ifdef
! ! Ks(i,j,k)=Ks(i,j,k) + nu0c
! ! # endif
! endif
! # endif
! enddo
! # endif
! ccc CHANGE cccccccccccccc


          do k=kbl(i)-1,1,-1

            ghat(i,j,k)=0.


            if (bvf(i,j,k).lt.0.) then ! Add convective
              Kv(i,j,k)=Kv(i,j,k) + nu0c ! adjustment
! Kt(i,j,k)=Kt(i,j,k) + nu0c ! if LMD_BKPP is defined, this will be done later !
! # ifdef
! Ks(i,j,k)=Ks(i,j,k) + nu0c
! # endif
            endif

          enddo
        enddo


      enddo !<-- j



!======================================


      return
      end
# 26 "R_tools_fort_cuc.F" 2

# 1 "R_tools_fort_routines_gula/get_akt_cuc.F" 1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subpart of the lmd_kpp.F routine (myroms version)
! used to compute the Kv
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




c----#define WND_AT_RHO_POINTS



      subroutine get_akt_cuc (Lm,Mm,N,alpha,beta, z_r,z_w
     & , stflx, srflx, swr_frac, sustr, svstr ,Ricr, hbls, f
     & , u, v, bvf, rmask, r_D
     & , Kv, Kt)

      implicit none

      integer Lm,Mm,N,NT, i,j,k
     & ,itemp,isalt

      integer imin,imax,jmin,jmax

      real epsil

# 1 "R_tools_fort_routines_gula/scalars.h" 1
! This is include file "scalars.h"
!----- -- ------- ---- -----------
! The following common block contains time variables and indices
! for 2D (k-indices) and 3D (n-indices) computational engines. Since
! they are changed together, they are placed into the same cache line
! despite their mixed type, so that only one cachene is being
! invalidated and has to be propagated accross the cluster.
! Additionally, variables proc and CPU_time are to hold process ID
! numbers of individual threads and to measure CPU time consumed by
! each of them during the whole model run (these are for purely
! diagnostic/performance measurements and do not affect the model
! results.)
!
! Note that real values are placed first into the common block before
! integers. This is done to prevent misallignment of the 8-byte
! objects in the case when an uneven number of 4-byte integers is
! placed before a 8-byte real (in the case when default real size is
! set to 8 Bytes). Although misallignment is not formally a violation
! of fortran standard, it may cause performance degradation and/or
! make compiler issue a warning message (Sun, DEC Alpha) or even
! crash (Alpha).
!

!
! Physical constants: Earth radius [m]; Aceleration of gravity
!--------- ---------- duration of the day in seconds; Specific
! heat [Joules/kg/degC] for seawater (it is approximately 4000,
! and varies only slightly, see Gill, 1982, Appendix 3); von
! Karman constant.
!
      real pi, Eradius,g, Cp,vonKar, deg2rad,rad2deg,day2sec
      parameter (pi=3.14159265358979323, Eradius=6371315.,
     & deg2rad=pi/180., rad2deg=180./pi, day2sec=86400.,
     & Cp=3985., vonKar=0.41)
      parameter (g=9.81)

      real ,parameter :: sec2day=1./86400.
# 31 "R_tools_fort_routines_gula/get_akt_cuc.F" 2

      parameter (NT=2)
      parameter (itemp=1,isalt=2)
      parameter (epsil=1.E-16)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      real ustar3,
     & Bfsfc, zscale,
     & zetahat, ws,wm


      real*8 Bo(0:Lm+1,0:Mm+1), Bosol(0:Lm+1,0:Mm+1)
     & ,Bfsfc_bl(0:Lm+1)
     & ,z_bl
     & ,ustar(0:Lm+1,0:Mm+1)
     & ,Cr(0:Lm+1,0:N)
     & ,FC(0:Lm+1,0:N)
     & ,wrk1(0:Lm+1,0:N)
     & ,wrk2(0:Lm+1,0:N)
     & ,Hz(0:Lm+1,0:Mm+1,N)

     & ,FX(0:Lm+1,0:Mm+1)
     & ,FE(0:Lm+1,0:Mm+1)
     & ,FE1(0:Lm+1,0:Mm+1)



     & ,Gm1(0:Lm+1), Av_bl,
     & dGm1dS(0:Lm+1), dAv_bl, f1,
     & Gt1(0:Lm+1), At_bl, a1,
     & dGt1dS(0:Lm+1), dAt_bl, a2,
     & Gs1(0:Lm+1), As_bl, a3,
     & dGs1dS(0:Lm+1), dAs_bl

      integer kbl(0:Lm+1)

      real Kern, Vtc, Vtsq, sigma, cff,cff1, cff_up,cff_dn







      real nubl, nu0c, Cv, Ricr, Ri_inv, betaT, epssfc, C_Ek, C_MO,
     & Cstar, Cg, eps, zeta_m, a_m, c_m, zeta_s, a_s, c_s,
     & r2,r3,r4

      parameter (nubl=0.01,
     & nu0c=0.5,Cv=1.8,
     & betaT=-0.2,epssfc=0.1,C_MO=1.,C_Ek=258.,
     & Cstar=10.,eps=1.E-20,zeta_m=-0.2,a_m=1.257,
     & c_m=8.360,zeta_s=-1.0,a_s=-28.86,c_s=98.96,
     & r2=0.5, r3=1./3., r4=0.25)




      real*8 hbl(0:Lm+1,0:Mm+1)
     & ,umask(0:Lm+1,0:Mm+1),vmask(0:Lm+1,0:Mm+1)


      real*8 rdrg, Zob


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      real*8 ghat(0:Lm+1,0:Mm+1,N)
     & ,r_D(0:Lm+1,0:Mm+1)



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      ! Variables IN
      real*8 stflx(0:Lm+1,0:Mm+1,NT)
     & ,srflx(0:Lm+1,0:Mm+1)
     & ,swr_frac(0:Lm+1,0:Mm+1,0:N)
     & ,alpha(0:Lm+1,0:Mm+1), beta(0:Lm+1,0:Mm+1)
     & ,z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N)
     & ,sustr(1:Lm+1,0:Mm+1), svstr(0:Lm+1,1:Mm+1)
     & ,hbls(0:Lm+1,0:Mm+1), f(0:Lm+1,0:Mm+1)
     & ,u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N)
     & ,bvf(0:Lm+1,0:Mm+1,0:N),rmask(0:Lm+1,0:Mm+1)


      ! Variables OUT
      real*8 Kv(0:Lm+1,0:Mm+1,0:N)
     & ,Kt(0:Lm+1,0:Mm+1,0:N)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!# include "compute_tile_bounds.h"
# 1 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h" 1
! Auxiliary module "compute_auxiliary_bounds.h":
!---------- ------ -----------------------------
! Compute derived bounds for the loop indices over a subdomain
! "tile". The extended bounds [labelled by suffix R] are designed to
! cover also the outer ghost points, if the subdomain "tile" is
! adjacent to a PHYSICAL boundary. (NOTE: istrR,iendR,jstrR,jendR
! computed by this module DO NOT COVER ghost points associated with
! periodic boundaries (if any) or with 2-point computational marhins
! of MPI subdomains.
!
! This module also computes loop-bounds for U- and V-type variables
! which belong to the interior of the computational domain. These are
! labelled by suffixes U,V and they step one grid point inward from
! the side of the subdomain adjacent to the physical boundary.
! Conversely, for an internal subdomain [which does not have segments
! of physical boundary] all variables with suffixes R,U,V are set to
! the same values are the corresponding non-suffixed variables.
!
! Because this module also contains type declarations for these
! bounds, it must be included just after the last type declaration
! inside a subroutine, but before the first executable statement.
!
# 35 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h"
      integer istrU, istrR, iendR
# 46 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h"
      integer jstrV, jstrR, jendR


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      integer istr,iend,jstr,jend

        istr=1
        iend=Lm
        jstr=1
        jend=Mm

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




      if (istr.eq.1) then
        istrR=istr-1
        istrU=istr+1
      else
        istrR=istr
        istrU=istr
      endif
      if (iend.eq.Lm) then
        iendR=iend+1
      else
        iendR=iend
      endif



      if (jstr.eq.1) then
        jstrR=jstr-1
        jstrV=jstr+1
      else
        jstrR=jstr
        jstrV=jstr
      endif
      if (jend.eq.Mm) then
        jendR=jend+1
      else
        jendR=jend
      endif
# 130 "R_tools_fort_routines_gula/get_akt_cuc.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,alpha,beta ,z_r,z_w,stflx,srflx, swr_frac, sustr, svstr ,Ricr,hbls, f, u, v, bvf, rmask, r_D
Cf2py intent(out) Kv, Kt

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jstrR,jendR
        do i=istr,iendR
            umask(i,j) = rmask(i,j)*rmask(i-1,j)
          enddo
      enddo



      do j=jstr,jendR
        do i=istrR,iendR
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
          enddo
      enddo

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      Ri_inv=1./Ricr


      Cg=Cstar * vonKar * (c_s*vonKar*epssfc)**(1./3.)
      Vtc=Cv * sqrt(-betaT/(c_s*epssfc)) / (Ricr*vonKar**2)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1






      if (istr.eq.1) then
        imin=istr
      else
        imin=istr-1
      endif
      if (iend.eq.Lm) then
        imax=iend
      else
        imax=iend+1
      endif





      if (jstr.eq.1) then
        jmin=jstr
      else
        jmin=jstr-1
      endif
      if (jend.eq.Mm) then
        jmax=jend
      else
        jmax=jend+1
      endif
# 207 "R_tools_fort_routines_gula/get_akt_cuc.F"
        write(*,*) 'befKv(1,1,1),Kt(1,1,1)',Kv(1,1,1),Kt(1,1,1)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

       CALL lmd_vmix (Lm,Mm,N,u,v,z_r
     & ,bvf,Kv,Kt)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

        write(*,*) 'aftKv(1,1,1),Kt(1,1,1)',Kv(1,1,1),Kt(1,1,1)


! Compute thermal expansion coefficient "alpha" [kg/m^3/decC] and
! saline contraction coefficient "beta" [kg/m^3/PSU] at surface, then
! compute surface turbulent buoyancy forcing "Bo" [m^2/s^3] (in doing
! so remove incoming solar shortwave radiation component and save it
! separately as "Bosol"). Also get an approximation for ssurface
! layer depth using "epssfc" and boundary layer depth from previous
! time step (this is needed to estimate turbulent velocity scale
! in computation of "Vterm" in "Cr", before new hbl is found). Also
! compute turbulent friction velocity "ustar" from wind stress at
! RHO-points. Finally, initialize boundary layer depth "hbl" and
! index "kbl" to the maximum (bottomed out) values.
!

! call alfabeta_tile (istr,iend,jstr,jend, imin,imax,
! & jmin,jmax, alpha,beta)
      do j=jmin,jmax
        do i=imin,imax

          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo


          Bo(i,j)=g*( alpha(i,j)*(stflx(i,j,itemp)-srflx(i,j))

     & -beta(i,j)*stflx(i,j,isalt)

     & )
          Bosol(i,j)=g*alpha(i,j)*srflx(i,j)


          ustar(i,j)=sqrt(0.5*sqrt( (sustr(i,j)+sustr(i+1,j))**2
     & +(svstr(i,j)+svstr(i,j+1))**2))
          hbl(i,j)=hbls(i,j) !<-- use old value as initial guess

          kbl(i)=0





          FC(i,N)=0.
          Cr(i,N)=0.
          Cr(i,0)=0.
        enddo !--> discard alpha,beta; keep Bo,Bosol to the very end.


!
!======================================
! Integral at W-points
!======================================
!
# 485 "R_tools_fort_routines_gula/get_akt_cuc.F"
!================================
! Smoothing hbl/hbbl
!================================
!

      enddo !<-- j terminate j-loop only if smothing takes place.


      if (istr.eq.1) then
        do j=jmin,jmax
          hbl(istr-1,j)=hbl(istr,j)
        enddo
      endif
      if (iend.eq.Lm) then
        do j=jmin,jmax
          hbl(iend+1,j)=hbl(iend,j)
        enddo
      endif


      if (jstr.eq.1) then
        do i=imin,imax
          hbl(i,jstr-1)=hbl(i,jstr)
        enddo
      endif
      if (jend.eq.Mm) then
        do i=imin,imax
          hbl(i,jend+1)=hbl(i,jend)
        enddo
      endif

      if (istr.eq.1 .and. jstr.eq.1) then
        hbl(istr-1,jstr-1)=hbl(istr,jstr)
      endif
      if (istr.eq.1 .and. jend.eq.Mm) then
        hbl(istr-1,jend+1)=hbl(istr,jend)
      endif
      if (iend.eq.Lm .and. jstr.eq.1) then
        hbl(iend+1,jstr-1)=hbl(iend,jstr)
      endif
      if (iend.eq.Lm .and. jend.eq.Mm) then
        hbl(iend+1,jend+1)=hbl(iend,jend)
      endif


!
! Apply horizontal smoothing operator to hbl, while avoiding land-
! masked values which is accomplished by explessing everything in
! terms of elementary differences, subject to masking by U,V-rules.
! In the absense of masking the stencil of smoothing operator has
! the following weights, depending on coefficient settings in the
! code segment below:
!
! cff = 1/8, 1/4 cff = 1/12, 3/16 cff = 0, 1/8
!
! 1/16 1/8 1/16 1/32 1/8 1/32 1/8
!
! 1/8 1/4 1/8 1/8 3/8 1/8 1/8 1/4 1/8
!
! 1/16 1/8 1/16 1/32 1/8 1/32 1/8
!
! 2D 1-2-1-Hanning isotropic 5-point
! window smoother Laplacian Laplacian
!
! in all three cases the smoothing operator suppresses cheque-board
! mode in just one iteration; however, only the first one eliminates
! the 1D (flat-front) 2dx-modes in one iteration; the two others
! attenuate 1D 2dx-mode by factors of 1/4 and 1/2 per iteration
! respectively.
!
      do j=jstr-1,jend+1
        do i=istr,iend+1
          FX(i,j)=(hbl(i,j)-hbl(i-1,j))

     & *umask(i,j)

        enddo
      enddo
      do j=jstr,jend+1
        do i=istr-1,iend+1
          FE(i,j)=(hbl(i,j)-hbl(i,j-1))

     & *vmask(i,j)

        enddo
      enddo
      cff=1.D0/12.D0
      cff1=3.D0/16.D0
      do j=jstr,jend+1
        do i=istr,iend
          FE1(i,j)=FE(i,j)+cff*( FX(i+1,j)+FX(i,j-1)
     & -FX(i,j)-FX(i+1,j-1))
        enddo
      enddo
      do j=jstr,jend
        do i=istr,iend+1
          FX(i,j)=FX(i,j)+cff*( FE(i,j+1)+FE(i-1,j)
     & -FE(i,j)-FE(i-1,j+1))
        enddo
      enddo
      do j=jstr,jend
        do i=istr,iend
          hbl(i,j)=hbl(i,j)+cff1*( FX(i+1,j)-FX(i,j)
     & +FE1(i,j+1)-FE1(i,j))

          hbl(i,j)=hbl(i,j)*rmask(i,j)

        enddo !--> discard FX,FE,FE1
      enddo

      do j=jstr,jend !--> restart j-loop



!================================
! Surface KPP
!================================
!
        do i=istr,iend
          kbl(i)=N !<-- initialize search
        enddo
        do k=N-1,1,-1 ! find new boundary layer index "kbl".
          do i=istr,iend
            if (z_w(i,j,k) .gt. z_w(i,j,N)-hbl(i,j)) kbl(i)=k
          enddo
        enddo
!
! Find buoyancy forcing for final "hbl" values, and compute
! tubulent velocity scales (wm,ws) at "hbl".
! Then compute nondimensional shape function coefficients Gx( ) by
! matching values and vertical derivatives of interior mixing
! coefficients at hbl (sigma=1).
!
        do i=istr,iend
          k=kbl(i)
          z_bl=z_w(i,j,N)-hbl(i,j)
          zscale=hbl(i,j)

          if (swr_frac(i,j,k-1).gt. 0.) then
            Bfsfc=Bo(i,j) +Bosol(i,j)*( 1. -swr_frac(i,j,k-1)
     & *swr_frac(i,j,k)*(z_w(i,j,k)-z_w(i,j,k-1))
     & /( swr_frac(i,j,k )*(z_w(i,j,k) -z_bl)
     & +swr_frac(i,j,k-1)*(z_bl -z_w(i,j,k-1))
     & ))
          else
            Bfsfc=Bo(i,j)+Bosol(i,j)
          endif

# 1 "R_tools_fort_routines_gula/lmd_wscale_wm_and_ws.h" 1

            if (Bfsfc.lt.0.) zscale=min(zscale, hbl(i,j)*epssfc)




            zscale=zscale*rmask(i,j)

            zetahat=vonKar*zscale*Bfsfc
            ustar3=ustar(i,j)**3
!
! Stable regime.
!
            if (zetahat.ge.0.) then
              wm=vonKar*ustar(i,j)*ustar3/max( ustar3+5.*zetahat,
     & 1.E-20)
              ws=wm
!
! Unstable regime: note that zetahat is always negative here, also
! negative are constants "zeta_m" and "zeta_s".
!
            else
              if (zetahat .gt. zeta_m*ustar3) then
                wm=vonKar*( ustar(i,j)*(ustar3-16.*zetahat) )**r4
              else
                wm=vonKar*(a_m*ustar3-c_m*zetahat)**r3
              endif
              if (zetahat .gt. zeta_s*ustar3) then
                ws=vonKar*( (ustar3-16.*zetahat)/ustar(i,j) )**r2
              else
                ws=vonKar*(a_s*ustar3-c_s*zetahat)**r3
              endif
            endif
# 634 "R_tools_fort_routines_gula/get_akt_cuc.F" 2


          f1=5.0 * max(0., Bfsfc) * vonKar/(ustar(i,j)**4+eps)



          cff=1./(z_w(i,j,k)-z_w(i,j,k-1))
          cff_up=cff*(z_bl -z_w(i,j,k-1))
          cff_dn=cff*(z_w(i,j,k) -z_bl)

          Av_bl=cff_up*Kv(i,j,k)+cff_dn*Kv(i,j,k-1)
          dAv_bl=cff * (Kv(i,j,k) - Kv(i,j,k-1))
          Gm1(i)=Av_bl/(hbl(i,j)*wm+eps)
          dGm1dS(i)=min(0., Av_bl*f1-dAv_bl/(wm+eps))

          At_bl=cff_up*Kt(i,j,k)+cff_dn*Kt(i,j,k-1)
          dAt_bl=cff * (Kt(i,j,k) - Kt(i,j,k-1))
          Gt1(i)=At_bl/(hbl(i,j)*ws+eps)
          dGt1dS(i)=min(0., At_bl*f1-dAt_bl/(ws+eps))

! # ifdef
! As_bl=cff_up*Ks(i,j,k)+cff_dn*Ks(i,j,k-1)
! dAs_bl=cff * (Ks(i,j,k) - Ks(i,j,k-1))
! Gs1(i)=As_bl/(hbl(i,j)*ws+eps)
! dGs1dS(i)=min(0., As_bl*f1-dAs_bl/(ws+eps))
! # endif
          Bfsfc_bl(i)=Bfsfc
        enddo

!
! Compute boundary layer mixing coefficients.
!--------- -------- ----- ------ -------------
! Compute turbulent velocity scales at vertical W-points.
!
        do i=istr,iend
          do k=N-1,kbl(i),-1
            Bfsfc=Bfsfc_bl(i)
            zscale=z_w(i,j,N)-z_w(i,j,k)

# 1 "R_tools_fort_routines_gula/lmd_wscale_wm_and_ws.h" 1

            if (Bfsfc.lt.0.) zscale=min(zscale, hbl(i,j)*epssfc)




            zscale=zscale*rmask(i,j)

            zetahat=vonKar*zscale*Bfsfc
            ustar3=ustar(i,j)**3
!
! Stable regime.
!
            if (zetahat.ge.0.) then
              wm=vonKar*ustar(i,j)*ustar3/max( ustar3+5.*zetahat,
     & 1.E-20)
              ws=wm
!
! Unstable regime: note that zetahat is always negative here, also
! negative are constants "zeta_m" and "zeta_s".
!
            else
              if (zetahat .gt. zeta_m*ustar3) then
                wm=vonKar*( ustar(i,j)*(ustar3-16.*zetahat) )**r4
              else
                wm=vonKar*(a_m*ustar3-c_m*zetahat)**r3
              endif
              if (zetahat .gt. zeta_s*ustar3) then
                ws=vonKar*( (ustar3-16.*zetahat)/ustar(i,j) )**r2
              else
                ws=vonKar*(a_s*ustar3-c_s*zetahat)**r3
              endif
            endif
# 674 "R_tools_fort_routines_gula/get_akt_cuc.F" 2
!
! Compute vertical mixing coefficients
!
            sigma=(z_w(i,j,N)-z_w(i,j,k))/max(hbl(i,j),eps)

            a1=sigma-2.
            a2=3.-2.*sigma
            a3=sigma-1.

            if (sigma.lt.0.07D0) then
              cff=0.5*(sigma-0.07D0)**2/0.07D0
            else
              cff=0.D0
            endif

            Kv(i,j,k)=wm*hbl(i,j)*( cff + sigma*( 1.+sigma*(
     & a1+a2*Gm1(i)+a3*dGm1dS(i) )))

            Kt(i,j,k)=ws*hbl(i,j)*( cff + sigma*( 1.+sigma*(
     & a1+a2*Gt1(i)+a3*dGt1dS(i) )))
! # ifdef
! Ks(i,j,k)=ws*hbl(i,j)*( cff + sigma*( 1.+sigma*(
! & a1+a2*Gs1(i)+a3*dGs1dS(i) )))
! # endif

            if (Bfsfc .lt. 0.) then
              ghat(i,j,k)=Cg * sigma*(1.-sigma)**2
            else
              ghat(i,j,k)=0.
            endif

          enddo

ccc CHANGE cccccccccccccc

          do k=N-1,kbl(i),-1

            if (bvf(i,j,k).lt.0.) then ! Add convective
              Kv(i,j,k)=Kv(i,j,k) + nu0c ! adjustment
              Kt(i,j,k)=Kt(i,j,k) + nu0c
! # ifdef
! Ks(i,j,k)=Ks(i,j,k) + nu0c
! # endif
            endif

          enddo

ccc CHANGE cccccccccccccc

          do k=kbl(i)-1,1,-1

            ghat(i,j,k)=0.


            if (bvf(i,j,k).lt.0.) then ! Add convective
              Kv(i,j,k)=Kv(i,j,k) + nu0c ! adjustment
              Kt(i,j,k)=Kt(i,j,k) + nu0c ! if LMD_BKPP is defined, this will be done later !
! # ifdef
! Ks(i,j,k)=Ks(i,j,k) + nu0c
! # endif
            endif

          enddo
        enddo


      enddo !<-- j



!======================================


      return
      end
# 28 "R_tools_fort_cuc.F" 2

# 1 "R_tools_fort_routines_gula/get_hbl_cuc.F" 1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subpart of the lmd_kpp.F routine (myroms version)
! used to compute the Kv
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




c----#define WND_AT_RHO_POINTS



      subroutine get_hbl_cuc (Lm,Mm,N,alpha,beta, z_r,z_w
     & , stflx, srflx, swr_frac, sustr, svstr ,Ricr, hbls, f
     & , u, v, bvf, rmask
     & , hbl)

      implicit none

      integer Lm,Mm,N,NT, i,j,k
     & ,itemp,isalt

      integer imin,imax,jmin,jmax

      real epsil

# 1 "R_tools_fort_routines_gula/scalars.h" 1
! This is include file "scalars.h"
!----- -- ------- ---- -----------
! The following common block contains time variables and indices
! for 2D (k-indices) and 3D (n-indices) computational engines. Since
! they are changed together, they are placed into the same cache line
! despite their mixed type, so that only one cachene is being
! invalidated and has to be propagated accross the cluster.
! Additionally, variables proc and CPU_time are to hold process ID
! numbers of individual threads and to measure CPU time consumed by
! each of them during the whole model run (these are for purely
! diagnostic/performance measurements and do not affect the model
! results.)
!
! Note that real values are placed first into the common block before
! integers. This is done to prevent misallignment of the 8-byte
! objects in the case when an uneven number of 4-byte integers is
! placed before a 8-byte real (in the case when default real size is
! set to 8 Bytes). Although misallignment is not formally a violation
! of fortran standard, it may cause performance degradation and/or
! make compiler issue a warning message (Sun, DEC Alpha) or even
! crash (Alpha).
!

!
! Physical constants: Earth radius [m]; Aceleration of gravity
!--------- ---------- duration of the day in seconds; Specific
! heat [Joules/kg/degC] for seawater (it is approximately 4000,
! and varies only slightly, see Gill, 1982, Appendix 3); von
! Karman constant.
!
      real pi, Eradius,g, Cp,vonKar, deg2rad,rad2deg,day2sec
      parameter (pi=3.14159265358979323, Eradius=6371315.,
     & deg2rad=pi/180., rad2deg=180./pi, day2sec=86400.,
     & Cp=3985., vonKar=0.41)
      parameter (g=9.81)

      real ,parameter :: sec2day=1./86400.
# 31 "R_tools_fort_routines_gula/get_hbl_cuc.F" 2

      parameter (NT=2)
      parameter (itemp=1,isalt=2)
      parameter (epsil=1.E-16)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      real ustar3,
     & Bfsfc, zscale,
     & zetahat, ws,wm


      real*8 Bo(0:Lm+1,0:Mm+1), Bosol(0:Lm+1,0:Mm+1)
     & ,Bfsfc_bl(0:Lm+1)
     & ,z_bl
     & ,ustar(0:Lm+1,0:Mm+1)
     & ,Cr(0:Lm+1,0:N)
     & ,FC(0:Lm+1,0:N)
     & ,wrk1(0:Lm+1,0:N)
     & ,wrk2(0:Lm+1,0:N)
     & ,Hz(0:Lm+1,0:Mm+1,N)

     & ,FX(0:Lm+1,0:Mm+1)
     & ,FE(0:Lm+1,0:Mm+1)
     & ,FE1(0:Lm+1,0:Mm+1)



     & ,Gm1(0:Lm+1), Av_bl,
     & dGm1dS(0:Lm+1), dAv_bl, f1,
     & Gt1(0:Lm+1), At_bl, a1,
     & dGt1dS(0:Lm+1), dAt_bl, a2,
     & Gs1(0:Lm+1), As_bl, a3,
     & dGs1dS(0:Lm+1), dAs_bl

      integer kbl(0:Lm+1)

      real Kern, Vtc, Vtsq, sigma, cff,cff1, cff_up,cff_dn







      real nubl, nu0c, Cv, Ricr, Ri_inv, betaT, epssfc, C_Ek, C_MO,
     & Cstar, Cg, eps, zeta_m, a_m, c_m, zeta_s, a_s, c_s,
     & r2,r3,r4

      parameter (nubl=0.01,
     & nu0c=0.5,Cv=1.8,
     & betaT=-0.2,epssfc=0.1,C_MO=1.,C_Ek=258.,
     & Cstar=10.,eps=1.E-20,zeta_m=-0.2,a_m=1.257,
     & c_m=8.360,zeta_s=-1.0,a_s=-28.86,c_s=98.96,
     & r2=0.5, r3=1./3., r4=0.25)




      real*8 hbl(0:Lm+1,0:Mm+1)
     & ,umask(0:Lm+1,0:Mm+1),vmask(0:Lm+1,0:Mm+1)


      real*8 rdrg, Zob


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      real*8 ghat(0:Lm+1,0:Mm+1,N)



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      ! Variables IN
      real*8 stflx(0:Lm+1,0:Mm+1,NT)
     & ,srflx(0:Lm+1,0:Mm+1)
     & ,swr_frac(0:Lm+1,0:Mm+1,0:N)
     & ,alpha(0:Lm+1,0:Mm+1), beta(0:Lm+1,0:Mm+1)
     & ,z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N)
     & ,sustr(1:Lm+1,0:Mm+1), svstr(0:Lm+1,1:Mm+1)
     & ,hbls(0:Lm+1,0:Mm+1), f(0:Lm+1,0:Mm+1)
     & ,u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N)
     & ,bvf(0:Lm+1,0:Mm+1,0:N),rmask(0:Lm+1,0:Mm+1)


      ! Variables OUT
      real*8 Kv(0:Lm+1,0:Mm+1,0:N)
     & ,Kt(0:Lm+1,0:Mm+1,0:N)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!# include "compute_tile_bounds.h"
# 1 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h" 1
! Auxiliary module "compute_auxiliary_bounds.h":
!---------- ------ -----------------------------
! Compute derived bounds for the loop indices over a subdomain
! "tile". The extended bounds [labelled by suffix R] are designed to
! cover also the outer ghost points, if the subdomain "tile" is
! adjacent to a PHYSICAL boundary. (NOTE: istrR,iendR,jstrR,jendR
! computed by this module DO NOT COVER ghost points associated with
! periodic boundaries (if any) or with 2-point computational marhins
! of MPI subdomains.
!
! This module also computes loop-bounds for U- and V-type variables
! which belong to the interior of the computational domain. These are
! labelled by suffixes U,V and they step one grid point inward from
! the side of the subdomain adjacent to the physical boundary.
! Conversely, for an internal subdomain [which does not have segments
! of physical boundary] all variables with suffixes R,U,V are set to
! the same values are the corresponding non-suffixed variables.
!
! Because this module also contains type declarations for these
! bounds, it must be included just after the last type declaration
! inside a subroutine, but before the first executable statement.
!
# 35 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h"
      integer istrU, istrR, iendR
# 46 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h"
      integer jstrV, jstrR, jendR


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      integer istr,iend,jstr,jend

        istr=1
        iend=Lm
        jstr=1
        jend=Mm

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




      if (istr.eq.1) then
        istrR=istr-1
        istrU=istr+1
      else
        istrR=istr
        istrU=istr
      endif
      if (iend.eq.Lm) then
        iendR=iend+1
      else
        iendR=iend
      endif



      if (jstr.eq.1) then
        jstrR=jstr-1
        jstrV=jstr+1
      else
        jstrR=jstr
        jstrV=jstr
      endif
      if (jend.eq.Mm) then
        jendR=jend+1
      else
        jendR=jend
      endif
# 129 "R_tools_fort_routines_gula/get_hbl_cuc.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,alpha,beta ,z_r,z_w,stflx,srflx, swr_frac, sustr, svstr ,Ricr,hbls, f, u, v, bvf, rmask
Cf2py intent(out) hbl

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jstrR,jendR
        do i=istr,iendR
            umask(i,j) = rmask(i,j)*rmask(i-1,j)
          enddo
      enddo



      do j=jstr,jendR
        do i=istrR,iendR
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
          enddo
      enddo

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      Ri_inv=1./Ricr


      Cg=Cstar * vonKar * (c_s*vonKar*epssfc)**(1./3.)
      Vtc=Cv * sqrt(-betaT/(c_s*epssfc)) / (Ricr*vonKar**2)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1






      if (istr.eq.1) then
        imin=istr
      else
        imin=istr-1
      endif
      if (iend.eq.Lm) then
        imax=iend
      else
        imax=iend+1
      endif





      if (jstr.eq.1) then
        jmin=jstr
      else
        jmin=jstr-1
      endif
      if (jend.eq.Mm) then
        jmax=jend
      else
        jmax=jend+1
      endif
# 208 "R_tools_fort_routines_gula/get_hbl_cuc.F"
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

       CALL lmd_vmix (Lm,Mm,N,u,v,z_r
     & ,bvf,Kv,Kt)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1




! Compute thermal expansion coefficient "alpha" [kg/m^3/decC] and
! saline contraction coefficient "beta" [kg/m^3/PSU] at surface, then
! compute surface turbulent buoyancy forcing "Bo" [m^2/s^3] (in doing
! so remove incoming solar shortwave radiation component and save it
! separately as "Bosol"). Also get an approximation for ssurface
! layer depth using "epssfc" and boundary layer depth from previous
! time step (this is needed to estimate turbulent velocity scale
! in computation of "Vterm" in "Cr", before new hbl is found). Also
! compute turbulent friction velocity "ustar" from wind stress at
! RHO-points. Finally, initialize boundary layer depth "hbl" and
! index "kbl" to the maximum (bottomed out) values.
!

! call alfabeta_tile (istr,iend,jstr,jend, imin,imax,
! & jmin,jmax, alpha,beta)
      do j=jmin,jmax
        do i=imin,imax

          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo


          Bo(i,j)=g*( alpha(i,j)*(stflx(i,j,itemp)-srflx(i,j))

     & -beta(i,j)*stflx(i,j,isalt)

     & )
          Bosol(i,j)=g*alpha(i,j)*srflx(i,j)


          ustar(i,j)=sqrt(0.5*sqrt( (sustr(i,j)+sustr(i+1,j))**2
     & +(svstr(i,j)+svstr(i,j+1))**2))
          hbl(i,j)=hbls(i,j) !<-- use old value as initial guess

          kbl(i)=0





          FC(i,N)=0.
          Cr(i,N)=0.
          Cr(i,0)=0.
        enddo !--> discard alpha,beta; keep Bo,Bosol to the very end.


!
!======================================
! Integral at W-points
!======================================
!
# 483 "R_tools_fort_routines_gula/get_hbl_cuc.F"
!================================
! Smoothing hbl/hbbl
!================================
!

      enddo !<-- j terminate j-loop only if smothing takes place.


      if (istr.eq.1) then
        do j=jmin,jmax
          hbl(istr-1,j)=hbl(istr,j)
        enddo
      endif
      if (iend.eq.Lm) then
        do j=jmin,jmax
          hbl(iend+1,j)=hbl(iend,j)
        enddo
      endif


      if (jstr.eq.1) then
        do i=imin,imax
          hbl(i,jstr-1)=hbl(i,jstr)
        enddo
      endif
      if (jend.eq.Mm) then
        do i=imin,imax
          hbl(i,jend+1)=hbl(i,jend)
        enddo
      endif

      if (istr.eq.1 .and. jstr.eq.1) then
        hbl(istr-1,jstr-1)=hbl(istr,jstr)
      endif
      if (istr.eq.1 .and. jend.eq.Mm) then
        hbl(istr-1,jend+1)=hbl(istr,jend)
      endif
      if (iend.eq.Lm .and. jstr.eq.1) then
        hbl(iend+1,jstr-1)=hbl(iend,jstr)
      endif
      if (iend.eq.Lm .and. jend.eq.Mm) then
        hbl(iend+1,jend+1)=hbl(iend,jend)
      endif


!
! Apply horizontal smoothing operator to hbl, while avoiding land-
! masked values which is accomplished by explessing everything in
! terms of elementary differences, subject to masking by U,V-rules.
! In the absense of masking the stencil of smoothing operator has
! the following weights, depending on coefficient settings in the
! code segment below:
!
! cff = 1/8, 1/4 cff = 1/12, 3/16 cff = 0, 1/8
!
! 1/16 1/8 1/16 1/32 1/8 1/32 1/8
!
! 1/8 1/4 1/8 1/8 3/8 1/8 1/8 1/4 1/8
!
! 1/16 1/8 1/16 1/32 1/8 1/32 1/8
!
! 2D 1-2-1-Hanning isotropic 5-point
! window smoother Laplacian Laplacian
!
! in all three cases the smoothing operator suppresses cheque-board
! mode in just one iteration; however, only the first one eliminates
! the 1D (flat-front) 2dx-modes in one iteration; the two others
! attenuate 1D 2dx-mode by factors of 1/4 and 1/2 per iteration
! respectively.
!
      do j=jstr-1,jend+1
        do i=istr,iend+1
          FX(i,j)=(hbl(i,j)-hbl(i-1,j))

     & *umask(i,j)

        enddo
      enddo
      do j=jstr,jend+1
        do i=istr-1,iend+1
          FE(i,j)=(hbl(i,j)-hbl(i,j-1))

     & *vmask(i,j)

        enddo
      enddo
      cff=1.D0/12.D0
      cff1=3.D0/16.D0
      do j=jstr,jend+1
        do i=istr,iend
          FE1(i,j)=FE(i,j)+cff*( FX(i+1,j)+FX(i,j-1)
     & -FX(i,j)-FX(i+1,j-1))
        enddo
      enddo
      do j=jstr,jend
        do i=istr,iend+1
          FX(i,j)=FX(i,j)+cff*( FE(i,j+1)+FE(i-1,j)
     & -FE(i,j)-FE(i-1,j+1))
        enddo
      enddo
      do j=jstr,jend
        do i=istr,iend
          hbl(i,j)=hbl(i,j)+cff1*( FX(i+1,j)-FX(i,j)
     & +FE1(i,j+1)-FE1(i,j))

          hbl(i,j)=hbl(i,j)*rmask(i,j)

        enddo !--> discard FX,FE,FE1
      enddo


!======================================



      return
      end
# 30 "R_tools_fort_cuc.F" 2

# 1 "R_tools_fort_routines_gula/lmd_vmix.F" 1




      subroutine lmd_vmix (Lm,Mm,N,u,v,z_r
     & ,bvf,Kv,Kt)
!
! This subroutine computes vertical mixing coefficients for momentum
! and tracers at the ocean interior using the Large, McWilliams and
! Doney (1994) mixing scheme.
!
! On Output:
! Kv vertical viscosity coefficient [m^2/s].
! Kt vertical diffusion coefficient for potential
! temperature [m^2/s].
! Ks vertical diffusion coefficient for salinity [m^2/s].
!
! Reference:
!
! Large, W.G., J.C. McWilliams, and S.C. Doney, 1994: A Review
! and model with a nonlocal boundary layer parameterization,
! Reviews of Geophysics, 32,363-403.
!
      implicit none

      integer Lm,Mm,N,imin,imax,jmin,jmax, i,j,k


      real*8 Rig(0:Lm+1,0:Mm+1,0:N),
     & Kv(0:Lm+1,0:Mm+1,0:N),
     & Kt(0:Lm+1,0:Mm+1,0:N)
! & ,Ks(0:Lm+1,0:Mm+1,0:N)

      real nu_sx, cff,dudz,dvdz

      real Ri0, nuwm, nuws, nu0m, nu0s, nu0c, lmd_nu, lmd_Rrho0,
     & lmd_nuf, lmd_fdd, lmd_tdd1, lmd_tdd2, lmd_tdd3, lmd_sdd1,
     & lmd_sdd2, lmd_sdd3, eps

      parameter ( Ri0=0.7,
     & nu0m=50.e-4,
     & nu0s=50.e-4,
     & nuwm=1.0e-4,
     & nuws=0.1e-4,
     & nu0c=0.1)

      parameter (eps=1.E-14)


      ! Variables IN
      real*8 z_r(0:Lm+1,0:Mm+1,N)
     & ,u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N)
     & ,bvf(0:Lm+1,0:Mm+1,0:N)



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!# include "compute_tile_bounds.h"
# 1 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h" 1
! Auxiliary module "compute_auxiliary_bounds.h":
!---------- ------ -----------------------------
! Compute derived bounds for the loop indices over a subdomain
! "tile". The extended bounds [labelled by suffix R] are designed to
! cover also the outer ghost points, if the subdomain "tile" is
! adjacent to a PHYSICAL boundary. (NOTE: istrR,iendR,jstrR,jendR
! computed by this module DO NOT COVER ghost points associated with
! periodic boundaries (if any) or with 2-point computational marhins
! of MPI subdomains.
!
! This module also computes loop-bounds for U- and V-type variables
! which belong to the interior of the computational domain. These are
! labelled by suffixes U,V and they step one grid point inward from
! the side of the subdomain adjacent to the physical boundary.
! Conversely, for an internal subdomain [which does not have segments
! of physical boundary] all variables with suffixes R,U,V are set to
! the same values are the corresponding non-suffixed variables.
!
! Because this module also contains type declarations for these
! bounds, it must be included just after the last type declaration
! inside a subroutine, but before the first executable statement.
!
# 35 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h"
      integer istrU, istrR, iendR
# 46 "R_tools_fort_routines_gula/compute_auxiliary_bounds.h"
      integer jstrV, jstrR, jendR


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      integer istr,iend,jstr,jend

        istr=1
        iend=Lm
        jstr=1
        jend=Mm

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




      if (istr.eq.1) then
        istrR=istr-1
        istrU=istr+1
      else
        istrR=istr
        istrU=istr
      endif
      if (iend.eq.Lm) then
        iendR=iend+1
      else
        iendR=iend
      endif



      if (jstr.eq.1) then
        jstrR=jstr-1
        jstrV=jstr+1
      else
        jstrR=jstr
        jstrV=jstr
      endif
      if (jend.eq.Mm) then
        jendR=jend+1
      else
        jendR=jend
      endif
# 60 "R_tools_fort_routines_gula/lmd_vmix.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



Cf2py intent(in) Lm,Mm,N,u, v,z_r, bvf
Cf2py intent(inout) Kv,Kt

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!







      if (istr.eq.1) then
        imin=istr
      else
        imin=istr-1
      endif
      if (iend.eq.Lm) then
        imax=iend
      else
        imax=iend+1
      endif





      if (jstr.eq.1) then
        jmin=jstr
      else
        jmin=jstr-1
      endif
      if (jend.eq.Mm) then
        jmax=jend
      else
        jmax=jend+1
      endif
# 111 "R_tools_fort_routines_gula/lmd_vmix.F"
! ! Compute horizontal velocity
! Compute local Richardson number: ! shear (du/dz)^2+(dv/dz)^2 at
!-------- ----- ---------- ------- ! horizontal RHO- and vertical
! ! W-points. Then compute gradient
      do k=1,N-1 ! Richardson number (already
        do j=jmin,jmax ! divided by its critical value.
          do i=imin,imax
            cff=0.5/(z_r(i,j,k+1)-z_r(i,j,k))
            dudz=cff*( u(i ,j,k+1)-u(i ,j,k)
     & +u(i+1,j,k+1)-u(i+1,j,k))
            dvdz=cff*( v(i,j ,k+1)-v(i,j ,k)
     & +v(i,j+1,k+1)-v(i,j+1,k))

            Rig(i,j,k)=bvf(i,j,k)/( Ri0*max(
     & dudz*dudz+dvdz*dvdz, 1.E-10 ))
          enddo
        enddo



        if (istr.eq.1) then
          do j=jmin,jmax
            Rig(istr-1,j,k)=Rig(istr,j,k)
          enddo
        endif
        if (iend.eq.Lm) then
          do j=jmin,jmax
            Rig(iend+1,j,k)=Rig(iend,j,k)
          enddo
        endif


        if (jstr.eq.1) then
          do i=imin,imax
            Rig(i,jstr-1,k)=Rig(i,jstr,k)
          enddo
        endif
        if (jend.eq.Mm) then
          do i=imin,imax
            Rig(i,jend+1,k)=Rig(i,jend,k)
          enddo
        endif

        if (istr.eq.1 .and.
     & jstr.eq.1) then
          Rig(istr-1,jstr-1,k)=Rig(istr,jstr,k)
        endif
        if (istr.eq.1 .and.
     & jend.eq.Mm) then
          Rig(istr-1,jend+1,k)=Rig(istr,jend,k)
        endif
        if (iend.eq.Lm .and.
     & jstr.eq.1) then
          Rig(iend+1,jstr-1,k)=Rig(iend,jstr,k)
        endif
        if (iend.eq.Lm .and.
     & jend.eq.Mm) then
          Rig(iend+1,jend+1,k)=Rig(iend,jend,k)
        endif


                                   ! Smooth Rig horizontally: use
        do j=jstr-1,jend ! array Rig(:,:,0) as scratch.
          do i=istr-1,iend
            Rig(i,j,0)=0.25*(Rig(i,j ,k)+Rig(i+1,j ,k)
     & +Rig(i,j+1,k)+Rig(i+1,j+1,k))
          enddo
        enddo
        do j=jstr,jend
          do i=istr,iend
            Rig(i,j,k)=0.25*(Rig(i,j ,0)+Rig(i-1,j ,0)
     & +Rig(i,j-1,0)+Rig(i-1,j-1,0))
          enddo
        enddo !--> discard Rig(:,:,0)


!
! Compute "interior" viscosities and diffusivities everywhere
! as the superposition of three processes: local Richardson number
! instability due to resolved vertical shear, internal wave breaking,
! and double diffusion.
!
        do j=jstr,jend
          do i=istr,iend

            cff=min(1., max(0., Rig(i,j,k))) ! Compute mixing die
            nu_sx=1. - cff*cff ! to shear instability
            nu_sx=nu_sx*nu_sx*nu_sx ! and internal wave
                                              ! breaking.
            Kv(i,j,k)=nuwm + nu0m*nu_sx
            Kt(i,j,k)=nuws + nu0s*nu_sx





          enddo
        enddo
      enddo ! <-- k

!
! Pad out surface and bottom values for lmd_blmix calculations.
! The interior values used here may not be the best values to
! use for the padding.
!
! do j=jstr,jend
! do i=istr,iend
! Kv(i,j,N)=Kv(i,j,N-1)
! Ks(i,j,N)=Ks(i,j,N-1)
! Kt(i,j,N)=Kt(i,j,N-1)
! Kv(i,j,0)=Kv(i,j, 1)
! Ks(i,j,0)=Ks(i,j, 1)
! Kt(i,j,0)=Kt(i,j, 1)
! enddo
! enddo




      return
      end
# 32 "R_tools_fort_cuc.F" 2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
