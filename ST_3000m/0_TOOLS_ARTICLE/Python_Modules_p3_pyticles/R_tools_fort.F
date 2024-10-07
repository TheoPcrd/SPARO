# 1 "R_tools_fort.F"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "R_tools_fort.F"

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




! Included are:
!
! subroutine rho_eos(Lm,Mm,N, T,S, z_r,z_w,rho0,
! & rho1,qp1,rho,bvf)
! subroutine rho_grad(Lm,Mm,N, T,S, z_r,z_w,rho0,pm,pn,
! & rho1,qp1,drdz,drdx,drdy)
! subroutine sigma_to_z_intr (Lm,Mm,N, nz, z_r, z_w, rmask, var,
! & z_lev, var_zlv, imin,jmin,kmin, FillValue)
! subroutine zlevs(Lm,Mm,N, h,zeta, hc, Cs_r, Cs_w,z_r,z_w)
!
!

# 1 "R_tools_fort_routines/cppdefs.h" 1
# 15 "R_tools_fort_routines/cppdefs.h"
c-# define PV_CUBIC
# 60 "R_tools_fort_routines/cppdefs.h"
# 1 "R_tools_fort_routines/set_global_definitions.h" 1
# 16 "R_tools_fort_routines/set_global_definitions.h"
c--#define ALLOW_SINGLE_BLOCK_MODE
# 260 "R_tools_fort_routines/set_global_definitions.h"
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
# 61 "R_tools_fort_routines/cppdefs.h" 2
# 32 "R_tools_fort.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Compute density anomaly (adapted from rho_eos.F)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 1 "R_tools_fort_routines/rho_eos.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Compute density anomaly (adapted from rho_eos.F)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine rho_eos(Lm,Mm,N, T,S, z_r,z_w,rho0, rho)



!
! Compute density anomaly from T,S via Equation Of State (EOS) for
!-------- ------- ------- ---- for seawater. Following Jackett and
! McDougall, 1995, physical EOS is assumed to have form
!
! rho0 + rho1(T,S)
! rho(T,S,z) = ------------------------ (1)
! 1 - 0.1*|z|/K(T,S,|z|)
!
! where rho1(T,S) is sea-water density perturbation[kg/m^3] at
! standard pressure of 1 Atm (sea surface); |z| is absolute depth,
! i.e. distance from free-surface to the point at which density is
! computed, and
!
! K(T,S,|z|) = K00 + K01(T,S) + K1(T,S)*|z| + K2(T,S)*|z|^2. (2)
!
! To reduce errors of pressure-gradient scheme associated with
! nonlinearity of compressibility effects, as well as to reduce
! roundoff errors, the dominant part of density profile,
!
! rho0
! ---------------- (3)
! 1 - 0.1|z|/K00
!
! is removed from from (1). [Since (3) is purely a function of z,
! it does not contribute to pressure gradient.] This results in
!
! rho1 - rho0*[K01+K1*|z|+K2*|z|^2]/[K00-0.1|z|]
! rho1 + 0.1|z| -----------------------------------------------
! K00 + K01 + (K1-0.1)*|z| + K2*|z|^2
! (4)
! which is suitable for pressure-gradient calculation.
!
! Optionally, if CPP-switch is defined, term proportional
! to |z| is linearized using smallness 0.1|z|/[K00 + K01] << 1 and
! the resultant EOS has form
!
! rho(T,S,z) = rho1(T,S) + qp1(T,S)*|z| (5)
!
! where
! rho1(T,S) - rho0*K01(T,S)/K00
! qp1(T,S)= 0.1 ------------------------------- (6)
! K00 + K01(T,S)
!
! is stored in a special array.
!
! This splitting allows representation of spatial derivatives (and
! also differences) of density as sum of adiabatic derivatives and
! compressible part according to
!
! d rho d rho1 d qp1 d |z|
! ------- = -------- + |z| * ------- + qp1 * ------- (7)
! d x,s d x,s d x,s d x,s
!
! |<----- adiabatic ----->| |<- compress ->|
!
! so that constraining of adiabatic derivative for monotonicity is
! equivalent to enforcement of physically stable stratification.
! [This separation and constraining algorithm is subsequently used
! in computation of pressure gradient within prsgrd32ACx-family
! schemes.]
!
! If so prescribed compute the Brunt-Vaisala frequency [1/s^2] at
! horizontal RHO-points and vertical W-points,
!
! g d rho |
! bvf^2 = - ------ ------- | (8)
! rho0 d z | adiabatic
!
! where density anomaly difference is computed by adiabatically
! rising/lowering the water parcel from RHO point above/below to
! the W-point depth at "z_w".
!
! WARNING: Shared target arrays in the code below: "rho1",
! "bvf" (if needed), and
!
! is defined: "qp1" ["rho" does not exist]
! not defined "rho" ["qp1" does not exist]
!
!
! Reference: Jackett, D. R. and T. J. McDougall, 1995, Minimal
! Adjustment of Hydrostatic Profiles to Achieve Static
! Stability. J. Atmos. Ocean. Tec., vol. 12, pp. 381-389.
!
! << This equation of state formulation has been derived by Jackett
! and McDougall (1992), unpublished manuscript, CSIRO, Australia. It
! computes in-situ density anomaly as a function of potential
! temperature (Celsius) relative to the surface, salinity (PSU),
! and depth (meters). It assumes no pressure variation along
! geopotential surfaces, that is, depth and pressure are
! interchangeable. >>
! John Wilkin, 29 July 92
!


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k

      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & rho(0:Lm+1,0:Mm+1,N),
     & rho1(0:Lm+1,0:Mm+1,N), qp1(0:Lm+1,0:Mm+1,N),
     & dpth,cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00

      real*8, parameter :: r00=999.842594, r01=6.793952E-2,
     & r02=-9.095290E-3, r03=1.001685E-4, r04=-1.120083E-6,
     & r05=6.536332E-9,
     & r10=0.824493, r11=-4.08990E-3, r12=7.64380E-5,
     & r13=-8.24670E-7, r14=5.38750E-9,
     & rS0=-5.72466E-3, rS1=1.02270E-4, rS2=-1.65460E-6,
     & r20=4.8314E-4,
     & K00=19092.56, K01=209.8925, K02=-3.041638,
     & K03=-1.852732e-3, K04=-1.361629e-5,
     & K10=104.4077, K11=-6.500517, K12=0.1553190,
     & K13=2.326469e-4,
     & KS0=-5.587545, KS1=+0.7390729, KS2=-1.909078e-2,
     & qp2=0.0000172, g=9.81


      real rho1_0, K0_Duk


      integer numthreads, trd, chunk_size, margin, jstr,jend


Cf2py intent(in) Lm,Mm,N, T,S, z_r,z_w, rho0
Cf2py intent(out) rho



!!!!!!!!!!!!!!!!!!




      Tt=3.8D0
      Ts=34.5D0
      sqrtTs=sqrt(Ts)

      dr00=r00-1000.D0

      rho1_0=dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*( r04+Tt*r05 ))))
     & +Ts*( R10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*( rS0+Tt*( rS1+Tt*rS2 ))+Ts*r20 )

      K0_Duk= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))


      dr00=r00-rho0

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


      numthreads=1
C$ numthreads=omp_get_num_threads()
      trd=0
C$ trd=omp_get_thread_num()
      chunk_size=(jmax-jmin + numthreads)/numthreads
      margin=(chunk_size*numthreads -jmax+jmin-1)/2
      jstr=max( trd *chunk_size -margin, jmin )
      jend=min( (trd+1)*chunk_size-1-margin, jmax )




      do j=jstr,jend
        do k=1,N
          do i=imin,imax
            Tt=T(i,j,k)

            Ts=max(S(i,j,k), 0.)
            sqrtTs=sqrt(Ts)

            rho1(i,j,k)=( dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*(
     & r04+Tt*r05 ))))
     & +Ts*( r10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*(rS0+Tt*(
     & rS1+Tt*rS2 ))+Ts*r20 ))

            K0= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))





            qp1(i,j,k)= 0.1D0*(rho0+rho1(i,j,k))*(K0_Duk-K0)
     & /((K00+K0)*(K00+K0_Duk))





         rho(i,j,k) = rho1(i,j,k) + qp1(i,j,k)*(z_w(i,j,N)-z_r(i,j,k))


          enddo
        enddo



      enddo ! <-- j

      return
      end
# 38 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/rho1_eos.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Compute density anomaly (adapted from rho_eos.F)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine rho1_eos(Lm,Mm,N, T,S, z_r,rho0, rho)



!
! Compute density anomaly from T,S via Equation Of State (EOS) for
!-------- ------- ------- ---- for seawater. Following Jackett and
! McDougall, 1995, physical EOS is assumed to have form
!
! rho0 + rho1(T,S)
! rho(T,S,z) = ------------------------ (1)
! 1 - 0.1*|z|/K(T,S,|z|)
!
! where rho1(T,S) is sea-water density perturbation[kg/m^3] at
! standard pressure of 1 Atm (sea surface); |z| is absolute depth,
! i.e. distance from free-surface to the point at which density is
! computed, and
!
! K(T,S,|z|) = K00 + K01(T,S) + K1(T,S)*|z| + K2(T,S)*|z|^2. (2)
!
! To reduce errors of pressure-gradient scheme associated with
! nonlinearity of compressibility effects, as well as to reduce
! roundoff errors, the dominant part of density profile,
!
! rho0
! ---------------- (3)
! 1 - 0.1|z|/K00
!
! is removed from from (1). [Since (3) is purely a function of z,
! it does not contribute to pressure gradient.] This results in
!
! rho1 - rho0*[K01+K1*|z|+K2*|z|^2]/[K00-0.1|z|]
! rho1 + 0.1|z| -----------------------------------------------
! K00 + K01 + (K1-0.1)*|z| + K2*|z|^2
! (4)
! which is suitable for pressure-gradient calculation.
!
! Optionally, if CPP-switch is defined, term proportional
! to |z| is linearized using smallness 0.1|z|/[K00 + K01] << 1 and
! the resultant EOS has form
!
! rho(T,S,z) = rho1(T,S) + qp1(T,S)*|z| (5)
!
! where
! rho1(T,S) - rho0*K01(T,S)/K00
! qp1(T,S)= 0.1 ------------------------------- (6)
! K00 + K01(T,S)
!
! is stored in a special array.
!
! This splitting allows representation of spatial derivatives (and
! also differences) of density as sum of adiabatic derivatives and
! compressible part according to
!
! d rho d rho1 d qp1 d |z|
! ------- = -------- + |z| * ------- + qp1 * ------- (7)
! d x,s d x,s d x,s d x,s
!
! |<----- adiabatic ----->| |<- compress ->|
!
! so that constraining of adiabatic derivative for monotonicity is
! equivalent to enforcement of physically stable stratification.
! [This separation and constraining algorithm is subsequently used
! in computation of pressure gradient within prsgrd32ACx-family
! schemes.]
!
! If so prescribed compute the Brunt-Vaisala frequency [1/s^2] at
! horizontal RHO-points and vertical W-points,
!
! g d rho |
! bvf^2 = - ------ ------- | (8)
! rho0 d z | adiabatic
!
! where density anomaly difference is computed by adiabatically
! rising/lowering the water parcel from RHO point above/below to
! the W-point depth at "z_w".
!
! WARNING: Shared target arrays in the code below: "rho1",
! "bvf" (if needed), and
!
! is defined: "qp1" ["rho" does not exist]
! not defined "rho" ["qp1" does not exist]
!
!
! Reference: Jackett, D. R. and T. J. McDougall, 1995, Minimal
! Adjustment of Hydrostatic Profiles to Achieve Static
! Stability. J. Atmos. Ocean. Tec., vol. 12, pp. 381-389.
!
! << This equation of state formulation has been derived by Jackett
! and McDougall (1992), unpublished manuscript, CSIRO, Australia. It
! computes in-situ density anomaly as a function of potential
! temperature (Celsius) relative to the surface, salinity (PSU),
! and depth (meters). It assumes no pressure variation along
! geopotential surfaces, that is, depth and pressure are
! interchangeable. >>
! John Wilkin, 29 July 92
!


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k

      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N),
     & rho(0:Lm+1,0:Mm+1,N),
     & dpth,cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00


      real*8, parameter :: r00=999.842594, r01=6.793952E-2,
     & r02=-9.095290E-3, r03=1.001685E-4, r04=-1.120083E-6,
     & r05=6.536332E-9,
     & r10=0.824493, r11=-4.08990E-3, r12=7.64380E-5,
     & r13=-8.24670E-7, r14=5.38750E-9,
     & rS0=-5.72466E-3, rS1=1.02270E-4, rS2=-1.65460E-6,
     & r20=4.8314E-4,
     & K00=19092.56, K01=209.8925, K02=-3.041638,
     & K03=-1.852732e-3, K04=-1.361629e-5,
     & K10=104.4077, K11=-6.500517, K12=0.1553190,
     & K13=2.326469e-4,
     & KS0=-5.587545, KS1=+0.7390729, KS2=-1.909078e-2,
     & qp2=0.0000172, g=9.81


      real rho1_0, K0_Duk


      integer numthreads, trd, chunk_size, margin, jstr,jend


Cf2py intent(in) Lm,Mm,N, T,S, z_r,z_w, rho0
Cf2py intent(out) rho



!!!!!!!!!!!!!!!!!!




      Tt=3.8D0
      Ts=34.5D0
      sqrtTs=sqrt(Ts)

      dr00=r00-1000.D0

      rho1_0=dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*( r04+Tt*r05 ))))
     & +Ts*( R10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*( rS0+Tt*( rS1+Tt*rS2 ))+Ts*r20 )

      K0_Duk= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))


      dr00=r00-rho0



      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


      numthreads=1
C$ numthreads=omp_get_num_threads()
      trd=0
C$ trd=omp_get_thread_num()
      chunk_size=(jmax-jmin + numthreads)/numthreads
      margin=(chunk_size*numthreads -jmax+jmin-1)/2
      jstr=max( trd *chunk_size -margin, jmin )
      jend=min( (trd+1)*chunk_size-1-margin, jmax )



      do j=jstr,jend
        do k=1,N
          do i=imin,imax
            Tt=T(i,j,k)

            Ts=max(S(i,j,k), 0.)
            sqrtTs=sqrt(Ts)

            rho(i,j,k)=( dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*(
     & r04+Tt*r05 ))))
     & +Ts*( r10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*(rS0+Tt*(
     & rS1+Tt*rS2 ))+Ts*r20 ))




          enddo
        enddo



      enddo ! <-- j



      return
      end
# 40 "R_tools_fort.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Compute Brunt-Vaissala frequency (adapted from rho_eos.F)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 1 "R_tools_fort_routines/bvf_eos.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Compute density anomaly (adapted from rho_eos.F)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine bvf_eos(Lm,Mm,N, T,S, z_r,z_w,rho0,bvf)



!
! Compute density anomaly from T,S via Equation Of State (EOS) for
!-------- ------- ------- ---- for seawater. Following Jackett and
! McDougall, 1995, physical EOS is assumed to have form
!
! rho0 + rho1(T,S)
! rho(T,S,z) = ------------------------ (1)
! 1 - 0.1*|z|/K(T,S,|z|)
!
! where rho1(T,S) is sea-water density perturbation[kg/m^3] at
! standard pressure of 1 Atm (sea surface); |z| is absolute depth,
! i.e. distance from free-surface to the point at which density is
! computed, and
!
! K(T,S,|z|) = K00 + K01(T,S) + K1(T,S)*|z| + K2(T,S)*|z|^2. (2)
!
! To reduce errors of pressure-gradient scheme associated with
! nonlinearity of compressibility effects, as well as to reduce
! roundoff errors, the dominant part of density profile,
!
! rho0
! ---------------- (3)
! 1 - 0.1|z|/K00
!
! is removed from from (1). [Since (3) is purely a function of z,
! it does not contribute to pressure gradient.] This results in
!
! rho1 - rho0*[K01+K1*|z|+K2*|z|^2]/[K00-0.1|z|]
! rho1 + 0.1|z| -----------------------------------------------
! K00 + K01 + (K1-0.1)*|z| + K2*|z|^2
! (4)
! which is suitable for pressure-gradient calculation.
!
! Optionally, if CPP-switch is defined, term proportional
! to |z| is linearized using smallness 0.1|z|/[K00 + K01] << 1 and
! the resultant EOS has form
!
! rho(T,S,z) = rho1(T,S) + qp1(T,S)*|z| (5)
!
! where
! rho1(T,S) - rho0*K01(T,S)/K00
! qp1(T,S)= 0.1 ------------------------------- (6)
! K00 + K01(T,S)
!
! is stored in a special array.
!
! This splitting allows representation of spatial derivatives (and
! also differences) of density as sum of adiabatic derivatives and
! compressible part according to
!
! d rho d rho1 d qp1 d |z|
! ------- = -------- + |z| * ------- + qp1 * ------- (7)
! d x,s d x,s d x,s d x,s
!
! |<----- adiabatic ----->| |<- compress ->|
!
! so that constraining of adiabatic derivative for monotonicity is
! equivalent to enforcement of physically stable stratification.
! [This separation and constraining algorithm is subsequently used
! in computation of pressure gradient within prsgrd32ACx-family
! schemes.]
!
! If so prescribed compute the Brunt-Vaisala frequency [1/s^2] at
! horizontal RHO-points and vertical W-points,
!
! g d rho |
! bvf^2 = - ------ ------- | (8)
! rho0 d z | adiabatic
!
! where density anomaly difference is computed by adiabatically
! rising/lowering the water parcel from RHO point above/below to
! the W-point depth at "z_w".
!
! WARNING: Shared target arrays in the code below: "rho1",
! "bvf" (if needed), and
!
! is defined: "qp1" ["rho" does not exist]
! not defined "rho" ["qp1" does not exist]
!
!
! Reference: Jackett, D. R. and T. J. McDougall, 1995, Minimal
! Adjustment of Hydrostatic Profiles to Achieve Static
! Stability. J. Atmos. Ocean. Tec., vol. 12, pp. 381-389.
!
! << This equation of state formulation has been derived by Jackett
! and McDougall (1992), unpublished manuscript, CSIRO, Australia. It
! computes in-situ density anomaly as a function of potential
! temperature (Celsius) relative to the surface, salinity (PSU),
! and depth (meters). It assumes no pressure variation along
! geopotential surfaces, that is, depth and pressure are
! interchangeable. >>
! John Wilkin, 29 July 92
!


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k

      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & rho(0:Lm+1,0:Mm+1,N), bvf(0:Lm+1,0:Mm+1,0:N),
     & rho1(0:Lm+1,0:Mm+1,N), qp1(0:Lm+1,0:Mm+1,N),
     & dpth,cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00

      real*8, parameter :: r00=999.842594, r01=6.793952E-2,
     & r02=-9.095290E-3, r03=1.001685E-4, r04=-1.120083E-6,
     & r05=6.536332E-9,
     & r10=0.824493, r11=-4.08990E-3, r12=7.64380E-5,
     & r13=-8.24670E-7, r14=5.38750E-9,
     & rS0=-5.72466E-3, rS1=1.02270E-4, rS2=-1.65460E-6,
     & r20=4.8314E-4,
     & K00=19092.56, K01=209.8925, K02=-3.041638,
     & K03=-1.852732e-3, K04=-1.361629e-5,
     & K10=104.4077, K11=-6.500517, K12=0.1553190,
     & K13=2.326469e-4,
     & KS0=-5.587545, KS1=+0.7390729, KS2=-1.909078e-2,
     & qp2=0.0000172, g=9.81


      real rho1_0, K0_Duk


      integer numthreads, trd, chunk_size, margin, jstr,jend


Cf2py intent(in) Lm,Mm,N, T,S, z_r,z_w, rho0
Cf2py intent(out) bvf



!!!!!!!!!!!!!!!!!!




      Tt=3.8D0
      Ts=34.5D0
      sqrtTs=sqrt(Ts)

      dr00=r00-1000.D0

      rho1_0=dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*( r04+Tt*r05 ))))
     & +Ts*( R10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*( rS0+Tt*( rS1+Tt*rS2 ))+Ts*r20 )

      K0_Duk= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))


      dr00=r00-rho0

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


      numthreads=1
C$ numthreads=omp_get_num_threads()
      trd=0
C$ trd=omp_get_thread_num()
      chunk_size=(jmax-jmin + numthreads)/numthreads
      margin=(chunk_size*numthreads -jmax+jmin-1)/2
      jstr=max( trd *chunk_size -margin, jmin )
      jend=min( (trd+1)*chunk_size-1-margin, jmax )




      do j=jstr,jend
        do k=1,N
          do i=imin,imax
            Tt=T(i,j,k)

            Ts=max(S(i,j,k), 0.)
            sqrtTs=sqrt(Ts)

            rho1(i,j,k)=( dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*(
     & r04+Tt*r05 ))))
     & +Ts*( r10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*(rS0+Tt*(
     & rS1+Tt*rS2 ))+Ts*r20 ))

            K0= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))





            qp1(i,j,k)= 0.1D0*(rho0+rho1(i,j,k))*(K0_Duk-K0)
     & /((K00+K0)*(K00+K0_Duk))






          enddo
        enddo

        cff=g/rho0
        do k=1,N-1
          do i=imin,imax

            dpth=z_w(i,j,N)-0.5*(z_r(i,j,k+1)+z_r(i,j,k))
            cff2=( rho1(i,j,k+1)-rho1(i,j,k) ! Elementary
     & +(qp1(i,j,k+1)-qp1(i,j,k)) ! adiabatic
     & *dpth*(1.-2.*qp2*dpth) ! difference
     & )


            bvf(i,j,k)=-cff*cff2 / (z_r(i,j,k+1)-z_r(i,j,k))

          enddo
        enddo




        do i=imin,imax
          bvf (i,j,N) = bvf (i,j,N-1)
          bvf (i,j,0) = bvf (i,j, 1)
        enddo




      enddo ! <-- j

      return
      end
# 46 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/bvf_lineos.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Compute density anomaly (adapted from rho_eos.F)
!! without and with linear EOS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine bvf_lineos(Lm,Mm,N, T,S,Tcoef, Scoef, z_r,R0,rho0,bvf)

      implicit none
      integer Lm,Mm,N, i,j,k,
     & istrR,iendR,jstrR,jendR

      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N),
     & rho(0:Lm+1,0:Mm+1,N), bvf(0:Lm+1,0:Mm+1,0:N),
     & cff, rho0, R0,Tcoef, Scoef

      real*8, parameter :: qp2=0.0000172, g=9.81

      integer numthreads, trd, chunk_size, margin, jstr,jend


Cf2py intent(in) Lm,Mm,N,T,S,Tcoef, Scoef, z_r,R0,rho0
Cf2py intent(out) bvf


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! R0 Coefficients for linear Equation of State (EOS)
! T0,Tcoef
! S0,Scoef rho = R0 - Tcoef*(T-T0) + Scoef*(S-S0)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      istrR=0
      iendR=Lm+1
      jstrR=0
      jendR=Mm+1

      do j=jstrR,jendR
        do k=1,N
          do i=istrR,iendR
            rho(i,j,k)=R0 -Tcoef*T(i,j,k)

     & +Scoef*S(i,j,k)


          enddo
        enddo


        cff=g/rho0
        do k=1,N-1
          do i=istrR,iendR
            bvf(i,j,k)=cff*(rho(i,j,k)-rho(i,j,k+1))
     & /(z_r(i,j,k+1)-z_r(i,j,k))

          enddo
        enddo
        do i=istrR,iendR
          bvf(i,j,N)=bvf(i,j,N-1)
          bvf(i,j,0)=bvf(i,j, 1)
        enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      enddo ! <-- j
      return
      end
# 48 "R_tools_fort.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Compute buoyancy (adapted from rho_eos.F)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 1 "R_tools_fort_routines/get_buoy.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Compute density anomaly (adapted from rho_eos.F)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_buoy(Lm,Mm,N, T,S, z_r,z_w,rho0, buoy)



!
! Compute density anomaly from T,S via Equation Of State (EOS) for
!-------- ------- ------- ---- for seawater. Following Jackett and
! McDougall, 1995, physical EOS is assumed to have form
!
! rho0 + rho1(T,S)
! rho(T,S,z) = ------------------------ (1)
! 1 - 0.1*|z|/K(T,S,|z|)
!
! where rho1(T,S) is sea-water density perturbation[kg/m^3] at
! standard pressure of 1 Atm (sea surface); |z| is absolute depth,
! i.e. distance from free-surface to the point at which density is
! computed, and
!
! K(T,S,|z|) = K00 + K01(T,S) + K1(T,S)*|z| + K2(T,S)*|z|^2. (2)
!
! To reduce errors of pressure-gradient scheme associated with
! nonlinearity of compressibility effects, as well as to reduce
! roundoff errors, the dominant part of density profile,
!
! rho0
! ---------------- (3)
! 1 - 0.1|z|/K00
!
! is removed from from (1). [Since (3) is purely a function of z,
! it does not contribute to pressure gradient.] This results in
!
! rho1 - rho0*[K01+K1*|z|+K2*|z|^2]/[K00-0.1|z|]
! rho1 + 0.1|z| -----------------------------------------------
! K00 + K01 + (K1-0.1)*|z| + K2*|z|^2
! (4)
! which is suitable for pressure-gradient calculation.
!
! Optionally, if CPP-switch is defined, term proportional
! to |z| is linearized using smallness 0.1|z|/[K00 + K01] << 1 and
! the resultant EOS has form
!
! rho(T,S,z) = rho1(T,S) + qp1(T,S)*|z| (5)
!
! where
! rho1(T,S) - rho0*K01(T,S)/K00
! qp1(T,S)= 0.1 ------------------------------- (6)
! K00 + K01(T,S)
!
! is stored in a special array.
!
! This splitting allows representation of spatial derivatives (and
! also differences) of density as sum of adiabatic derivatives and
! compressible part according to
!
! d rho d rho1 d qp1 d |z|
! ------- = -------- + |z| * ------- + qp1 * ------- (7)
! d x,s d x,s d x,s d x,s
!
! |<----- adiabatic ----->| |<- compress ->|
!
! so that constraining of adiabatic derivative for monotonicity is
! equivalent to enforcement of physically stable stratification.
! [This separation and constraining algorithm is subsequently used
! in computation of pressure gradient within prsgrd32ACx-family
! schemes.]
!
! If so prescribed compute the Brunt-Vaisala frequency [1/s^2] at
! horizontal RHO-points and vertical W-points,
!
! g d rho |
! bvf^2 = - ------ ------- | (8)
! rho0 d z | adiabatic
!
! where density anomaly difference is computed by adiabatically
! rising/lowering the water parcel from RHO point above/below to
! the W-point depth at "z_w".
!
! WARNING: Shared target arrays in the code below: "rho1",
! "bvf" (if needed), and
!
! is defined: "qp1" ["rho" does not exist]
! not defined "rho" ["qp1" does not exist]
!
!
! Reference: Jackett, D. R. and T. J. McDougall, 1995, Minimal
! Adjustment of Hydrostatic Profiles to Achieve Static
! Stability. J. Atmos. Ocean. Tec., vol. 12, pp. 381-389.
!
! << This equation of state formulation has been derived by Jackett
! and McDougall (1992), unpublished manuscript, CSIRO, Australia. It
! computes in-situ density anomaly as a function of potential
! temperature (Celsius) relative to the surface, salinity (PSU),
! and depth (meters). It assumes no pressure variation along
! geopotential surfaces, that is, depth and pressure are
! interchangeable. >>
! John Wilkin, 29 July 92
!


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k

      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & buoy(0:Lm+1,0:Mm+1,N),
     & rho1, qp1,
     & dpth,cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00

      real*8, parameter :: r00=999.842594, r01=6.793952E-2,
     & r02=-9.095290E-3, r03=1.001685E-4, r04=-1.120083E-6,
     & r05=6.536332E-9,
     & r10=0.824493, r11=-4.08990E-3, r12=7.64380E-5,
     & r13=-8.24670E-7, r14=5.38750E-9,
     & rS0=-5.72466E-3, rS1=1.02270E-4, rS2=-1.65460E-6,
     & r20=4.8314E-4,
     & K00=19092.56, K01=209.8925, K02=-3.041638,
     & K03=-1.852732e-3, K04=-1.361629e-5,
     & K10=104.4077, K11=-6.500517, K12=0.1553190,
     & K13=2.326469e-4,
     & KS0=-5.587545, KS1=+0.7390729, KS2=-1.909078e-2,
     & qp2=0.0000172, g=9.81


      real rho1_0, K0_Duk


      integer numthreads, trd, chunk_size, margin, jstr,jend


Cf2py intent(in) Lm,Mm,N, T,S, z_r,z_w, rho0
Cf2py intent(out) buoy



!!!!!!!!!!!!!!!!!!




      Tt=3.8D0
      Ts=34.5D0
      sqrtTs=sqrt(Ts)

      dr00=r00-1000.D0

      rho1_0=dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*( r04+Tt*r05 ))))
     & +Ts*( R10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*( rS0+Tt*( rS1+Tt*rS2 ))+Ts*r20 )

      K0_Duk= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))


      dr00=r00-rho0

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


      numthreads=1
C$ numthreads=omp_get_num_threads()
      trd=0
C$ trd=omp_get_thread_num()
      chunk_size=(jmax-jmin + numthreads)/numthreads
      margin=(chunk_size*numthreads -jmax+jmin-1)/2
      jstr=max( trd *chunk_size -margin, jmin )
      jend=min( (trd+1)*chunk_size-1-margin, jmax )


      cff = g/rho0

      do j=jstr,jend
        do k=1,N
          do i=imin,imax
            Tt=T(i,j,k)

            Ts=max(S(i,j,k), 0.)
            sqrtTs=sqrt(Ts)

            rho1=( dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*(
     & r04+Tt*r05 ))))
     & +Ts*( r10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*(rS0+Tt*(
     & rS1+Tt*rS2 ))+Ts*r20 ))

            K0= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))



            qp1= 0.1D0*(rho0+rho1)*(K0_Duk-K0)
     & /((K00+K0)*(K00+K0_Duk))


            buoy(i,j,k) = -cff * (rho1 + qp1*(z_w(i,j,N)-z_r(i,j,k)))


          enddo
        enddo
      enddo ! <-- j

      return
      end
# 54 "R_tools_fort.F" 2


# 1 "R_tools_fort_routines/get_buoy1.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Compute density anomaly (adapted from rho1_eos.F)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_buoy1(Lm,Mm,N, T,S, z_r,z_w,rho0, buoy)



!
! Compute density anomaly from T,S via Equation Of State (EOS) for
!-------- ------- ------- ---- for seawater. Following Jackett and
! McDougall, 1995, physical EOS is assumed to have form
!
! rho0 + rho1(T,S)
! rho(T,S,z) = ------------------------ (1)
! 1 - 0.1*|z|/K(T,S,|z|)
!
! where rho1(T,S) is sea-water density perturbation[kg/m^3] at
! standard pressure of 1 Atm (sea surface); |z| is absolute depth,
! i.e. distance from free-surface to the point at which density is
! computed, and
!
! K(T,S,|z|) = K00 + K01(T,S) + K1(T,S)*|z| + K2(T,S)*|z|^2. (2)
!
! To reduce errors of pressure-gradient scheme associated with
! nonlinearity of compressibility effects, as well as to reduce
! roundoff errors, the dominant part of density profile,
!
! rho0
! ---------------- (3)
! 1 - 0.1|z|/K00
!
! is removed from from (1). [Since (3) is purely a function of z,
! it does not contribute to pressure gradient.] This results in
!
! rho1 - rho0*[K01+K1*|z|+K2*|z|^2]/[K00-0.1|z|]
! rho1 + 0.1|z| -----------------------------------------------
! K00 + K01 + (K1-0.1)*|z| + K2*|z|^2
! (4)
! which is suitable for pressure-gradient calculation.
!
! Optionally, if CPP-switch is defined, term proportional
! to |z| is linearized using smallness 0.1|z|/[K00 + K01] << 1 and
! the resultant EOS has form
!
! rho(T,S,z) = rho1(T,S) + qp1(T,S)*|z| (5)
!
! where
! rho1(T,S) - rho0*K01(T,S)/K00
! qp1(T,S)= 0.1 ------------------------------- (6)
! K00 + K01(T,S)
!
! is stored in a special array.
!
! This splitting allows representation of spatial derivatives (and
! also differences) of density as sum of adiabatic derivatives and
! compressible part according to
!
! d rho d rho1 d qp1 d |z|
! ------- = -------- + |z| * ------- + qp1 * ------- (7)
! d x,s d x,s d x,s d x,s
!
! |<----- adiabatic ----->| |<- compress ->|
!
! so that constraining of adiabatic derivative for monotonicity is
! equivalent to enforcement of physically stable stratification.
! [This separation and constraining algorithm is subsequently used
! in computation of pressure gradient within prsgrd32ACx-family
! schemes.]
!
! If so prescribed compute the Brunt-Vaisala frequency [1/s^2] at
! horizontal RHO-points and vertical W-points,
!
! g d rho |
! bvf^2 = - ------ ------- | (8)
! rho0 d z | adiabatic
!
! where density anomaly difference is computed by adiabatically
! rising/lowering the water parcel from RHO point above/below to
! the W-point depth at "z_w".
!
! WARNING: Shared target arrays in the code below: "rho1",
! "bvf" (if needed), and
!
! is defined: "qp1" ["rho" does not exist]
! not defined "rho" ["qp1" does not exist]
!
!
! Reference: Jackett, D. R. and T. J. McDougall, 1995, Minimal
! Adjustment of Hydrostatic Profiles to Achieve Static
! Stability. J. Atmos. Ocean. Tec., vol. 12, pp. 381-389.
!
! << This equation of state formulation has been derived by Jackett
! and McDougall (1992), unpublished manuscript, CSIRO, Australia. It
! computes in-situ density anomaly as a function of potential
! temperature (Celsius) relative to the surface, salinity (PSU),
! and depth (meters). It assumes no pressure variation along
! geopotential surfaces, that is, depth and pressure are
! interchangeable. >>
! John Wilkin, 29 July 92
!


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k

      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & buoy(0:Lm+1,0:Mm+1,N),
     & rho1, qp1,
     & dpth,cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00

      real*8, parameter :: r00=999.842594, r01=6.793952E-2,
     & r02=-9.095290E-3, r03=1.001685E-4, r04=-1.120083E-6,
     & r05=6.536332E-9,
     & r10=0.824493, r11=-4.08990E-3, r12=7.64380E-5,
     & r13=-8.24670E-7, r14=5.38750E-9,
     & rS0=-5.72466E-3, rS1=1.02270E-4, rS2=-1.65460E-6,
     & r20=4.8314E-4,
     & K00=19092.56, K01=209.8925, K02=-3.041638,
     & K03=-1.852732e-3, K04=-1.361629e-5,
     & K10=104.4077, K11=-6.500517, K12=0.1553190,
     & K13=2.326469e-4,
     & KS0=-5.587545, KS1=+0.7390729, KS2=-1.909078e-2,
     & qp2=0.0000172, g=9.81


      real rho1_0, K0_Duk


      integer numthreads, trd, chunk_size, margin, jstr,jend


Cf2py intent(in) Lm,Mm,N, T,S, z_r,z_w, rho0
Cf2py intent(out) buoy



!!!!!!!!!!!!!!!!!!




      Tt=3.8D0
      Ts=34.5D0
      sqrtTs=sqrt(Ts)

      dr00=r00-1000.D0

      rho1_0=dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*( r04+Tt*r05 ))))
     & +Ts*( R10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*( rS0+Tt*( rS1+Tt*rS2 ))+Ts*r20 )

      K0_Duk= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))


      dr00=r00-rho0

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


      numthreads=1
C$ numthreads=omp_get_num_threads()
      trd=0
C$ trd=omp_get_thread_num()
      chunk_size=(jmax-jmin + numthreads)/numthreads
      margin=(chunk_size*numthreads -jmax+jmin-1)/2
      jstr=max( trd *chunk_size -margin, jmin )
      jend=min( (trd+1)*chunk_size-1-margin, jmax )


      cff = g/rho0

      do j=jstr,jend
        do k=1,N
          do i=imin,imax
            Tt=T(i,j,k)

            Ts=max(S(i,j,k), 0.)
            sqrtTs=sqrt(Ts)

            rho1=( dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*(
     & r04+Tt*r05 ))))
     & +Ts*( r10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*(rS0+Tt*(
     & rS1+Tt*rS2 ))+Ts*r20 ))




            buoy(i,j,k) = -cff * rho1


          enddo
        enddo
      enddo ! <-- j

      return
      end
# 57 "R_tools_fort.F" 2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute neutral density gradients
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 1 "R_tools_fort_routines/rho_grad.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute neutral density gradients
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine rho_grad(Lm,Mm,N, T,S, z_r,z_w,rho0,pm,pn,
     & drdx,drdy,drdz)


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k
      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & rho1(0:Lm+1,0:Mm+1,N), drdz(0:Lm+1,0:Mm+1,0:N),
     & drdx(1:Lm+1,0:Mm+1,N), drdy(0:Lm+1,1:Mm+1,N),
     & qp1(0:Lm+1,0:Mm+1,N),
     & dpth,cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00

      real*8, parameter :: r00=999.842594, r01=6.793952E-2,
     & r02=-9.095290E-3, r03=1.001685E-4, r04=-1.120083E-6,
     & r05=6.536332E-9,
     & r10=0.824493, r11=-4.08990E-3, r12=7.64380E-5,
     & r13=-8.24670E-7, r14=5.38750E-9,
     & rS0=-5.72466E-3, rS1=1.02270E-4, rS2=-1.65460E-6,
     & r20=4.8314E-4,
     & K00=19092.56, K01=209.8925, K02=-3.041638,
     & K03=-1.852732e-3, K04=-1.361629e-5,
     & K10=104.4077, K11=-6.500517, K12=0.1553190,
     & K13=2.326469e-4,
     & KS0=-5.587545, KS1=+0.7390729, KS2=-1.909078e-2,
     & qp2=0.0000172, g=9.81


      real rho1_0, K0_Duk


      integer numthreads, trd, chunk_size, margin, jstr,jend


Cf2py intent(in) Lm,Mm,N, T,S, z_r,z_w, rho0,pm,pn
Cf2py intent(out) drdx,drdy,drdz



!!!!!!!!!!!!!!!!!!




      Tt=3.8D0
      Ts=34.5D0
      sqrtTs=sqrt(Ts)

      dr00=r00-1000.D0

      rho1_0=dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*( r04+Tt*r05 ))))
     & +Ts*( R10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*( rS0+Tt*( rS1+Tt*rS2 ))+Ts*r20 )

      K0_Duk= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))


      dr00=r00-rho0


      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

      do j=jmin,jmax
        do k=1,N
          do i=imin,imax
            Tt=T(i,j,k)

            Ts=max(S(i,j,k), 0.)
            sqrtTs=sqrt(Ts)

            rho1(i,j,k)=( dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*(
     & r04+Tt*r05 ))))
     & +Ts*( r10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*(rS0+Tt*(
     & rS1+Tt*rS2 ))+Ts*r20 ))

            K0= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))




            qp1(i,j,k)= 0.1D0*(rho0+rho1(i,j,k))*(K0_Duk-K0)
     & /((K00+K0)*(K00+K0_Duk))






          enddo
        enddo

        cff=g/rho0

        do k=1,N-1
          do i=imin,imax

            dpth=z_w(i,j,N)-0.5*(z_r(i,j,k+1)+z_r(i,j,k))
            cff2=( rho1(i,j,k+1)-rho1(i,j,k) ! Elementary
     & +(qp1(i,j,k+1)-qp1(i,j,k)) ! adiabatic
     & *dpth*(1.-2.*qp2*dpth) ! difference
     & )


            drdz(i,j,k)=-cff*cff2 / (z_r(i,j,k+1)-z_r(i,j,k))

          enddo
        enddo


      enddo ! <-- j


!---------------------------------------------------------------------------------------
      do k=N,1,-1

        do j=jmin,jmax
          do i=imin+1,imax


            dpth=0.5*( z_w(i,j,N)+z_w(i-1,j,N)
     & -z_r(i,j,k)-z_r(i-1,j,k))

            drdx(i,j,k)=-cff*( rho1(i,j,k)-rho1(i-1,j,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i-1,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference
     & *0.5*(pm(i,j)+pm(i-1,j))


          enddo
        enddo

!---------------------------------------------------------------------------------------
        do j=jmin+1,jmax
          do i=imin,imax


            dpth=0.5*( z_w(i,j,N)+z_w(i,j-1,N)
     & -z_r(i,j,k)-z_r(i,j-1,k))

            drdy(i,j,k)=-cff*( rho1(i,j,k)-rho1(i,j-1,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i,j-1,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) )
     & *0.5*(pn(i,j)+pn(i,j-1))




          enddo
        enddo

      enddo
!---------------------------------------------------------------------------------------

      return
      end
# 64 "R_tools_fort.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Z interpolation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 1 "R_tools_fort_routines/sigma_to_z_intr_sfc.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Z interpolation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine sigma_to_z_intr_sfc (Lm,Mm,N, nz, z_r, z_w, rmask, var,
     & z_lev, var_zlv, imin,jmin,kmin, FillValue)
!
! Interpolate field "var" defined in sigma-space to 3-D z_lev.
!


      implicit none

      integer Lm,Mm,N, nz, imin,imax,jmin,jmax, kmin, i,j,k,m

      integer km(0:Lm+1)

      real*8 var(imin:Lm+1,jmin:Mm+1,kmin:N),
     & z_r(0:Lm+1,0:Mm+1,N), rmask(0:Lm+1,0:Mm+1),
     & z_w(0:Lm+1,0:Mm+1,0:N), z_lev(imin:Lm+1,jmin:Mm+1,nz),
     & FillValue, var_zlv(imin:Lm+1,jmin:Mm+1,nz),
     & zz(0:Lm+1,0:N+1), dpth

     & , dz(0:Lm+1,kmin-1:N), FC(0:Lm+1,kmin-1:N), p,q,cff

      integer numthreads, trd, chunk_size, margin, jstr,jend
C$ integer omp_get_num_threads, omp_get_thread_num


      imax=Lm+1
      jmax=Mm+1

      numthreads=1
C$ numthreads=omp_get_num_threads()
      trd=0
C$ trd=omp_get_thread_num()
      chunk_size=(jmax-jmin + numthreads)/numthreads
      margin=(chunk_size*numthreads -jmax+jmin-1)/2
      jstr=jmin !max( trd *chunk_size -margin, jmin )
      jend=jmax !min( (trd+1)*chunk_size-1-margin, jmax )


Cf2py intent(in) Lm,Mm,N, nz, z_r, z_w, rmask, var, z_lev, imin,jmin,kmin, FillValue
Cf2py intent(out) var_zlv
# 53 "R_tools_fort_routines/sigma_to_z_intr_sfc.F"
      do j=jstr,jend
        if (kmin.eq.1) then
          if (imin.eq.0 .and. jmin.eq.0) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=z_r(i,j,k)
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=z_w(i,j,0)
              zz(i,N+1)=z_w(i,j,N)
            enddo
          elseif (imin.eq.1 .and. jmin.eq.0) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.5D0*(z_r(i,j,k)+z_r(i-1,j,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.5D0*(z_w(i-1,j,0)+z_w(i,j,0))
              zz(i,N+1)=0.5D0*(z_w(i-1,j,N)+z_w(i,j,N))
            enddo
          elseif (imin.eq.0 .and. jmin.eq.1) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.5*(z_r(i,j,k)+z_r(i,j-1,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.5D0*(z_w(i,j,0)+z_w(i,j-1,0))
              zz(i,N+1)=0.5D0*(z_w(i,j,N)+z_w(i,j-1,N))
            enddo
          elseif (imin.eq.1 .and. jmin.eq.1) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.25D0*( z_r(i,j,k)+z_r(i-1,j,k)
     & +z_r(i,j-1,k)+z_r(i-1,j-1,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.25D0*( z_w(i,j,0)+z_w(i-1,j,0)
     & +z_w(i,j-1,0)+z_w(i-1,j-1,0))

              zz(i,N+1)=0.25D0*( z_w(i,j,N)+z_w(i-1,j,N)
     & +z_w(i,j-1,N)+z_w(i-1,j-1,N))
             enddo
          endif
        else
          if (imin.eq.0 .and. jmin.eq.0) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=z_w(i,j,k)
              enddo
            enddo
          elseif (imin.eq.1 .and. jmin.eq.0) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.5D0*(z_w(i,j,k)+z_w(i-1,j,k))
              enddo
            enddo
          elseif (imin.eq.0 .and. jmin.eq.1) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.5*(z_w(i,j,k)+z_w(i,j-1,k))
              enddo
            enddo
          elseif (imin.eq.1 .and. jmin.eq.1) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.25D0*( z_w(i,j,k)+z_w(i-1,j,k)
     & +z_w(i,j-1,k)+z_w(i-1,j-1,k))
              enddo
            enddo
          endif
        endif

        do k=kmin,N-1
          do i=imin,imax
            dz(i,k)=zz(i,k+1)-zz(i,k)
            FC(i,k)=var(i,j,k+1)-var(i,j,k)
          enddo
        enddo
        do i=imin,imax
          dz(i,kmin-1)=dz(i,kmin)
          FC(i,kmin-1)=FC(i,kmin)

          dz(i,N)=dz(i,N-1)
          FC(i,N)=FC(i,N-1)
        enddo
        do k=N,kmin,-1 !--> irreversible
          do i=imin,imax
            cff=FC(i,k)*FC(i,k-1)
            if (cff.gt.0.D0) then
              FC(i,k)=cff*(dz(i,k)+dz(i,k-1))/( (FC(i,k)+FC(i,k-1))
     & *dz(i,k)*dz(i,k-1) )
            else
              FC(i,k)=0.D0
            endif
          enddo
        enddo

        do m=1,nz


          if (kmin.eq.0) then !
            do i=imin,imax !
              dpth=zz(i,N)-zz(i,0)
              if (rmask(i,j).lt.0.5) then
                km(i)=-3 !--> masked out
              elseif (dpth*(z_lev(i,j,m)-zz(i,N)).gt.0.) then
                km(i)=N+2 !<-- above surface
              elseif (dpth*(zz(i,0)-z_lev(i,j,m)).gt.0.) then
                km(i)=-2 !<-- below bottom
              else
                km(i)=-1 !--> to search
              endif
            enddo
          else
            do i=imin,imax
              dpth=zz(i,N+1)-zz(i,0)
              if (rmask(i,j).lt.0.5) then
                km(i)=-3 !--> masked out
              elseif (dpth*(z_lev(i,j,m)-zz(i,N+1)).gt.0.) then
                km(i)=N+2 !<-- above surface

              elseif (dpth*(z_lev(i,j,m)-zz(i,N)).gt.0.) then
                km(i)=N !<-- below surface, but above z_r(N)
              elseif (dpth*(zz(i,0)-z_lev(i,j,m)).gt.0.) then
                km(i)=-2 !<-- below bottom
              elseif (dpth*(zz(i,1)-z_lev(i,j,m)).gt.0.) then
                km(i)=0 !<-- above bottom, but below z_r(1)
              else
                km(i)=-1 !--> to search
              endif
            enddo
          endif
          do k=N-1,kmin,-1
            do i=imin,imax
              if (km(i).eq.-1) then
                if((zz(i,k+1)-z_lev(i,j,m))*(z_lev(i,j,m)-zz(i,k))
     & .ge. 0.) km(i)=k
              endif
            enddo
          enddo

          do i=imin,imax
            if (km(i).eq.-3) then
              var_zlv(i,j,m)=0. !<-- masked out
            elseif (km(i).eq.-2) then

              var_zlv(i,j,m)=FillValue !<-- below bottom

            elseif (km(i).eq.N+2) then

              var_zlv(i,j,m)=var(i,j,N) !-> R-point, above z_r(N)

     & +FC(i,N)*(z_lev(i,j,m)-zz(i,N))







            elseif (km(i).eq.N) then
              var_zlv(i,j,m)=var(i,j,N) !-> R-point, above z_r(N)

     & +FC(i,N)*(z_lev(i,j,m)-zz(i,N))




            elseif (km(i).eq.kmin-1) then !-> R-point below z_r(1),
              var_zlv(i,j,m)=var(i,j,kmin) ! but above bottom

     & -FC(i,kmin)*(zz(i,kmin)-z_lev(i,j,m))




            else
              k=km(i)
              !write(*,*) k,km

              cff=1.D0/(zz(i,k+1)-zz(i,k))
              p=z_lev(i,j,m)-zz(i,k)
              q=zz(i,k+1)-z_lev(i,j,m)

              var_zlv(i,j,m)=cff*( q*var(i,j,k) + p*var(i,j,k+1)
     & -cff*p*q*( cff*(q-p)*(var(i,j,k+1)-var(i,j,k))
     & +p*FC(i,k+1) -q*FC(i,k) )
     & )







            !write(*,*) 'bof',i,j,k,zz(i,k), zz(i,k+1), z_lev(i,j,m), m
# 266 "R_tools_fort_routines/sigma_to_z_intr_sfc.F"
            endif
          enddo
        enddo ! <-- m
      enddo !<-- j

      return
      end
# 70 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/sigma_to_z_intr_bot.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Z interpolation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine sigma_to_z_intr_bot (Lm,Mm,N,nz,z_r,z_w,rmask,var,
     & z_lev, var_zlv, below, imin,jmin,kmin, FillValue)
!
! Interpolate field "var" defined in sigma-space to 3-D z_lev.
!


      implicit none

      integer Lm,Mm,N, nz, imin,imax,jmin,jmax, kmin, i,j,k,m

      integer km(0:Lm+1)

      real*8 var(imin:Lm+1,jmin:Mm+1,kmin:N),
     & z_r(0:Lm+1,0:Mm+1,N), rmask(0:Lm+1,0:Mm+1),
     & z_w(0:Lm+1,0:Mm+1,0:N), z_lev(imin:Lm+1,jmin:Mm+1,nz),
     & FillValue, var_zlv(imin:Lm+1,jmin:Mm+1,nz),
     & zz(0:Lm+1,0:N+1), dpth, below

     & , dz(0:Lm+1,kmin-1:N), FC(0:Lm+1,kmin-1:N), p,q,cff

      integer numthreads, trd, chunk_size, margin, jstr,jend
C$ integer omp_get_num_threads, omp_get_thread_num


      imax=Lm+1
      jmax=Mm+1

      numthreads=1
C$ numthreads=omp_get_num_threads()
      trd=0
C$ trd=omp_get_thread_num()
      chunk_size=(jmax-jmin + numthreads)/numthreads
      margin=(chunk_size*numthreads -jmax+jmin-1)/2
      jstr=jmin !max( trd *chunk_size -margin, jmin )
      jend=jmax !min( (trd+1)*chunk_size-1-margin, jmax )


Cf2py intent(in) Lm,Mm,N, nz, z_r, z_w, rmask, var, z_lev, below, imin,jmin,kmin, FillValue
Cf2py intent(out) var_zlv
# 53 "R_tools_fort_routines/sigma_to_z_intr_bot.F"
      do j=jstr,jend
        if (kmin.eq.1) then
          if (imin.eq.0 .and. jmin.eq.0) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=z_r(i,j,k)
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=z_w(i,j,0)
              zz(i,N+1)=z_w(i,j,N)
            enddo
          elseif (imin.eq.1 .and. jmin.eq.0) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.5D0*(z_r(i,j,k)+z_r(i-1,j,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.5D0*(z_w(i-1,j,0)+z_w(i,j,0))
              zz(i,N+1)=0.5D0*(z_w(i-1,j,N)+z_w(i,j,N))
            enddo
          elseif (imin.eq.0 .and. jmin.eq.1) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.5*(z_r(i,j,k)+z_r(i,j-1,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.5D0*(z_w(i,j,0)+z_w(i,j-1,0))
              zz(i,N+1)=0.5D0*(z_w(i,j,N)+z_w(i,j-1,N))
            enddo
          elseif (imin.eq.1 .and. jmin.eq.1) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.25D0*( z_r(i,j,k)+z_r(i-1,j,k)
     & +z_r(i,j-1,k)+z_r(i-1,j-1,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.25D0*( z_w(i,j,0)+z_w(i-1,j,0)
     & +z_w(i,j-1,0)+z_w(i-1,j-1,0))

              zz(i,N+1)=0.25D0*( z_w(i,j,N)+z_w(i-1,j,N)
     & +z_w(i,j-1,N)+z_w(i-1,j-1,N))
             enddo
          endif
        else
          if (imin.eq.0 .and. jmin.eq.0) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=z_w(i,j,k)
              enddo
            enddo
          elseif (imin.eq.1 .and. jmin.eq.0) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.5D0*(z_w(i,j,k)+z_w(i-1,j,k))
              enddo
            enddo
          elseif (imin.eq.0 .and. jmin.eq.1) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.5*(z_w(i,j,k)+z_w(i,j-1,k))
              enddo
            enddo
          elseif (imin.eq.1 .and. jmin.eq.1) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.25D0*( z_w(i,j,k)+z_w(i-1,j,k)
     & +z_w(i,j-1,k)+z_w(i-1,j-1,k))
              enddo
            enddo
          endif
        endif

        do k=kmin,N-1
          do i=imin,imax
            dz(i,k)=zz(i,k+1)-zz(i,k)
            FC(i,k)=var(i,j,k+1)-var(i,j,k)
          enddo
        enddo
        do i=imin,imax
          dz(i,kmin-1)=dz(i,kmin)
          FC(i,kmin-1)=FC(i,kmin)

          dz(i,N)=dz(i,N-1)
          FC(i,N)=FC(i,N-1)
        enddo
        do k=N,kmin,-1 !--> irreversible
          do i=imin,imax
            cff=FC(i,k)*FC(i,k-1)
            if (cff.gt.0.D0) then
              FC(i,k)=cff*(dz(i,k)+dz(i,k-1))/( (FC(i,k)+FC(i,k-1))
     & *dz(i,k)*dz(i,k-1) )
            else
              FC(i,k)=0.D0
            endif
          enddo
        enddo

        do m=1,nz


          if (kmin.eq.0) then !
            do i=imin,imax !
              dpth=zz(i,N)-zz(i,0)
              if (rmask(i,j).lt.0.5) then
                km(i)=-3 !--> masked out
              elseif (dpth*(z_lev(i,j,m)-zz(i,N)).gt.0.) then
                km(i)=N+2 !<-- above surface
              elseif (dpth*(zz(i,0)-z_lev(i,j,m)).gt.0.) then
                km(i)=-2 !<-- below bottom
              else
                km(i)=-1 !--> to search
              endif
            enddo
          else
            do i=imin,imax
              dpth=zz(i,N+1)-zz(i,0)
              if (rmask(i,j).lt.0.5) then
                km(i)=-3 !--> masked out
              elseif (dpth*(z_lev(i,j,m)-zz(i,N+1)).gt.0.) then
                km(i)=N+2 !<-- above surface

              elseif (dpth*(z_lev(i,j,m)-zz(i,N)).gt.0.) then
                km(i)=N !<-- below surface, but above z_r(N)
              elseif (dpth*(zz(i,0)-below-z_lev(i,j,m)).gt.0.) then
                km(i)=-3 !<-- below bottom
              elseif (dpth*(zz(i,0)-z_lev(i,j,m)).gt.0.) then
                km(i)=-2 !<-- below bottom but close
              elseif (dpth*(zz(i,1)-z_lev(i,j,m)).gt.0.) then
                km(i)=0 !<-- above bottom, but below z_r(1)
              else
                km(i)=-1 !--> to search
              endif
            enddo
          endif
          do k=N-1,kmin,-1
            do i=imin,imax
              if (km(i).eq.-1) then
                if((zz(i,k+1)-z_lev(i,j,m))*(z_lev(i,j,m)-zz(i,k))
     & .ge. 0.) km(i)=k
              endif
            enddo
          enddo

          do i=imin,imax
            if (km(i).eq.-3) then
              var_zlv(i,j,m)=FillValue !<-- masked out
            elseif (km(i).eq.-2) then

              var_zlv(i,j,m)=var(i,j,kmin) !

     & -FC(i,kmin)*(zz(i,kmin)-z_lev(i,j,m))







            elseif (km(i).eq.N+2) then

              var_zlv(i,j,m)=var(i,j,N) !-> R-point, above z_r(N)

     & +FC(i,N)*(z_lev(i,j,m)-zz(i,N))







            elseif (km(i).eq.N) then
              var_zlv(i,j,m)=var(i,j,N) !-> R-point, above z_r(N)

     & +FC(i,N)*(z_lev(i,j,m)-zz(i,N))




            elseif (km(i).eq.kmin-1) then !-> R-point below z_r(1),
              var_zlv(i,j,m)=var(i,j,kmin) ! but above bottom

     & -FC(i,kmin)*(zz(i,kmin)-z_lev(i,j,m))




            else
              k=km(i)
              !write(*,*) k,km

              cff=1.D0/(zz(i,k+1)-zz(i,k))
              p=z_lev(i,j,m)-zz(i,k)
              q=zz(i,k+1)-z_lev(i,j,m)

              var_zlv(i,j,m)=cff*( q*var(i,j,k) + p*var(i,j,k+1)
     & -cff*p*q*( cff*(q-p)*(var(i,j,k+1)-var(i,j,k))
     & +p*FC(i,k+1) -q*FC(i,k) )
     & )







            !write(*,*) 'bof',i,j,k,zz(i,k), zz(i,k+1), z_lev(i,j,m), m
# 276 "R_tools_fort_routines/sigma_to_z_intr_bot.F"
            endif
          enddo
        enddo ! <-- m
      enddo !<-- j

      return
      end
# 72 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/sigma_to_z_intr_bot_2d.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Z interpolation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine sigma_to_z_intr_bot_2d (Lm,N, nz, z_r, z_w, rmask, var,
     & z_lev, var_zlv, below,imin,kmin, FillValue)
!
! Interpolate field "var" defined in sigma-space to 3-D z_lev.
!


      implicit none

      integer Lm,Mm,N, nz, imin,imax, kmin, i,k,m

      integer km(0:Lm+1)

      real*8 var(imin:Lm+1,kmin:N),
     & z_r(0:Lm+1,N), rmask(0:Lm+1),
     & z_w(0:Lm+1,0:N), z_lev(imin:Lm+1,nz),
     & FillValue, var_zlv(imin:Lm+1,nz),
     & zz(0:Lm+1,0:N+1), dpth, below

     & , dz(0:Lm+1,kmin-1:N), FC(0:Lm+1,kmin-1:N), p,q,cff

      integer numthreads, trd, chunk_size, margin, jstr,jend
C$ integer omp_get_num_threads, omp_get_thread_num


      imax=Lm+1


Cf2py intent(in) Lm,Mm,N, nz, z_r, z_w, rmask, var, z_lev, below, imin,kmin, FillValue
Cf2py intent(out) var_zlv



        if (kmin.eq.1) then
          if (imin.eq.0) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=z_r(i,k)
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=z_w(i,0)
              zz(i,N+1)=z_w(i,N)
            enddo
          elseif (imin.eq.1) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.5D0*(z_r(i,k)+z_r(i-1,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.5D0*(z_w(i-1,0)+z_w(i,0))
              zz(i,N+1)=0.5D0*(z_w(i-1,N)+z_w(i,N))
            enddo
          endif
        else
          if (imin.eq.0 ) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=z_w(i,k)
              enddo
            enddo
          elseif (imin.eq.1) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.5D0*(z_w(i,k)+z_w(i-1,k))
              enddo
            enddo
          endif
        endif

        do k=kmin,N-1
          do i=imin,imax
            dz(i,k)=zz(i,k+1)-zz(i,k)
            FC(i,k)=var(i,k+1)-var(i,k)
          enddo
        enddo
        do i=imin,imax
          dz(i,kmin-1)=dz(i,kmin)
          FC(i,kmin-1)=FC(i,kmin)

          dz(i,N)=dz(i,N-1)
          FC(i,N)=FC(i,N-1)
        enddo
        do k=N,kmin,-1 !--> irreversible
          do i=imin,imax
            cff=FC(i,k)*FC(i,k-1)
            if (cff.gt.0.D0) then
              FC(i,k)=cff*(dz(i,k)+dz(i,k-1))/( (FC(i,k)+FC(i,k-1))
     & *dz(i,k)*dz(i,k-1) )
            else
              FC(i,k)=0.D0
            endif
          enddo
        enddo

        do m=1,nz


          if (kmin.eq.0) then !
            do i=imin,imax !
              dpth=zz(i,N)-zz(i,0)
              if (rmask(i).lt.0.5) then
                km(i)=-3 !--> masked out
              elseif (dpth*(z_lev(i,m)-zz(i,N)).gt.0.) then
                km(i)=N+2 !<-- above surface
              elseif (dpth*(zz(i,0)-z_lev(i,m)).gt.0.) then
                km(i)=-2 !<-- below bottom
              else
                km(i)=-1 !--> to search
              endif
            enddo
          else
            do i=imin,imax
              dpth=zz(i,N+1)-zz(i,0)
              if (rmask(i).lt.0.5) then
                km(i)=-3 !--> masked out
              elseif (dpth*(z_lev(i,m)-zz(i,N+1)).gt.0.) then
                km(i)=N+2 !<-- above surface

              elseif (dpth*(z_lev(i,m)-zz(i,N)).gt.0.) then
                km(i)=N !<-- below surface, but above z_r(N)
              elseif (dpth*(zz(i,0)-below-z_lev(i,m)).gt.0.) then
                km(i)=-3 !<-- below bottom
              elseif (dpth*(zz(i,0)-z_lev(i,m)).gt.0.) then
                km(i)=-2 !<-- below bottom but close
              elseif (dpth*(zz(i,1)-z_lev(i,m)).gt.0.) then
                km(i)=0 !<-- above bottom, but below z_r(1)
              else
                km(i)=-1 !--> to search
              endif
            enddo
          endif
          do k=N-1,kmin,-1
            do i=imin,imax
              if (km(i).eq.-1) then
                if((zz(i,k+1)-z_lev(i,m))*(z_lev(i,m)-zz(i,k))
     & .ge. 0.) km(i)=k
              endif
            enddo
          enddo

          do i=imin,imax
            if (km(i).eq.-3) then
              var_zlv(i,m)=FillValue !<-- masked out
            elseif (km(i).eq.-2) then

              var_zlv(i,m)=var(i,kmin) !

     & -FC(i,kmin)*(zz(i,kmin)-z_lev(i,m))







            elseif (km(i).eq.N+2) then

              var_zlv(i,m)=var(i,N) !-> R-point, above z_r(N)

     & +FC(i,N)*(z_lev(i,m)-zz(i,N))







            elseif (km(i).eq.N) then
              var_zlv(i,m)=var(i,N) !-> R-point, above z_r(N)

     & +FC(i,N)*(z_lev(i,m)-zz(i,N))




            elseif (km(i).eq.kmin-1) then !-> R-point below z_r(1),
              var_zlv(i,m)=var(i,kmin) ! but above bottom

     & -FC(i,kmin)*(zz(i,kmin)-z_lev(i,m))




            else
              k=km(i)
              !write(*,*) k,km

              cff=1.D0/(zz(i,k+1)-zz(i,k))
              p=z_lev(i,m)-zz(i,k)
              q=zz(i,k+1)-z_lev(i,m)

              var_zlv(i,m)=cff*( q*var(i,k) + p*var(i,k+1)
     & -cff*p*q*( cff*(q-p)*(var(i,k+1)-var(i,k))
     & +p*FC(i,k+1) -q*FC(i,k) )
     & )







            !write(*,*) 'bof',i,k,zz(i,k), zz(i,k+1), z_lev(i,m), m



            endif
          enddo
        enddo ! <-- m


      return
      end
# 74 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/sigma_to_z_intr.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Z interpolation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine sigma_to_z_intr (Lm,Mm,N, nz, z_r, z_w, rmask, var,
     & z_lev, var_zlv, imin,jmin,kmin, FillValue)
!
! Interpolate field "var" defined in sigma-space to 3-D z_lev.
!


      implicit none

      integer Lm,Mm,N, nz, imin,imax,jmin,jmax, kmin, i,j,k,m

      integer km(0:Lm+1)

      real*8 var(imin:Lm+1,jmin:Mm+1,kmin:N),
     & z_r(0:Lm+1,0:Mm+1,N), rmask(0:Lm+1,0:Mm+1),
     & z_w(0:Lm+1,0:Mm+1,0:N), z_lev(imin:Lm+1,jmin:Mm+1,nz),
     & FillValue, var_zlv(imin:Lm+1,jmin:Mm+1,nz),
     & zz(0:Lm+1,0:N+1), dpth

     & , dz(0:Lm+1,kmin-1:N), FC(0:Lm+1,kmin-1:N), p,q,cff

      integer numthreads, trd, chunk_size, margin, jstr,jend
C$ integer omp_get_num_threads, omp_get_thread_num


      imax=Lm+1
      jmax=Mm+1

      numthreads=1
C$ numthreads=omp_get_num_threads()
      trd=0
C$ trd=omp_get_thread_num()
      chunk_size=(jmax-jmin + numthreads)/numthreads
      margin=(chunk_size*numthreads -jmax+jmin-1)/2
      jstr=jmin !max( trd *chunk_size -margin, jmin )
      jend=jmax !min( (trd+1)*chunk_size-1-margin, jmax )


Cf2py intent(in) Lm,Mm,N, nz, z_r, z_w, rmask, var, z_lev, imin,jmin,kmin, FillValue
Cf2py intent(out) var_zlv
# 54 "R_tools_fort_routines/sigma_to_z_intr.F"
      do j=jstr,jend
        if (kmin.eq.1) then
          if (imin.eq.0 .and. jmin.eq.0) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=z_r(i,j,k)
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=z_w(i,j,0)
              zz(i,N+1)=z_w(i,j,N)
            enddo
          elseif (imin.eq.1 .and. jmin.eq.0) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.5D0*(z_r(i,j,k)+z_r(i-1,j,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.5D0*(z_w(i-1,j,0)+z_w(i,j,0))
              zz(i,N+1)=0.5D0*(z_w(i-1,j,N)+z_w(i,j,N))
            enddo
          elseif (imin.eq.0 .and. jmin.eq.1) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.5*(z_r(i,j,k)+z_r(i,j-1,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.5D0*(z_w(i,j,0)+z_w(i,j-1,0))
              zz(i,N+1)=0.5D0*(z_w(i,j,N)+z_w(i,j-1,N))
            enddo
          elseif (imin.eq.1 .and. jmin.eq.1) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.25D0*( z_r(i,j,k)+z_r(i-1,j,k)
     & +z_r(i,j-1,k)+z_r(i-1,j-1,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.25D0*( z_w(i,j,0)+z_w(i-1,j,0)
     & +z_w(i,j-1,0)+z_w(i-1,j-1,0))

              zz(i,N+1)=0.25D0*( z_w(i,j,N)+z_w(i-1,j,N)
     & +z_w(i,j-1,N)+z_w(i-1,j-1,N))
             enddo
          endif
        else
          if (imin.eq.0 .and. jmin.eq.0) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=z_w(i,j,k)
              enddo
            enddo
          elseif (imin.eq.1 .and. jmin.eq.0) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.5D0*(z_w(i,j,k)+z_w(i-1,j,k))
              enddo
            enddo
          elseif (imin.eq.0 .and. jmin.eq.1) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.5*(z_w(i,j,k)+z_w(i,j-1,k))
              enddo
            enddo
          elseif (imin.eq.1 .and. jmin.eq.1) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.25D0*( z_w(i,j,k)+z_w(i-1,j,k)
     & +z_w(i,j-1,k)+z_w(i-1,j-1,k))
              enddo
            enddo
          endif
        endif

        do k=kmin,N-1
          do i=imin,imax
            dz(i,k)=zz(i,k+1)-zz(i,k)
            FC(i,k)=var(i,j,k+1)-var(i,j,k)
          enddo
        enddo
        do i=imin,imax
          dz(i,kmin-1)=dz(i,kmin)
          FC(i,kmin-1)=FC(i,kmin)

          dz(i,N)=dz(i,N-1)
          FC(i,N)=FC(i,N-1)
        enddo
        do k=N,kmin,-1 !--> irreversible
          do i=imin,imax
            cff=FC(i,k)*FC(i,k-1)
            if (cff.gt.0.D0) then
              FC(i,k)=cff*(dz(i,k)+dz(i,k-1))/( (FC(i,k)+FC(i,k-1))
     & *dz(i,k)*dz(i,k-1) )
            else
              FC(i,k)=0.D0
            endif
          enddo
        enddo

        do m=1,nz


          if (kmin.eq.0) then !
            do i=imin,imax !
              dpth=zz(i,N)-zz(i,0)
              if (rmask(i,j).lt.0.5) then
                km(i)=-3 !--> masked out
              elseif (dpth*(z_lev(i,j,m)-zz(i,N)).gt.0.) then
                km(i)=N+2 !<-- above surface
              elseif (dpth*(zz(i,0)-z_lev(i,j,m)).gt.0.) then
                km(i)=-2 !<-- below bottom
              else
                km(i)=-1 !--> to search
              endif
            enddo
          else
            do i=imin,imax
              dpth=zz(i,N+1)-zz(i,0)
              if (rmask(i,j).lt.0.5) then
                km(i)=-3 !--> masked out
              elseif (dpth*(z_lev(i,j,m)-zz(i,N+1)).gt.0.) then
                km(i)=N+2 !<-- above surface

              elseif (dpth*(z_lev(i,j,m)-zz(i,N)).gt.0.) then
                km(i)=N !<-- below surface, but above z_r(N)
              elseif (dpth*(zz(i,0)-z_lev(i,j,m)).gt.0.) then
                km(i)=-2 !<-- below bottom
              elseif (dpth*(zz(i,1)-z_lev(i,j,m)).gt.0.) then
                km(i)=0 !<-- above bottom, but below z_r(1)
              else
                km(i)=-1 !--> to search
              endif
            enddo
          endif
          do k=N-1,kmin,-1
            do i=imin,imax
              if (km(i).eq.-1) then
                if((zz(i,k+1)-z_lev(i,j,m))*(z_lev(i,j,m)-zz(i,k))
     & .ge. 0.) km(i)=k
              endif
            enddo
          enddo

          do i=imin,imax
            if (km(i).eq.-3) then
              var_zlv(i,j,m)=0. !<-- masked out
            elseif (km(i).eq.-2) then
              var_zlv(i,j,m)=FillValue !<-- below bottom
            elseif (km(i).eq.N+2) then
              var_zlv(i,j,m)=-FillValue !<-- above surface
            elseif (km(i).eq.N) then
              var_zlv(i,j,m)=var(i,j,N) !-> R-point, above z_r(N)

     & +FC(i,N)*(z_lev(i,j,m)-zz(i,N))




            elseif (km(i).eq.kmin-1) then !-> R-point below z_r(1),
              var_zlv(i,j,m)=var(i,j,kmin) ! but above bottom

     & -FC(i,kmin)*(zz(i,kmin)-z_lev(i,j,m))




            else
              k=km(i)
              !write(*,*) k,km

              cff=1.D0/(zz(i,k+1)-zz(i,k))
              p=z_lev(i,j,m)-zz(i,k)
              q=zz(i,k+1)-z_lev(i,j,m)

              var_zlv(i,j,m)=cff*( q*var(i,j,k) + p*var(i,j,k+1)
     & -cff*p*q*( cff*(q-p)*(var(i,j,k+1)-var(i,j,k))
     & +p*FC(i,k+1) -q*FC(i,k) )
     & )







            !write(*,*) 'bof',i,j,k,zz(i,k), zz(i,k+1), z_lev(i,j,m), m
# 255 "R_tools_fort_routines/sigma_to_z_intr.F"
            endif
          enddo
        enddo ! <-- m
      enddo !<-- j

      return
      end
# 76 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/interp_1d.F" 1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Z interpolation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine interp_1d(N, var,z_r,z_w,z_lev,var_zlv,kmin,verbose)

      implicit none

      integer k, N, km, verbose
      integer kmin

      real*8 var(N), z_r(N), z_w(0:N), z(0:N+1),
     & z_lev, var_zlv, dpth,
     & dz(kmin-1:N), FC(kmin-1:N),
     & p,q,cff

      real*8, parameter :: FillValue = -9999.

Cf2py intent(in) N,var,z_r,z_w,z_lev,kmin,verbose
Cf2py intent(out) var_zlv



        !if (verbose.eq.1) write(*,*) 'depth was ', z_lev
        !if (z_lev.gt.0) write(*,*) 'depth is ', z_lev !z_lev = -1*z_lev
        !if (verbose.eq.1) write(*,*) 'depth is ', z_lev


        if ((z_r(N).eq.0).and.(z_w(N).eq.0)) then
            !write(*,*) 'this point should be masked'
            var_zlv = FillValue

        else



! if (verbose.eq.1) then
! if (z_lev.gt.z_w(N)) then
! write(*,*) 'depths are ', z_lev, z_w(N)
! write(*,*) z_w
! write(*,*) z_r
! z_lev=z_w(N)
! endif
! endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            do k=1,N
                z(k)=z_r(k)
            enddo
              z(0)=z_w(0)
              z(N+1)=z_w(N)

        if (verbose.eq.1) then
           write(*,*) ' '
           write(*,*) 'z',z
           write(*,*) ' '
           write(*,*) 'z_r',z_r
           write(*,*) ' '
           write(*,*) 'z_w',z_w
           write(*,*) ' '

        endif

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




        do k=0,N-1
            dz(k)=z(k+1)-z(k)
            FC(k)=var(k+1)-var(k)
        enddo

          dz(kmin-1)=dz(kmin)
          FC(kmin-1)=FC(kmin)

          dz(N)=dz(N-1)
          FC(N)=FC(N-1)


        do k=N,kmin,-1 !--> irreversible
            cff=FC(k)*FC(k-1)
            if (cff.gt.0.D0) then
              FC(k)=cff*(dz(k)+dz(k-1))/( (FC(k)+FC(k-1))
     & *dz(k)*dz(k-1) )
            else
              FC(k)=0.D0
            endif
        enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if (kmin.eq.0) then

            dpth=z(N)-z(0)

            if (dpth*(z_lev-z(N)).gt.0.) then
               km=N+2 !<-- above surface
            elseif (dpth*(z(0)-z_lev).gt.0.) then
               km=-2 !<-- below bottom
            else
               km=-1 !--> to search
            endif

        else

            dpth=z(N+1)-z(0)

        if (verbose.eq.1) then
           write(*,*) 'dpth',dpth
        endif


            if (dpth*(z_lev-z(N+1)).gt.0.) then
                km=N+2 !<-- above surface
            elseif (dpth*(z_lev-z(N)).gt.0.) then
                km=N !<-- below surface, but above z_r(N)
            elseif (dpth*(z(0)-z_lev).gt.0.) then
                km=-2 !<-- below bottom
            elseif (dpth*(z(1)-z_lev).gt.0.) then
                km=0 !<-- above bottom, but below z_r(1)
            else
                km=-1 !--> to search
            endif


        if (verbose.eq.1) then
           write(*,*) 'dpth*(z_lev-z(N+1)',dpth*(z_lev-z(N+1))
           write(*,*) 'dpth*(z_lev-z(N))',dpth*(z_lev-z(N))
           write(*,*) 'dpth*(z(0)-z_lev)',dpth*(z(0)-z_lev)
           write(*,*) 'dpth*(z(1)-z_lev)',dpth*(z(1)-z_lev)
           write(*,*) 'km',km
        endif

        endif

        do k=N-1,kmin,-1

! write(*,*) 'test', k, z(k+1), z(k), z_lev
              if (km.eq.-1) then
                if ((z(k+1)-z_lev)*(z_lev-z(k)) .ge. 0.) then
                    km=k
                endif
              endif
        enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


          if (km.eq.-3) then
              var_zlv=0. !<-- masked out
          elseif (km.eq.-2) then

              var_zlv=var(kmin) ! but above bottom

     & -FC(kmin)*(z(kmin)-z_lev)







            elseif (km.eq.N+2) then

              var_zlv=var(N) ! but above bottom

     & +FC(N)*(z_lev-z(N))
# 181 "R_tools_fort_routines/interp_1d.F"
            elseif (km.eq.N) then
              var_zlv=var(N) !-> R-point, above z_r(N)

     & +FC(N)*(z_lev-z(N))




            elseif (km.eq.kmin-1) then !-> R-point below z_r(1),
              var_zlv=var(kmin) ! but above bottom

     & -FC(kmin)*(z(kmin)-z_lev)




            else
              k=km



              cff=1.D0/(z(k+1)-z(k))
              p=z_lev-z(k)
              q=z(k+1)-z_lev

              !write(*,*) 'koko',cff,p,q,var(k),var(k+1)

              var_zlv=cff*( q*var(k) + p*var(k+1)
     & -cff*p*q*( cff*(q-p)*(var(k+1)-var(k))
     & +p*FC(k+1) -q*FC(k) )
     & )







        endif

        if (verbose.eq.1) then
           write(*,*) 'var_zlv', var_zlv
        endif


        endif


      return
      end
# 78 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/sigma_to_z_intr_bounded.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Z interpolation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine sigma_to_z_intr_bounded (Lm,Mm,N, nz, z_r, z_w,
     & rmask, var,z_lev, var_zlv,
     & imin,jmin,kmin, FillValue)
!
! Interpolate field "var" defined in sigma-space to 3-D z_lev.
!


      implicit none

      integer Lm,Mm,N, nz, imin,imax,jmin,jmax, kmin, i,j,k,m

      integer km(0:Lm+1)

      real*8 var(imin:Lm+1,jmin:Mm+1,kmin:N),
     & z_r(0:Lm+1,0:Mm+1,N), rmask(0:Lm+1,0:Mm+1),
     & z_w(0:Lm+1,0:Mm+1,0:N), z_lev(imin:Lm+1,jmin:Mm+1,nz),
     & FillValue, var_zlv(imin:Lm+1,jmin:Mm+1,nz),
     & zz(0:Lm+1,0:N+1), dpth



      integer numthreads, trd, chunk_size, margin, jstr,jend
C$ integer omp_get_num_threads, omp_get_thread_num


      imax=Lm+1
      jmax=Mm+1

      numthreads=1
C$ numthreads=omp_get_num_threads()
      trd=0
C$ trd=omp_get_thread_num()
      chunk_size=(jmax-jmin + numthreads)/numthreads
      margin=(chunk_size*numthreads -jmax+jmin-1)/2
      jstr=jmin !max( trd *chunk_size -margin, jmin )
      jend=jmax !min( (trd+1)*chunk_size-1-margin, jmax )


Cf2py intent(in) Lm,Mm,N, nz, z_r, z_w, rmask, var, z_lev, imin,jmin,kmin, FillValue
Cf2py intent(out) var_zlv
# 54 "R_tools_fort_routines/sigma_to_z_intr_bounded.F"
      do j=jstr,jend
        if (kmin.eq.1) then
          if (imin.eq.0 .and. jmin.eq.0) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=z_r(i,j,k)
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=z_w(i,j,0)
              zz(i,N+1)=z_w(i,j,N)
            enddo
          elseif (imin.eq.1 .and. jmin.eq.0) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.5D0*(z_r(i,j,k)+z_r(i-1,j,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.5D0*(z_w(i-1,j,0)+z_w(i,j,0))
              zz(i,N+1)=0.5D0*(z_w(i-1,j,N)+z_w(i,j,N))
            enddo
          elseif (imin.eq.0 .and. jmin.eq.1) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.5*(z_r(i,j,k)+z_r(i,j-1,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.5D0*(z_w(i,j,0)+z_w(i,j-1,0))
              zz(i,N+1)=0.5D0*(z_w(i,j,N)+z_w(i,j-1,N))
            enddo
          elseif (imin.eq.1 .and. jmin.eq.1) then
            do k=1,N
              do i=imin,imax
                zz(i,k)=0.25D0*( z_r(i,j,k)+z_r(i-1,j,k)
     & +z_r(i,j-1,k)+z_r(i-1,j-1,k))
              enddo
            enddo
            do i=imin,imax
              zz(i,0)=0.25D0*( z_w(i,j,0)+z_w(i-1,j,0)
     & +z_w(i,j-1,0)+z_w(i-1,j-1,0))

              zz(i,N+1)=0.25D0*( z_w(i,j,N)+z_w(i-1,j,N)
     & +z_w(i,j-1,N)+z_w(i-1,j-1,N))
             enddo
          endif
        else
          if (imin.eq.0 .and. jmin.eq.0) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=z_w(i,j,k)
              enddo
            enddo
          elseif (imin.eq.1 .and. jmin.eq.0) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.5D0*(z_w(i,j,k)+z_w(i-1,j,k))
              enddo
            enddo
          elseif (imin.eq.0 .and. jmin.eq.1) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.5*(z_w(i,j,k)+z_w(i,j-1,k))
              enddo
            enddo
          elseif (imin.eq.1 .and. jmin.eq.1) then
            do k=0,N
              do i=imin,imax
                zz(i,k)=0.25D0*( z_w(i,j,k)+z_w(i-1,j,k)
     & +z_w(i,j-1,k)+z_w(i-1,j-1,k))
              enddo
            enddo
          endif
        endif
# 155 "R_tools_fort_routines/sigma_to_z_intr_bounded.F"
        do m=1,nz


          if (kmin.eq.0) then !
            do i=imin,imax !
              dpth=zz(i,N)-zz(i,0)
              if (rmask(i,j).lt.0.5) then
                km(i)=-3 !--> masked out
              elseif (dpth*(z_lev(i,j,m)-zz(i,N)).gt.0.) then
                km(i)=N+2 !<-- above surface
              elseif (dpth*(zz(i,0)-z_lev(i,j,m)).gt.0.) then
                km(i)=-2 !<-- below bottom
              else
                km(i)=-1 !--> to search
              endif
            enddo
          else
            do i=imin,imax
              dpth=zz(i,N+1)-zz(i,0)
              if (rmask(i,j).lt.0.5) then
                km(i)=-3 !--> masked out
              elseif (dpth*(z_lev(i,j,m)-zz(i,N+1)).gt.0.) then
                km(i)=N+2 !<-- above surface

              elseif (dpth*(z_lev(i,j,m)-zz(i,N)).gt.0.) then
                km(i)=N !<-- below surface, but above z_r(N)
              elseif (dpth*(zz(i,0)-z_lev(i,j,m)).gt.0.) then
                km(i)=-2 !<-- below bottom
              elseif (dpth*(zz(i,1)-z_lev(i,j,m)).gt.0.) then
                km(i)=0 !<-- above bottom, but below z_r(1)
              else
                km(i)=-1 !--> to search
              endif
            enddo
          endif
          do k=N-1,kmin,-1
            do i=imin,imax
              if (km(i).eq.-1) then
                if((zz(i,k+1)-z_lev(i,j,m))*(z_lev(i,j,m)-zz(i,k))
     & .ge. 0.) km(i)=k
              endif
            enddo
          enddo

          do i=imin,imax
            if (km(i).eq.-3) then
              var_zlv(i,j,m)=FillValue !<-- masked out
            elseif (km(i).eq.-2) then
              var_zlv(i,j,m)=var(i,j,1) !<-- below bottom
            elseif (km(i).eq.N+2) then
              var_zlv(i,j,m)=var(i,j,N) !<-- above surface
            elseif (km(i).eq.N) then
              var_zlv(i,j,m)=var(i,j,N) !-> R-point, above z_r(N)
            elseif (km(i).eq.kmin-1) then !-> R-point below z_r(1),
              var_zlv(i,j,m)=var(i,j,1) !<-- below bottom
            else
              k=km(i)
              !write(*,*) k,km
# 223 "R_tools_fort_routines/sigma_to_z_intr_bounded.F"
              var_zlv(i,j,m)=( var(i,j,k)*(zz(i,k+1)-z_lev(i,j,m))
     & +var(i,j,k+1)*(z_lev(i,j,m)-zz(i,k))
     & )/(zz(i,k+1)-zz(i,k))



            !write(*,*) 'bof',i,j,k,zz(i,k), zz(i,k+1), z_lev(i,j,m), m
# 243 "R_tools_fort_routines/sigma_to_z_intr_bounded.F"
            endif
          enddo
        enddo ! <-- m
      enddo !<-- j

      return
      end
# 80 "R_tools_fort.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute z_r and z_w for NEW_S_COORD
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 1 "R_tools_fort_routines/zlevs.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute z_r and z_w for NEW_S_COORD
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      subroutine zlevs(Lm,Mm,N, h,zeta, hc, Cs_r, Cs_w,z_r,z_w)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k

      real*8 Cs_w(0:N), Cs_r(N), cff_w, cff_r, cff1_w, cff1_r,
     & hc, ds,
     & zeta(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N),z_w(0:Lm+1,0:Mm+1,0:N),
     & h(0:Lm+1,0:Mm+1),hinv(0:Lm+1,0:Mm+1)


Cf2py intent(in) Lm,Mm,N, h,zeta, hc, Cs_w, Cs_r
Cf2py intent(out) z_r,z_w


      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


      ds=1.D0/dble(N)

      do j=jmin,jmax
        do i=imin,imax


          hinv(i,j)=1./(h(i,j)+hc)
          z_w(i,j,0)=-h(i,j)

        enddo

        do k=1,N,+1 !--> irreversible because of recursion in Hz


          cff_w=hc*ds* dble(k-N)
          cff_r=hc*ds*(dble(k-N)-0.5)

          cff1_w=Cs_w(k)
          cff1_r=Cs_r(k)


          do i=imin,imax

            z_w(i,j,k)=zeta(i,j) +(zeta(i,j)+h(i,j))
     & *(cff_w+cff1_w*h(i,j))*hinv(i,j)

            z_r(i,j,k)=zeta(i,j) +(zeta(i,j)+h(i,j))
     & *(cff_r+cff1_r*h(i,j))*hinv(i,j)


          enddo
        enddo
      enddo
      end
# 86 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/zlev.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute z_r and z_w for NEW_S_COORD
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      subroutine zlev(Lm,Mm,N, h,zeta, hc, Cs_r, Cs_w,z_r)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k

      real*8 Cs_w(0:N), Cs_r(N), cff_r,cff1_r,
     & hc, ds,
     & zeta(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N),
     & h(0:Lm+1,0:Mm+1),hinv(0:Lm+1,0:Mm+1)


Cf2py intent(in) Lm,Mm,N, h,zeta, hc, Cs_w, Cs_r
Cf2py intent(out) z_r


      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


      ds=1.D0/dble(N)

      do j=jmin,jmax
        do i=imin,imax


          hinv(i,j)=1./(h(i,j)+hc)

        enddo

        do k=1,N,+1 !--> irreversible because of recursion in Hz


          cff_r=hc*ds*(dble(k-N)-0.5)

          cff1_r=Cs_r(k)


          do i=imin,imax


            z_r(i,j,k)=zeta(i,j) +(zeta(i,j)+h(i,j))
     & *(cff_r+cff1_r*h(i,j))*hinv(i,j)


          enddo
        enddo
      enddo
      end
# 88 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/zlevs_agrif.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute z_r and z_w for NEW_S_COORD
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      subroutine zlevs_agrif(Lm,Mm,N, h,zeta, hc, Cs_r, Cs_w,
     & sc_r,sc_w,z_r,z_w)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k

      real*8 Cs_w(0:N), Cs_r(N), cff_w, cff_r, cff1_w, cff1_r,
     & sc_w(0:N), sc_r(N),
     & hc, ds, cff0_w, cff0_r,
     & zeta(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N),z_w(0:Lm+1,0:Mm+1,0:N),
     & h(0:Lm+1,0:Mm+1),hinv(0:Lm+1,0:Mm+1)


Cf2py intent(in) Lm,Mm,N, h,zeta, hc, Cs_w, Cs_r, sc_r,sc_w
Cf2py intent(out) z_r,z_w


      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


      ds=1.D0/dble(N)

      do j=jmin,jmax
        do i=imin,imax


          hinv(i,j)=1./(h(i,j)+hc)
          z_w(i,j,0)=-h(i,j)

        enddo

        do k=1,N,+1 !--> irreversible because of recursion in Hz


          cff_w =hc*sc_w(k)
          cff_r =hc*sc_r(k)

          cff1_w=Cs_w(k)
          cff1_r=Cs_r(k)

          do i=imin,imax

            cff0_w=cff_w+cff1_w*h(i,j)
            cff0_r=cff_r+cff1_r*h(i,j)


            z_w(i,j,k)=cff0_w*h(i,j)*hinv(i,j)+zeta(i,j)
     & *(1.+cff0_w*hinv(i,j))
            z_r(i,j,k)=cff0_r*h(i,j)*hinv(i,j)+zeta(i,j)
     & *(1.+cff0_r*hinv(i,j))



          enddo
        enddo
      enddo
      end
# 90 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/zlevs_kau.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute z_r and z_w for VERT_COORD_TYPE_KAU
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      subroutine zlevs_kau(Lm,Mm,N, h, hc, Cs_r, Cs_w,z_r,z_w)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k

      real*8 Cs_w(0:N), Cs_r(N), cff_w, cff_r, Cs_r_k, Cs_w_k,
     & hc, ds,
     & z_r(0:Lm+1,0:Mm+1,N),z_w(0:Lm+1,0:Mm+1,0:N),
     & h(0:Lm+1,0:Mm+1)

      real*8 hinv_r(0:Lm+1), hinv_w(0:Lm+1)
      real Gcoord_w,Gcoord_r,s_r, s_w
      real, parameter :: eps=1.E-20


Cf2py intent(in) Lm,Mm,N, h,hc, Cs_w, Cs_r
Cf2py intent(out) z_r,z_w


      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


      ds=1.D0/dble(N)

      do j=jmin,jmax

        Cs_r_k=Cs_r(N)
        s_r = -0.5D0 * ds
        Gcoord_r= 0.16+abs(s_r)**0.8*(1+s_r)**0.2 + (0.3 * exp(-1 /
     & max(abs(s_r), eps)) / max(abs(s_r), eps)**1.7)
! Gcoord_r= 1 !! SM09 CASE

        cff_r=hc*s_r*Gcoord_r

        do i=imin,imax

          hinv_r(i)=h(i,j)/(h(i,j)+hc*Gcoord_r)
          z_w(i,j,N)=0.
          z_r(i,j,N)=hinv_r(i)*( cff_r + Cs_r_k*h(i,j) )

          z_w(i,j,0)=-h(i,j)

        enddo

        do k=N-1,1,-1

          Cs_w_k = Cs_w(k)
          Cs_r_k = Cs_r(k)

          s_w = ds* dble(k-N)
          s_r = ds* (dble(k-N)-0.5D0)


! first define the Gcoord parameter
          Gcoord_w= 0.16+abs(s_w)**0.8*(1+s_w)**0.2 + (0.3 * exp(-1 /
     & max(abs(s_w), eps)) / max(abs(s_w), eps)**1.7)
!
          Gcoord_r= 0.16+abs(s_r)**0.8*(1+s_r)**0.2 + (0.3 * exp(-1 /
     & max(abs(s_r), eps)) / max(abs(s_r), eps)**1.7)

! Gcoord_w= 1 !!! SM09 CASE
! Gcoord_r= 1 !!! SM09 CASE

          cff_w=hc*s_w*Gcoord_w
          cff_r=hc*s_r*Gcoord_r

          do i=imin,imax

            hinv_w(i)=1./(h(i,j)+hc*Gcoord_w)
            z_w(i,j,k)= h(i,j) * (cff_w+Cs_w_k*h(i,j))*hinv_w(i)

            hinv_r(i)=1./(h(i,j)+hc*Gcoord_r)
            z_r(i,j,k)= h(i,j)*(cff_r+Cs_r_k*h(i,j))*hinv_r(i)


          enddo
        enddo
      enddo
      end
# 92 "R_tools_fort.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute Various
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# 1 "R_tools_fort_routines/get_rot.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute ROT
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_rot(Lm,Mm, u,v,pm,pn,rot)


      implicit none
      integer Lm,Mm, imin,imax,jmin,jmax, i,j,k
      real*8 rot(1:Lm+1,1:Mm+1),
     & u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dvdx, dudy


Cf2py intent(in) Lm,Mm,u,v,pm,pn
Cf2py intent(out) rot

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


        do j=jmin+1,jmax
          do i=imin+1,imax

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            dvdx = (v(i,j) - v(i-1,j)) * 0.25 *
     & (pm(i,j)+pm(i-1,j)+pm(i,j-1)+pm(i-1,j-1))



            dudy = (u(i,j) - u(i,j-1) )* 0.25 *
     & (pn(i,j)+pn(i-1,j)+pn(i,j-1)+pn(i-1,j-1))



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


            rot(i,j) = dvdx - dudy

            !write(*,*) i,j,dvdx, dudy, rot(i,j)


        enddo !<- i
      enddo !<- j



      return
      end
# 99 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/get_grad.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute GRAD amplitude for a PSI function
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_grad(Lm,Mm, psi,pm,pn,grad)


      implicit none
      integer Lm,Mm, imin,imax,jmin,jmax, i,j,k
      real*8 grad(1:Lm+1,1:Mm+1),
     & psi(0:Lm+1,0:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dvdx, dudy


Cf2py intent(in) Lm,Mm,psi,pm,pn
Cf2py intent(out) grad

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


        do j=jmin+1,jmax
          do i=imin+1,imax

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            dvdx = 0.5*((psi(i,j) - psi(i-1,j)) * 0.5 *
     & (pm(i,j)+pm(i-1,j))
     & + (psi(i,j-1)-psi(i-1,j-1)) * 0.5 *
     & (pm(i,j-1)+pm(i-1,j-1)))

            dudy = 0.5*((psi(i,j) - psi(i,j-1) )* 0.5 *
     & (pn(i,j)+pn(i,j-1))
     & + (psi(i-1,j) - psi(i-1,j-1) )* 0.5 *
     & (pn(i-1,j)+pn(i-1,j-1)))


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


            grad(i,j) = sqrt(dvdx**2 + dudy**2)

            !write(*,*) i,j,dvdx, dudy, rot(i,j)


        enddo !<- i
      enddo !<- j



      return
      end
# 101 "R_tools_fort.F" 2



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute vertical velocity
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# 1 "R_tools_fort_routines/get_wvlcty.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute vertical velocity
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






      subroutine get_wvlcty(Lm,Mm,N,u,v, z_r,z_w,pm,pn
     & ,Wvlc)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4


      real*8 Wrk(0:Lm+1,0:N),Wvlc(0:Lm+1,0:Mm+1,N),
     & Wxi(1:Lm+1,0:Mm+1),Weta(0:Lm+1,1:Mm+1)

# 1 "R_tools_fort_routines/scalars.h" 1
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

      real ,parameter :: sec2day=1./86400
# 36 "R_tools_fort_routines/get_wvlcty.F" 2




Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn
Cf2py intent(out) Wvlc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!
! Compute "omega" vertical velocity by means of integration of mass
! divergence of mass fluxes from bottom up. In this computation,
! unlike that in omega.F, there is (1) immediate multiplication by
! pm*pn so that the result has meaning of velocity, rather than
! finite volume mass flux through vertical facet of tracer grid box;
! and (2, also unlike omega.F) no subtraction of vertical velocity
! of moving grid-box interface (the effect of "breething" of vertical
! grid system due to evolving free surface) is made now.
! Consequently, Wrk(:,N).ne.0, unlike its counterpart W(:,:,N).eqv.0
! in omega.F. Once omega vertical velocity is computed, interpolate
! it to vertical RHO-points.
!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      do j=jmin,jmax
        do i=imin,imax
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo
        enddo
      enddo




      do j=jmin,jmax
        do i=imin+1,imax
            dn_u(i,j) = 2./(pn(i,j)+pn(i-1,j))
            do k=1,N,+1
              FlxU(i,j,k) = 0.5*(Hz(i,j,k)+Hz(i-1,j,k))*dn_u(i,j)
     & * u(i,j,k)
            enddo
          enddo
      enddo



      do j=jmin+1,jmax
        do i=imin,imax
            dm_v(i,j) = 2./(pm(i,j)+pm(i,j-1))
            do k=1,N,+1
              FlxV(i,j,k) = 0.5*(Hz(i,j,k)+Hz(i,j-1,k))*dm_v(i,j)
     & * v(i,j,k)
            enddo
          enddo
      enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      do j=jmin,jmax-1

        do i=imin,imax
          Wrk(i,0)=0.
        enddo

        do k=1,N,+1
          do i=imin,imax-1
            Wrk(i,k)=Wrk(i,k-1)-pm(i,j)*pn(i,j)*(
     & FlxU(i+1,j,k)-FlxU(i,j,k)
     & +FlxV(i,j+1,k)-FlxV(i,j,k))
          enddo
        enddo

        do i=imin+1,imax
          Wvlc(i,j,N)=+0.375*Wrk(i,N) +0.75*Wrk(i,N-1)
     & -0.125*Wrk(i,N-2)
        enddo
        do k=N-1,2,-1
          do i=imin+1,imax
            Wvlc(i,j,k)=+0.5625*(Wrk(i,k )+Wrk(i,k-1))
     & -0.0625*(Wrk(i,k+1)+Wrk(i,k-2))
          enddo
        enddo
        do i=imin+1,imax
          Wvlc(i,j, 1)= -0.125*Wrk(i,2) +0.75*Wrk(i,1)
     & +0.375*Wrk(i,0)
        enddo
      enddo
!
! Compute and add contributions due to (quasi-)horizontal motions
! along S=const surfaces by multiplying horizontal velocity
! components by slops S-coordinate surfaces:
!
      do k=1,N
        do j=jmin,jmax
          do i=imin+1,imax
            Wxi(i,j)=u(i,j,k)*(z_r(i,j,k)-z_r(i-1,j,k))
     & *(pm(i,j)+pm(i-1,j))
          enddo
        enddo
        do j=jmin+1,jmax
          do i=imin,imax
            Weta(i,j)=v(i,j,k)*(z_r(i,j,k)-z_r(i,j-1,k))
     & *(pn(i,j)+pn(i,j-1))
          enddo
        enddo
        do j=jmin+1,jmax-1
          do i=imin+1,imax-1
            Wvlc(i,j,k)=Wvlc(i,j,k)+0.25*(Wxi(i,j)+Wxi(i+1,j)
     & +Weta(i,j)+Weta(i,j+1))
          enddo
        enddo
      enddo
# 174 "R_tools_fort_routines/get_wvlcty.F"
      return
      end
# 110 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/get_omega.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute vertical velocity
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






      subroutine get_omega(Lm,Mm,N,u,v, z_r,z_w,pm,pn
     & ,W)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4


      real*8 wrk(0:Lm+1),W(0:Lm+1,0:Mm+1,0:N)

# 1 "R_tools_fort_routines/scalars.h" 1
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

      real ,parameter :: sec2day=1./86400
# 35 "R_tools_fort_routines/get_omega.F" 2




Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn
Cf2py intent(out) W

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


      istr=1
      iend=Lm
      jstr=1
      jend=Mm

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!
! Compute "omega" vertical velocity by means of integration of mass
! divergence of mass fluxes from bottom up. In this computation,
! unlike that in omega.F, there is (1) immediate multiplication by
! pm*pn so that the result has meaning of velocity, rather than
! finite volume mass flux through vertical facet of tracer grid box;
! and (2, also unlike omega.F) no subtraction of vertical velocity
! of moving grid-box interface (the effect of "breething" of vertical
! grid system due to evolving free surface) is made now.
! Consequently, Wrk(:,N).ne.0, unlike its counterpart W(:,:,N).eqv.0
! in omega.F. Once omega vertical velocity is computed, interpolate
! it to vertical RHO-points.
!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      do j=jmin,jmax
        do i=imin,imax
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo
        enddo
      enddo




      do j=jmin,jmax
        do i=imin+1,imax
            dn_u(i,j) = 2./(pn(i,j)+pn(i-1,j))
            do k=1,N,+1
              FlxU(i,j,k) = 0.5*(Hz(i,j,k)+Hz(i-1,j,k))*dn_u(i,j)
     & * u(i,j,k)
            enddo
          enddo
      enddo



      do j=jmin+1,jmax
        do i=imin,imax
            dm_v(i,j) = 2./(pm(i,j)+pm(i,j-1))
            do k=1,N,+1
              FlxV(i,j,k) = 0.5*(Hz(i,j,k)+Hz(i,j-1,k))*dm_v(i,j)
     & * v(i,j,k)
            enddo
          enddo
      enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jstr,jend
        do i=istr,iend
          W(i,j,0)=0.
        enddo

        do k=1,N,+1 !--> recursive
          do i=istr,iend
            W(i,j,k)=W(i,j,k-1) -FlxU(i+1,j,k)+FlxU(i,j,k)
     & -FlxV(i,j+1,k)+FlxV(i,j,k)
          enddo
        enddo

        do i=istr,iend
          wrk(i)=W(i,j,N)/(z_w(i,j,N)-z_w(i,j,0))
          W(i,j,N)=0.
        enddo

        do k=N-1,1,-1
          do i=istr,iend
            W(i,j,k)=W(i,j,k)-wrk(i)*(z_w(i,j,k)-z_w(i,j,0))
          enddo
        enddo
      enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jstr,jend
        do i=istr,iend
           do k=1,N,+1

             W(i,j,k)=W(i,j,k)*pm(i,j)*pn(i,j)

          enddo
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                       ! Set lateral
        do k=0,N ! boundary
          do j=jstr,jend ! conditions
            W(istr-1,j,k)=W(istr,j,k)
            W(iend+1,j,k)=W(iend,j,k)
          enddo
        enddo

        do k=0,N
          do i=istr,iend
            W(i,jstr-1,k)=W(i,jstr,k)
            W(i,jend+1,k)=W(i,jend,k)
          enddo
        enddo

        do k=0,N
          W(istr-1,jstr-1,k)=W(istr,jstr,k)
        enddo

        do k=0,N
          W(istr-1,jend+1,k)=W(istr,jend,k)
        enddo

        do k=0,N
          W(iend+1, jstr-1,k)=W(iend,jstr,k)
        enddo

        do k=0,N
          W(iend+1,jend+1,k)=W(iend,jend,k)
        enddo

      return
      end
# 112 "R_tools_fort.F" 2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute ROMS stuffs
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# 1 "R_tools_fort_routines/get_swr_frac.F" 1


      subroutine get_swr_frac (Lm,Mm,N, Hz, swr_frac )
!
! Compute fraction of solar shortwave flux penetrating to the
! specified depth due to exponential decay in Jerlov water type
! using Paulson and Simpson (1977) two-wavelength-band solar
! absorption model.
!
! Reference: Paulson, C.A., and J.J. Simpson, 1977: Irradiance
! meassurements in the upper ocean, J. Phys. Oceanogr., 7, 952-956.
!
! This routine was adapted from Bill Large 1995 code.
!
! output: swr_frac (in "mixing.h") shortwave radiation fraction
!
      implicit none

      integer Lm,Mm,N, i,j,k
      integer istr,iend,jstr,jend, Jwt

      real*8 swdk1(0:Lm+1), swdk2(0:Lm+1)
      real*8 mu1(5),mu2(5), r1(5), attn1, attn2, xi1,xi2

      real*8 Hz(0:Lm+1,0:Mm+1,N)
     & ,swr_frac(0:Lm+1,0:Mm+1,0:N)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N, Hz
Cf2py intent(out) swr_frac

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      istr=0
      iend=Lm+1
      jstr=0
      jend=Mm+1


      mu1(1)=0.35 ! reciprocal of the absorption coefficient
      mu1(2)=0.6 ! for each of the two solar wavelength bands
      mu1(3)=1.0 ! as a function of Jerlov water type (Paulson
      mu1(4)=1.5 ! and Simpson, 1977) [dimensioned as length,
      mu1(5)=1.4 ! meters];

      mu2(1)=23.0
      mu2(2)=20.0
      mu2(3)=17.0
      mu2(4)=14.0
      mu2(5)=7.9

      r1(1)=0.58 ! fraction of the total radiance for
      r1(2)=0.62 ! wavelength band 1 as a function of Jerlov
      r1(3)=0.67 ! water type (fraction for band 2 is always
      r1(4)=0.77 ! r2=1-r1);
      r1(5)=0.78
                     ! set Jerlov water type to assign everywhere
      Jwt=1 ! (an integer from 1 to 5).

      attn1=-1./mu1(Jwt)
      attn2=-1./mu2(Jwt)


      do j=jstr,jend ! Algorithm: set fractions
        do i=istr,iend
          swdk1(i)=r1(Jwt) ! surface, then attenuate
          swdk2(i)=1.-swdk1(i) ! them separately throughout
          swr_frac(i,j,N)=1. ! the water column.
        enddo

        do k=N,1,-1
          do i=istr,iend
            xi1=attn1*Hz(i,j,k)
            if (xi1 .gt. -20.) then ! this logic to avoid
              swdk1(i)=swdk1(i)*exp(xi1) ! computing exponent for
            else ! a very large argument
              swdk1(i)=0.
            endif

            xi2=attn2*Hz(i,j,k)
            if (xi2 .gt. -20.) then
              swdk2(i)=swdk2(i)*exp(xi2)
            else
              swdk2(i)=0.
            endif
            swr_frac(i,j,k-1)=swdk1(i)+swdk2(i)
          enddo
        enddo
      enddo

      return
      end
# 120 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines_gula/get_ghat.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      subroutine get_ghat (Lm,Mm,N,alpha,beta, z_r,z_w
     & ,stflx, srflx, hbl, swr_frac
     & ,ghat)

      implicit none

      integer Lm,Mm,N,NT, i,j,k
     & ,imin,imax,jmin,jmax
     & ,itemp,isalt

      real epsil

      real nubl, nu0c, Cv, Ricr, Ri_inv, betaT, epssfc, C_Ek, C_MO,
     & Cstar, Cg, eps, zeta_m, a_m, c_m, zeta_s, a_s, c_s,
     & r2,r3,r4
      parameter (nubl=0.01,
     & nu0c=0.1,Cv=1.8,Ricr=0.45,Ri_inv=1./Ricr,
     & betaT=-0.2,epssfc=0.1,C_MO=1.,C_Ek=258.,
     & Cstar=10.,eps=1.E-20,zeta_m=-0.2,a_m=1.257,
     & c_m=8.360,zeta_s=-1.0,a_s=-28.86,c_s=98.96,
     & r2=0.5, r3=1./3., r4=0.25)

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
# 29 "R_tools_fort_routines_gula/get_ghat.F" 2

      parameter (NT=2)
      parameter (itemp=1,isalt=2)
      parameter (epsil=1.E-16)


      real*8 sigma, Bfsfc
     & ,Bo(0:Lm+1,0:Mm+1), Bosol(0:Lm+1,0:Mm+1)
     & , Bfsfc_bl(0:Lm+1)
     & ,hbl(0:Lm+1,0:Mm+1), z_bl

      integer kbl(0:Lm+1)

      real*8 stflx(0:Lm+1,0:Mm+1,NT)
     & ,srflx(0:Lm+1,0:Mm+1)
     & ,ghat(0:Lm+1,0:Mm+1,N)
     & ,swr_frac(0:Lm+1,0:Mm+1,0:N)
     & ,alpha(0:Lm+1,0:Mm+1), beta(0:Lm+1,0:Mm+1)
     & ,z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N)
! & ,Hz(0:Lm+1,0:Mm+1,N)



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
# 55 "R_tools_fort_routines_gula/get_ghat.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,alpha,beta ,z_r,z_w,stflx,srflx, hbl, swr_frac
Cf2py intent(out) ghat

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      Cg=Cstar * vonKar * (c_s*vonKar*epssfc)**(1./3.)



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



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1



! do j=jmin,jmax
! do i=imin,imax
! do k=1,N,+1
! Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
! enddo
! enddo
! enddo


       !!call get_swr_frac (Lm,Mm,N, Hz, swr_frac )


      do j=jmin,jmax
        do i=imin,imax
          Bo(i,j)=g*( alpha(i,j)*(stflx(i,j,itemp)-srflx(i,j))

     & -beta(i,j)*stflx(i,j,isalt)

     & )
          Bosol(i,j)=g*alpha(i,j)*srflx(i,j)
        enddo

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

        do i=istr,iend

          k=kbl(i)
          z_bl=z_w(i,j,N)-hbl(i,j)

          if (swr_frac(i,j,k-1).gt. 0.) then
            Bfsfc=Bo(i,j) +Bosol(i,j)*( 1. -swr_frac(i,j,k-1)
     & *swr_frac(i,j,k)*(z_w(i,j,k)-z_w(i,j,k-1))
     & /( swr_frac(i,j,k )*(z_w(i,j,k) -z_bl)
     & +swr_frac(i,j,k-1)*(z_bl -z_w(i,j,k-1))
     & ))
          else
            Bfsfc=Bo(i,j)+Bosol(i,j)
          endif


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

!
! Compute vertical mixing coefficients
!
            sigma=(z_w(i,j,N)-z_w(i,j,k))/max(hbl(i,j),eps)



            if (Bfsfc .lt. 0.) then
              ghat(i,j,k)=Cg * sigma*(1.-sigma)**2
            else
              ghat(i,j,k)=0.
            endif

          enddo


          do k=kbl(i)-1,1,-1

            ghat(i,j,k)=0.

          enddo


        enddo

       enddo



      return
      end
# 122 "R_tools_fort.F" 2

# 1 "R_tools_fort_routines/alfabeta.F" 1


      subroutine alfabeta (Lm,Mm,t,s,rho0,alpha,beta)
!
! Compute thermal expansion and saline contraction coefficients
! as functions of potential temperature, salinity from a polynomial
! expression (Jackett & McDougall, 1992). The coefficients are
! evaluated at the surface.
!
! alpha(Ts,Tt,0)=-d(rho1(Ts,Tt,0))/d(Tt) / rho0
! beta(Ts,Tt,0) = d(rho1(Ts,Tt,0))/d(Ts) / rho0
!
! Adapted from original "rati" and "beta" routines.
!
      implicit none

      integer Lm,Mm, i,j

      integer imin,imax,jmin,jmax

      real t(0:Lm+1,0:Mm+1), s(0:Lm+1,0:Mm+1),
     & alpha(0:Lm+1,0:Mm+1), beta(0:Lm+1,0:Mm+1),
     & rho0
# 1 "R_tools_fort_routines/scalars.h" 1
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

      real ,parameter :: sec2day=1./86400
# 25 "R_tools_fort_routines/alfabeta.F" 2

      real Q01, Q02, Q03, Q04, Q05, U00, U01, U02, U03, U04,
     & V00, V01, V02, W00
      parameter( Q01=6.793952E-2, Q02=-9.095290E-3,
     & Q03=+1.001685E-4, Q04=-1.120083E-6, Q05=+6.536332E-9,
     & U00=+0.824493 , U01=-4.08990E-3 , U02=+7.64380E-5 ,
     & U03=-8.24670E-7 , U04=+5.38750E-9 , V00=-5.72466E-3 ,
     & V01=+1.02270E-4 , V02=-1.65460E-6 , W00=+4.8314E-4 )
      real sqrtTs, cff, Tt, Ts


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Cf2py intent(in) Lm,Mm,t,s, rho0
Cf2py intent(out) alpha,beta

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        imin=0
        imax=Lm+1
        jmin=0
        jmax=Mm+1

      cff=1./rho0
      do j=jmin,jmax
        do i=imin,imax
          Tt=t(i,j)

          Ts=s(i,j)
          sqrtTs=sqrt(Ts)






! ! rho1=(dr00+Tt*( Q01+Tt*( Q02+Tt*( Q03+Tt*(
! ! & Q04+Tt*Q05 ))))
! ! & +Ts*( U00+Tt*( U01+Tt*( U02+Tt*(
! ! & U03+Tt*U04 )))
! ! & +sqrtTs*(V00+Tt*(
! ! & V01+Tt*V02 ))+Ts*W00 ))

          alpha(i,j)=-cff*( Q01+Tt*( 2.*Q02+Tt*( 3.*Q03+Tt*(
     & 4.*Q04 +Tt*5.*Q05 )))
     & +Ts*( U01+Tt*( 2.*U02+Tt*(
     & 3.*U03 +Tt*4.*U04 ))
     & +sqrtTs*( V01+Tt*2.*V02))
     & )

          beta(i,j)= cff*( U00+Tt*(U01+Tt*(U02+Tt*(U03+Tt*U04)))
     & +1.5*(V00+Tt*(V01+Tt*V02))*sqrtTs+2.*W00*Ts )
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      return
      end
# 124 "R_tools_fort.F" 2


# 1 "R_tools_fort_routines/get_hbbl.F" 1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subpart of the lmd_kpp.F routine (myroms version)
! used to compute the new hbl
! (the part used to compute the new Kv, Kt has been removed)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






c----#define WND_AT_RHO_POINTS

      subroutine get_hbbl (Lm,Mm,N,alpha,beta, z_r,z_w
     & , stflx, srflx, swr_frac, sustr, svstr ,Ricr, hbls, f
     & , u, v, bvf
     & , hbbl, out1, out2, out3, out4)

      implicit none

      integer Lm,Mm,N,NT, i,j,k
     & ,istr,iend,jstr,jend
     & ,itemp,isalt

      real epsil

      real nubl, nu0c, Cv, Ricr, Ri_inv, betaT, epssfc, C_Ek, C_MO,
     & Cstar, Cg, eps, zeta_m, a_m, c_m, zeta_s, a_s, c_s,
     & r2,r3,r4
      parameter (nubl=0.01,
     & nu0c=0.1,Cv=1.8,
     & betaT=-0.2,epssfc=0.1,C_MO=1.,C_Ek=258.,
     & Cstar=10.,eps=1.E-20,zeta_m=-0.2,a_m=1.257,
     & c_m=8.360,zeta_s=-1.0,a_s=-28.86,c_s=98.96,
     & r2=0.5, r3=1./3., r4=0.25)

# 1 "R_tools_fort_routines/scalars.h" 1
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

      real ,parameter :: sec2day=1./86400
# 40 "R_tools_fort_routines/get_hbbl.F" 2

      parameter (NT=2)
      parameter (itemp=1,isalt=2)
      parameter (epsil=1.E-16)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      real*8 sigma
     & ,Bo(0:Lm+1,0:Mm+1), Bosol(0:Lm+1,0:Mm+1)
     & ,Bfsfc_bl(0:Lm+1)
     & ,z_bl
     & ,ustar(0:Lm+1,0:Mm+1)
     & ,Cr(0:Lm+1,0:N)
     & ,FC(0:Lm+1,0:N)
     & ,wrk1(0:Lm+1,0:N)
     & ,wrk2(0:Lm+1,0:N)
     & ,cff, cff1
     & ,Hz(0:Lm+1,0:Mm+1,N)

     & ,FX(0:Lm+1,0:Mm+1)
     & ,FE(0:Lm+1,0:Mm+1)
     & ,FE1(0:Lm+1,0:Mm+1)




      integer kbl(0:Lm+1)

      real Kern, Vtc, Vtsq
     & , Bfsfc,zscale
     & , ustar3, zetahat, ws, wm


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      real*8 hbl(0:Lm+1,0:Mm+1)

      ! Variables IN
      real*8 stflx(0:Lm+1,0:Mm+1,NT)
     & ,srflx(0:Lm+1,0:Mm+1)
     & ,swr_frac(0:Lm+1,0:Mm+1,0:N)
     & ,alpha(0:Lm+1,0:Mm+1), beta(0:Lm+1,0:Mm+1)
     & ,z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N)
     & ,sustr(1:Lm+1,0:Mm+1), svstr(0:Lm+1,1:Mm+1)
     & ,hbls(0:Lm+1,0:Mm+1), f(0:Lm+1,0:Mm+1)
     & ,u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N)
     & ,bvf(0:Lm+1,0:Mm+1,0:N)

      ! Variables OUT
      real*8 hbbl(0:Lm+1,0:Mm+1)
     & , out1(0:Lm+1,0:Mm+1,0:N), out2(0:Lm+1,0:Mm+1,0:N)
     & , out3(0:Lm+1,0:Mm+1,0:N), out4(0:Lm+1,0:Mm+1,0:N)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,alpha,beta ,z_r,z_w,stflx,srflx, swr_frac, sustr, svstr ,Ricr,hbls, f, u, v, bvf
Cf2py intent(out) hbbl, out1, out2, out3, out4

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      Ri_inv=1./Ricr
      Cg=Cstar * vonKar * (c_s*vonKar*epssfc)**(1./3.)
      Vtc=Cv * sqrt(-betaT/(c_s*epssfc)) / (Ricr*vonKar**2)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1


        istr=0+1
        iend=Lm
        jstr=0+1
        jend=Mm







!================================

      do j=jstr,jend
        do i=istr,iend

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

!! hbl(i,j)=0. !<-- use old value as initial guess
          kbl(i)=0







          FC(i,N)=0.
          Cr(i,N)=0.
          Cr(i,0)=0.

          out1(i,j,N)=0.
          out2(i,j,N)=0.
          out3(i,j,N)=0.
          out4(i,j,N)=0.


        enddo

!======================================
! Integral at W-points
!======================================



!

!================================
! Search for mixed layer depth
!================================
!





      do k=1,N-1
        do i=istr,iend
          cff=1./(Hz(i,j,k)+Hz(i,j,k+1))
          wrk1(i,k)=cff*( u(i,j,k+1)+u(i+1,j,k+1)
     & -u(i,j,k )-u(i+1,j,k ))
          wrk2(i,k)=cff*( v(i,j,k+1)+v(i,j+1,k+1)
     & -v(i,j,k )-v(i,j+1,k ))
        enddo
      enddo

      do i=istr,iend
        wrk1(i,N)=wrk1(i,N-1)
        wrk2(i,N)=wrk2(i,N-1)
        wrk1(i,0)=wrk1(i, 1)
        wrk2(i,0)=wrk2(i, 1)
      enddo


      do k=N,1,-1
        do i=istr,iend
          zscale=z_w(i,j,N)-z_w(i,j,k-1)
          Kern=zscale/(zscale+epssfc*hbl(i,j))
          Bfsfc=Bo(i,j) +Bosol(i,j)*(1.-swr_frac(i,j,k-1))



# 1 "R_tools_fort_routines/lmd_wscale_ws_only.h" 1
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
# 216 "R_tools_fort_routines/get_hbbl.F" 2



          cff=bvf(i,j,k)*bvf(i,j,k-1)
          if (cff.gt.0.D0) then
            cff=cff/(bvf(i,j,k)+bvf(i,j,k-1))
          else
            cff=0.D0
          endif


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


          out1(i,j,k-1) = out1(i,j,k)+ Kern*Hz(i,j,k)*(
     & 0.375*( wrk1(i,k)**2 + wrk1(i,k-1)**2
     & +wrk2(i,k)**2 + wrk2(i,k-1)**2 )
     & +0.25 *(wrk1(i,k-1)*wrk1(i,k)+wrk2(i,k-1)*wrk2(i,k))
     & )

          out2(i,j,k-1) = out2(i,j,k)+ Kern*Hz(i,j,k)*(
     & -Ri_inv*( cff + 0.25*(bvf(i,j,k)+bvf(i,j,k-1)))
     & )

          out3(i,j,k-1) = out3(i,j,k)+ Kern*Hz(i,j,k)*(
     & -C_Ek*f(i,j)*f(i,j)
     & )

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


          FC(i,k-1)=FC(i,k) + Kern * Hz(i,j,k) * (
     & 0.375*( wrk1(i,k )**2 +wrk1(i,k-1)**2
     & +wrk2(i,k )**2 +wrk2(i,k-1)**2 )
     & +0.25 *( wrk1(i,k-1) * wrk1(i,k)
     & +wrk2(i,k-1) * wrk2(i,k) )
     & -Ri_inv*( cff + 0.25*( bvf(i,j,k)+bvf(i,j,k-1)) )
     & -C_Ek*f(i,j)*f(i,j)
     & )


          Vtsq=Vtc*ws*sqrt(max(0., bvf(i,j,k-1)))

          out4(i,j,k-1) = Vtsq

          Cr(i,k)=FC(i,k)+Vtsq

        enddo
      enddo
# 273 "R_tools_fort_routines/get_hbbl.F"
!
!======================================
! Search for bottom mixed layer depth
!======================================
!
        do i=istr,iend
          kbl(i) = 0 ! reset Cr at bottom and kbl for BKPP
          Cr(i,0) = 0.
        enddo
        do k=1,N,+1
          do i=istr,iend
            Cr(i,k)=FC(i,k)-FC(i,0)
            if (kbl(i).eq.0 .and. Cr(i,k).gt.0.) kbl(i)=k
          enddo
        enddo
        do i=istr,iend
          hbbl(i,j)=z_w(i,j,N)-z_w(i,j,0) !+eps ! total depth
          if (kbl(i).gt.0) then
            k=kbl(i)
            if (k.eq.1) then
              hbbl(i,j)=z_r(i,j,1)-z_w(i,j,0) ! no BBL
            else
              hbbl(i,j)=( z_w(i,j,k-1)*Cr(i,k)-z_w(i,j,k)*Cr(i,k-1)
     & )/(Cr(i,k)-Cr(i,k-1) ) - z_w(i,j,0)
            endif
          endif
        enddo !--> discard FC, Cr and kbl




!======================================


      enddo !<-- j





# 1 "R_tools_fort_routines/kpp_smooth.h" 1
!
! Apply horizontal smoothing operator to hbbl, while avoiding land-
! masked values which is accomplished by expressing everything in
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
      do j=jstr,jend
        do i=istr,iend
          FX(i,j)=(hbbl(i,j)-hbbl(i-1,j))
        enddo
      enddo
      do j=jstr,jend
        do i=istr,iend
          FE(i,j)=(hbbl(i,j)-hbbl(i,j-1))
        enddo
      enddo
      cff=1.D0/12.D0
      cff1=3.D0/16.D0
      do j=jstr,jend
        do i=istr,iend
          FE1(i,j)=FE(i,j)+cff*( FX(i+1,j)+FX(i,j-1)
     & -FX(i,j)-FX(i+1,j-1))
        enddo
      enddo
      do j=jstr,jend
        do i=istr,iend
          FX(i,j)=FX(i,j)+cff*( FE(i,j+1)+FE(i-1,j)
     & -FE(i,j)-FE(i-1,j+1))
        enddo
      enddo
      do j=jstr,jend
        do i=istr,iend
          hbbl(i,j)=hbbl(i,j)+cff1*( FX(i+1,j)-FX(i,j)
     & +FE1(i,j+1)-FE1(i,j))
        enddo !--> discard FX,FE,FE1
      enddo
# 314 "R_tools_fort_routines/get_hbbl.F" 2




!======================================


      return
      end
# 127 "R_tools_fort.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
