# 1 "R_tools_fort_gula.F"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "R_tools_fort_gula.F"

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

# 1 "R_tools_fort_routines_gula/cppdefs.h" 1
# 15 "R_tools_fort_routines_gula/cppdefs.h"
c-# define PV_CUBIC
# 63 "R_tools_fort_routines_gula/cppdefs.h"
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
# 64 "R_tools_fort_routines_gula/cppdefs.h" 2
# 32 "R_tools_fort_gula.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Compute density anomaly (adapted from rho_eos.F)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# 1 "R_tools_fort_routines_gula/rho_eos_nozw.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Compute density anomaly (adapted from rho_eos.F)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine rho_eos_nozw(Lm,Mm,N, T,S, z_r,rho0, rho)



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


Cf2py intent(in) Lm,Mm,N, T,S, z_r, rho0
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


         rho(i,j,k) = rho1(i,j,k) + qp1(i,j,k)*(0.-z_r(i,j,k))


          enddo
        enddo



      enddo ! <-- j

      return
      end
# 39 "R_tools_fort_gula.F" 2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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
# 45 "R_tools_fort_gula.F" 2



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute PV
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

c---# include "R_tools_fort_routines_gula/get_diagsPV_sol1.F"

c---# include "R_tools_fort_routines_gula/get_diagsPV_sol2.F"


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute PV fluxes
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



# 1 "R_tools_fort_routines_gula/old/get_J1_sol1.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute PV
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_J1_sol1(Lm,Mm,N, stflx,ssflx, u,v, z_r,z_w,rho0,pm,
     & pn,hbls,f,J1)

      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k
      real*8 stflx(0:Lm+1,0:Mm+1), ssflx(0:Lm+1,0:Mm+1),
     & u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & J1(1:Lm+1,1:Mm+1),f(0:Lm+1,0:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & hbls(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & dpth,cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00,
     & var1, var2,var3, var4,cff3,
     & absvrt(1:Lm+1,1:Mm+1),absvrt0(1:Lm+1,1:Mm+1)

      real*8, parameter :: g=9.81


Cf2py intent(in) Lm,Mm,N, stflx,ssflx, u,v,z_r,z_w,rho0,pm,pn,hbls,f
Cf2py intent(out) J1


      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE VORTICITY
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!---------------------------------------------------------------------------------------

       do i=imin+1,imax
        do j=jmin+1,jmax

            dpth=0.25*(z_r(i,j,N)+z_r(i-1,j,N)
     & + z_r(i-1,j-1,N)+z_r(i,j-1,N))

            CALL interp_1d(N,v(i,j,:)
     & ,0.5*(z_r(i,j,:)+z_r(i,j-1,:))
     & ,0.5*(z_w(i,j,:)+z_w(i,j-1,:))
     & ,dpth,var1,1,0)

            CALL interp_1d(N,v(i-1,j,:)
     & ,0.5*(z_r(i-1,j,:)+z_r(i-1,j-1,:))
     & ,0.5*(z_w(i-1,j,:)+z_w(i-1,j-1,:))
     & ,dpth,var2,1,0)


            CALL interp_1d(N,u(i,j,:)
     & ,0.5*(z_r(i-1,j,:)+z_r(i,j,:))
     & ,0.5*(z_w(i-1,j,:)+z_w(i,j,:))
     & ,dpth,var3,1,0)

            CALL interp_1d(N,u(i,j-1,:)
     & ,0.5*(z_r(i,j-1,:)+z_r(i-1,j-1,:))
     & ,0.5*(z_w(i,j-1,:)+z_w(i-1,j-1,:))
     & ,dpth,var4,1,0)

            cff = 0.25*(f(i,j) + f(i-1,j) + f(i,j-1) + f(i-1,j-1))
            cff2 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
            cff3 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

            absvrt(i,j)= cff
     & + (var1-var2) * cff2
     & - (var3-var4) * cff3


c absvrt0(i,j)= cff
c & + (v(i,j,N)-v(i-1,j,N)) * cff2
c & - (u(i,j,N)-u(i,j-1,N)) * cff3

c write(*,*) i,j,dpth,z_r(i,j,N),absvrt(i,j), absvrt0(i,j)


         enddo
       enddo


!---------------------------------------------------------------------------------------





!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        do j=jmin+1,jmax
          do i=imin+1,imax

                cff = g/((hbls(i,j) + hbls(i-1,j)
     & + hbls(i,j-1) + hbls(i-1,j-1)))



            J1(i,j) = cff * absvrt(i,j) * (
     & (stflx(i,j)+stflx(i-1,j)+stflx(i,j-1)+stflx(i-1,j-1))
     & - (ssflx(i,j)+ssflx(i-1,j)+ssflx(i,j-1)+ssflx(i-1,j-1))
     & )


          enddo
        enddo


      return
      end
# 64 "R_tools_fort_gula.F" 2
# 1 "R_tools_fort_routines_gula/old/get_J2_sol1.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute PV
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_J2_sol1(Lm,Mm,N, T,S, u,v, z_r,z_w,rho0,pm,pn,
     & hbls,J2)


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k
      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & J2(1:Lm+1,1:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & hbls(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & rho1(0:Lm+1,0:Mm+1,N),
     & drdx(1:Lm+1,0:Mm+1), drdy(0:Lm+1,1:Mm+1),
     & qp1(0:Lm+1,0:Mm+1,N),
     & dpth,cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00,
     & dvdx, dudy,
     & cffi(0:Lm+1), cffj(0:Mm+1),
     & var1, var2,var3, var4

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


Cf2py intent(in) Lm,Mm,N, T,S, u,v,z_r,z_w,rho0,pm,pn,hbls
Cf2py intent(out) J2
# 60 "R_tools_fort_routines_gula/old/get_J2_sol1.F"
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE NEUTRAL DENSITY GRADIENTS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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


!---------------------------------------------------------------------------------------
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
      enddo ! <-- j



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE BUOYANCY GRADIENTS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        cff=g/rho0


!---------------------------------------------------------------------------------------

       do i=imin+1,imax

        do j=jmin,jmax

            dpth=0.5*(z_r(i,j,N)+z_r(i-1,j,N))

            !if (dpth.gt.z_w(i,j,N)) then
            ! write(*,*) 'calling', dpth, z_r(i,j,N), z_w(i,j,N)
            !endif

            !if (dpth.gt.z_w(i-1,j,N)) then
            ! write(*,*) 'calling 2', dpth, z_r(i,j,N), z_w(i,j,N)
            !endif

            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var1,1,0)

            CALL interp_1d(N,rho1(i-1,j,:),z_r(i-1,j,:),z_w(i-1,j,:)
     & ,dpth,var2,1,0)



            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var3,1,0)
            CALL interp_1d(N,qp1(i-1,j,:),z_r(i-1,j,:),z_w(i-1,j,:)
     & ,dpth,var4,1,0)

            cffj(j)=-cff*( var1 - var2 ! Elementary
     & +(var3-var4) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference
     & *0.5*(pm(i,j)+pm(i-1,j))
         enddo



         do j=jmin+1,jmax

            drdx(i,j)= 0.5 * (cffj(j) + cffj(j-1))
            !!write(*,*) 'all3',dpth,cffj(j),cffj(j-1), drdx(i,j,k)

         enddo

         !write(*,*) 'all2',i,j,drdx(i,j)


       enddo


!---------------------------------------------------------------------------------------

        !write(*,*) 'bouble'

        cff=g/rho0

        do j=jmin+1,jmax

          do i=imin,imax
            dpth=0.5*(z_r(i,j,N)+z_r(i,j-1,N))


            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var1,1,0)
            CALL interp_1d(N,rho1(i,j-1,:),z_r(i,j-1,:),z_w(i,j-1,:)
     & ,dpth,var2,1,0)

            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var3,1,0)
            CALL interp_1d(N,qp1(i,j-1,:),z_r(i,j-1,:),z_w(i,j-1,:)
     & ,dpth,var4,1,0)




            cffi(i)=-cff*( var1 - var2 ! Elementary
     & +(var3-var4) ! adiabatic
     & *dpth*(1.-qp2*dpth) )
     & *0.5*(pn(i,j)+pn(i,j-1))


         !write(*,*) 'all2',i,j,dpth,cffi(i),var1,var2

          enddo



          do i=imin+1,imax

            drdy(i,j)= 0.5 * (cffi(i) + cffi(i-1))


          enddo



        enddo





!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        do j=jmin+1,jmax
          do i=imin+1,imax

            cff = 0.5/(rho0*0.25*(hbls(i,j) + hbls(i-1,j)
     & + hbls(i,j-1) + hbls(i-1,j-1)))

            J2(i,j) = cff * (drdy(i,j) * (u(i,j-1) + u(i,j))
     & - drdx(i,j) * (v(i-1,j)+v(i,j) ))

            !!write(*,*) 'all3',i,j,cff,drdy(i,j),u(i,j-1),J2(i,j)
          enddo
        enddo






      return
      end
# 65 "R_tools_fort_gula.F" 2
# 1 "R_tools_fort_routines_gula/old/get_Jbot_sol1.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute PV
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_Jbot_sol1(Lm,Mm,N, T,S, u,v, z_r,z_w,rho0,pm,pn,
     & hbbls,rdrg,Jbot)


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k
      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & Jbot(1:Lm+1,1:Mm+1),
     & dx(1:Lm+1,1:Mm+1),
     & dy(1:Lm+1,1:Mm+1),
     & dz(1:Lm+1,1:Mm+1),
     & rd(0:Lm+1,0:Mm+1),
     & ubot(1:Lm+1,1:Mm+1),vbot(1:Lm+1,1:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & hbbls(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1),
     & rho1(0:Lm+1,0:Mm+1,N), drdz(0:Lm+1,0:Mm+1),
     & drdx(1:Lm+1,0:Mm+1), drdy(0:Lm+1,1:Mm+1),
     & qp1(0:Lm+1,0:Mm+1,N),
     & dpth,cff,Tt,Ts,sqrtTs, rho0, K0, dr00,
     & cff2,cff3,
     & dvdx, dudy,
     & cffi(0:Lm+1), cffj(0:Mm+1),
     & var1, var2,var3, var4


      real*8 Zob, rdrg

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
     & qp2=0.0000172


      real rho1_0, K0_Duk


# 1 "R_tools_fort_routines_gula/old/scalars.h" 1
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
# 57 "R_tools_fort_routines_gula/old/get_Jbot_sol1.F" 2

Cf2py intent(in) Lm,Mm,N, T,S, u,v,z_r,z_w,rho0,pm,pn,hbbls,rdrg
Cf2py intent(out) Jbot
# 69 "R_tools_fort_routines_gula/old/get_Jbot_sol1.F"
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE NEUTRAL DENSITY GRADIENTS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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


!---------------------------------------------------------------------------------------
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

! if ((i.eq.100).and.(j.eq.100)) then
! write(*,*) i,j,k,rho1(i,j,k)
! endif
          enddo
        enddo
      enddo ! <-- j



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE BUOYANCY GRADIENTS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        cff=g/rho0



!---------------------------------------------------------------------------------------







      do j=jmin,jmax
          do i=imin,imax

            !dpth=z_w(i,j,N)-0.5*(z_r(i,j,k+1)+z_r(i,j,k))
            dpth=z_r(i,j,1)

            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,0),var1,1,0)
            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,1),var2,1,0)

            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,0),var3,1,0)
            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,1),var4,1,0)

            cff2=( var2-var1 ! Elementary
     & +(var4-var3) ! adiabatic
     & *dpth*(1.-2.*qp2*dpth) ! difference
     & )


            drdz(i,j) =- cff*cff2 / (z_w(i,j,1)-z_w(i,j,0))



          enddo
      enddo ! <-- j






!---------------------------------------------------------------------------------------

       do i=imin+1,imax
        do j=jmin,jmax

            dpth=0.5*(z_r(i,j,1)+z_r(i-1,j,1))

            !if ((z_r(i,j,N).ne.0).and.(z_r(i-1,j,N).ne.0)) then
            !if (dpth.gt.z_w(i,j,N)) then
            ! write(*,*) 'calling', dpth, z_r(i,j,N), z_w(i,j,N)
            ! write(*,*) z_w(i,j,:)
             !write(*,*) z_w(i-1,j,:)
            !endif
            !endif

            !if (dpth.gt.z_w(i-1,j,N)) then
            ! write(*,*) 'calling 2', dpth, z_r(i,j,N), z_w(i,j,N)
            !endif


            !if ((z_r(i,j,N).ne.0).and.(z_r(i-1,j,N).ne.0)) then

! if ((i.eq.100).and.(j.eq.100)) then
! CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
! & ,dpth,var1,1,1)
! else
            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var1,1,0)
! endif

            CALL interp_1d(N,rho1(i-1,j,:),z_r(i-1,j,:),z_w(i-1,j,:)
     & ,dpth,var2,1,0)


            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var3,1,0)
            CALL interp_1d(N,qp1(i-1,j,:),z_r(i-1,j,:),z_w(i-1,j,:)
     & ,dpth,var4,1,0)

            cffj(j)=-cff*( var1 - var2 ! Elementary
     & +(var3-var4) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference
     & *0.5*(pm(i,j)+pm(i-1,j))

! if ((i.eq.100).and.(j.eq.100)) then
! write(*,*) dpth
! write(*,*) rho1(i,j,:)
! write(*,*) z_r(i,j,:)
! write(*,*) z_w(i,j,:)
! write(*,*) i,j,var1,var2
! write(*,*) i,j,var3,var4
! write(*,*) i,j,cffj(j)
! endif

         enddo



         do j=jmin+1,jmax

            drdx(i,j)= 0.5 * (cffj(j) + cffj(j-1))
            !!write(*,*) 'all3',dpth,cffj(j),cffj(j-1), drdx(i,j,k)

         enddo

         !!write(*,*) 'all2',i,j,drdx(i,j)


       enddo


!---------------------------------------------------------------------------------------

        do j=jmin+1,jmax

          do i=imin,imax

            dpth=0.5*(z_r(i,j,1)+z_r(i,j-1,1))

            !write(*,*) 'all2',z_w(i,j,N),z_r(i,j,1),i,j

            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var1,1,0)
            CALL interp_1d(N,rho1(i,j-1,:),z_r(i,j-1,:),z_w(i,j-1,:)
     & ,dpth,var2,1,0)

            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var3,1,0)
            CALL interp_1d(N,qp1(i,j-1,:),z_r(i,j-1,:),z_w(i,j-1,:)
     & ,dpth,var4,1,0)




            cffi(i)=-cff*( var1 - var2 ! Elementary
     & +(var3-var4) ! adiabatic
     & *dpth*(1.-qp2*dpth) )
     & *0.5*(pn(i,j)+pn(i,j-1))


         !!!write(*,*) 'all2',i,j,dpth,cffi(i),var1,var2

          enddo



          do i=imin+1,imax

            drdy(i,j)= 0.5 * (cffi(i) + cffi(i-1))


          enddo



        enddo




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Bottom Drag
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

       Zob=0.01


       do j=jmin+1,jmax-1
         do i=imin+1,imax-1

            Hz(i,j) = z_w(i,j,1) - z_w(i,j,0)




            cff=sqrt( 0.333333333333*(
     & u(i,j)**2 +u(i+1,j)**2
     & +u(i,j)*u(i+1,j)
     & +v(i,j)**2+v(i,j+1)**2
     & +v(i,j)*v(i,j+1)
     & ))
            rd(i,j)=rdrg + cff*(vonKar/log(Hz(i,j)/Zob))**2




          enddo
        enddo

       do j=jmin+1,jmax
            rd(imax ,j)=rd(imax-1 ,j)
            rd(imin ,j)=rd(imin+1 ,j)
        enddo

       do i=imin+1,imax
            rd(i ,jmax)=rd(i ,jmax-1)
            rd(i ,jmin)=rd(i ,jmin+1)
        enddo







!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! HEIGHT GRADIENT
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        do j=jmin+1,jmax
          do i=imin+1,imax

            dx(i,j) = 0.5* (z_r(i,j,1) + z_r(i,j-1,1)
     & - z_r(i-1,j,1)- z_r(i-1,j-1,1))
     & * 0.25*(pm(i,j)+pm(i,j-1)+ pm(i-1,j)+pm(i-1,j-1))

            dy(i,j) = 0.5* (z_r(i,j,1) + z_r(i-1,j,1)
     & - z_r(i,j-1,1)- z_r(i-1,j-1,1))
     & * 0.25*(pn(i,j)+pn(i,j-1)+ pn(i-1,j)+pn(i-1,j-1))


            cff = 1./sqrt(1 + dx(i,j)**2 + dy(i,j)**2)

            dx(i,j) = cff * dx(i,j)
            dy(i,j) = cff * dy(i,j)
            dz(i,j) = cff

            !if (cff.ne.1) then
            ! write(*,*) i,j,dx(i,j), dy(i,j), dz(i,j)
            !endif

          enddo
        enddo




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        do j=jmin+1,jmax
          do i=imin+1,imax


            cff2 = 0.25 * (drdz(i-1,j) + drdz(i,j)
     & + drdz(i-1,j-1) + drdz(i,j-1))
            cff3 =0.25 * (rd(i,j)+rd(i-1,j)
     & + rd(i,j-1)+rd(i-1,j-1))



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            cff = 1./(0.25*(hbbls(i,j) + hbbls(i-1,j)
     & + hbbls(i,j-1) + hbbls(i-1,j-1)))

            ubot(i,j) = cff3*0.5*(u(i,j-1) + u(i,j))

            vbot(i,j) = cff3*0.5*(v(i-1,j)+v(i,j) )

            Jbot(i,j) = cff * ((drdx(i,j) * vbot(i,j)
     & - drdy(i,j) * ubot(i,j)) * dz(i,j)
     & - (cff2 * vbot(i,j)) * dx(i,j)
     & + (cff2 * ubot(i,j)) * dy(i,j)
     & )


          enddo
        enddo






      return
      end
# 66 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_Jbot_fromdiags.F" 1
!-------------------------------------------------------------------------
!
! compute Jbot = [grad b x F] on a psi grid and a rho-level
!
!-------------------------------------------------------------------------
! new version (updated 05/09/17)
! - Fix sign of dx,dy
! - choose vertical level (iz = python lev from 0 to nz-1)
!
!
!-------------------------------------------------------------------------

      subroutine get_Jbot_fromdiags(Lm,Mm,N, T,S, ubot,vbot, z_r,z_w,
     & rho0,pm,pn,iz,Jbot)


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,iz
      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & Jbot(1:Lm+1,1:Mm+1),
     & dx(1:Lm+1,1:Mm+1),dy(1:Lm+1,1:Mm+1),
     & dz(1:Lm+1,1:Mm+1),rd(0:Lm+1,0:Mm+1),
     & ubot(1:Lm+1,0:Mm+1),vbot(0:Lm+1,1:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & rho1(0:Lm+1,0:Mm+1,N), drdz(0:Lm+1,0:Mm+1),
     & drdx(1:Lm+1,0:Mm+1), drdy(0:Lm+1,1:Mm+1),
     & qp1(0:Lm+1,0:Mm+1,N),
     & dpth,cff,Tt,Ts,sqrtTs, rho0, K0, dr00,
     & cff2,cff3,dvdx, dudy,
     & cffi(0:Lm+1), cffj(0:Mm+1),
     & var1, var2,var3, var4


      real*8 Zob, rdrg

      real*8, parameter :: r00=999.842594, r01=6.793952E-2,
     & r02=-9.095290E-3, r03=1.00 -1685E-4, r04=-1.120083E-6,
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
     & qp2=0.0000172


      real rho1_0, K0_Duk


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
# 56 "R_tools_fort_routines_gula/get_Jbot_fromdiags.F" 2

Cf2py intent(in) Lm,Mm,N, T,S, ubot,vbot,z_r,z_w,rho0,pm,pn,iz
Cf2py intent(out) Jbot

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE NEUTRAL DENSITY GRADIENTS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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

!---------------------------------------------------------------------------------------

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
      enddo ! <-- j

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE BUOYANCY GRADIENTS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      cff=g/rho0

!---------------------------------------------------------------------------------------

      do j=jmin,jmax
          do i=imin,imax

            dpth=z_r(i,j,iz+1)

            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,0),var1,1,0)
            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,1),var2,1,0)

            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,0),var3,1,0)
            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,1),var4,1,0)

            cff2=( var2-var1 ! Elementary
     & +(var4-var3) ! adiabatic
     & *dpth*(1.-2.*qp2*dpth) ! difference
     & )

            drdz(i,j) =- cff*cff2 / (z_w(i,j,iz+1)-z_w(i,j,iz))

          enddo
      enddo ! <-- j

!---------------------------------------------------------------------------------------

       do i=imin+1,imax
        do j=jmin,jmax

            dpth=0.5*(z_r(i,j,iz+1)+z_r(i-1,j,iz+1))

            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var1,1,0)
            CALL interp_1d(N,rho1(i-1,j,:),z_r(i-1,j,:),z_w(i-1,j,:)
     & ,dpth,var2,1,0)

            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var3,1,0)
            CALL interp_1d(N,qp1(i-1,j,:),z_r(i-1,j,:),z_w(i-1,j,:)
     & ,dpth,var4,1,0)

            cffj(j)=-cff*( var1 - var2 ! Elementary
     & +(var3-var4) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference
     & *0.5*(pm(i,j)+pm(i-1,j))

         enddo

         do j=jmin+1,jmax

            drdx(i,j)= 0.5 * (cffj(j) + cffj(j-1))

         enddo
       enddo


!---------------------------------------------------------------------------------------

        do j=jmin+1,jmax
          do i=imin,imax

            dpth=0.5*(z_r(i,j,iz+1)+z_r(i,j-1,iz+1))

            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var1,1,0)
            CALL interp_1d(N,rho1(i,j-1,:),z_r(i,j-1,:),z_w(i,j-1,:)
     & ,dpth,var2,1,0)

            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,dpth,var3,1,0)
            CALL interp_1d(N,qp1(i,j-1,:),z_r(i,j-1,:),z_w(i,j-1,:)
     & ,dpth,var4,1,0)

            cffi(i)=-cff*( var1 - var2 ! Elementary
     & +(var3-var4) ! adiabatic
     & *dpth*(1.-qp2*dpth) )
     & *0.5*(pn(i,j)+pn(i,j-1))

          enddo

          do i=imin+1,imax

            drdy(i,j)= 0.5 * (cffi(i) + cffi(i-1))

          enddo
        enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! HEIGHT GRADIENT
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        do j=jmin+1,jmax
          do i=imin+1,imax

            dx(i,j) = 0.5* (z_r(i,j,iz+1) + z_r(i,j-1,iz+1)
     & - z_r(i-1,j,iz+1)- z_r(i-1,j-1,iz+1))
     & * 0.25*(pm(i,j)+pm(i,j-1)+ pm(i-1,j)+pm(i-1,j-1))

            dy(i,j) = 0.5* (z_r(i,j,iz+1) + z_r(i-1,j,iz+1)
     & - z_r(i,j-1,iz+1)- z_r(i-1,j-1,iz+1))
     & * 0.25*(pn(i,j)+pn(i,j-1)+ pn(i-1,j)+pn(i-1,j-1))

            cff = 1./sqrt(1 + dx(i,j)**2 + dy(i,j)**2)

            dx(i,j) = - cff * dx(i,j)
            dy(i,j) = - cff * dy(i,j)
            dz(i,j) = cff

          enddo
        enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        do j=jmin+1,jmax
          do i=imin+1,imax

            cff2 = 0.25 * (drdz(i-1,j) + drdz(i,j)
     & + drdz(i-1,j-1) + drdz(i,j-1))

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            cff = 1.

            Jbot(i,j) = cff * ((drdx(i,j) * vbot(i,j)
     & - drdy(i,j) * ubot(i,j)) * dz(i,j)
     & - (cff2 * vbot(i,j)) * dx(i,j)
     & + (cff2 * ubot(i,j)) * dy(i,j)
     & )

          enddo
        enddo

      return
      end
# 68 "R_tools_fort_gula.F" 2
# 1 "R_tools_fort_routines_gula/get_Jdiab_fromdiags.F" 1
!-------------------------------------------------------------------------
!
! compute Jdiab = [- absvrt(3d) . Db/Dt] on a psi grid and a rho-level
!
!-------------------------------------------------------------------------
! new version (updated 05/09/17)
! - Add horizontal components of vorticity
! - choose vertical level (iz = python lev from 0 to nz-1)
!
!
!-------------------------------------------------------------------------

      subroutine get_Jdiab_fromdiags(Lm,Mm,N,Tdiab,Sdiab,alpha,beta,
     & u,v, z_r,z_w,rho0,pm,pn,f,iz,Jdiab)

      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,iz
      real*8 Tdiab(0:Lm+1,0:Mm+1), Sdiab(0:Lm+1,0:Mm+1),
     & dbdt(0:Lm+1,0:Mm+1),
     & alpha(0:Lm+1,0:Mm+1), beta(0:Lm+1,0:Mm+1),
     & u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & work(1:Lm+1,1:Mm+1,N),
     & uz(1:Lm+1,0:Mm+1), vz(0:Lm+1,1:Mm+1),
     & Jdiab(1:Lm+1,1:Mm+1),f(0:Lm+1,0:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & dpth,cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00,
     & var1, var2,var3, var4,cff3,
     & absvrt(1:Lm+1,1:Mm+1),
     & dx(1:Lm+1,1:Mm+1),dy(1:Lm+1,1:Mm+1),
     & dz(1:Lm+1,1:Mm+1)

      real*8, parameter :: g=9.81

Cf2py intent(in) Lm,Mm,N,Tdiab, Sdiab, alpha, beta, u,v,z_r,z_w,rho0,pm,pn,f,iz
Cf2py intent(out) Jdiab

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

!-------------------------------------------------------------------------
! COMPUTE VERTICAL VORTICITY
!-------------------------------------------------------------------------

       do i=imin+1,imax
        do j=jmin+1,jmax

            dpth=0.25*(z_r(i,j,iz+1)+z_r(i-1,j,iz+1)
     & + z_r(i-1,j-1,iz+1)+z_r(i,j-1,iz+1))

            CALL interp_1d(N,v(i,j,:)
     & ,0.5*(z_r(i,j,:)+z_r(i,j-1,:))
     & ,0.5*(z_w(i,j,:)+z_w(i,j-1,:))
     & ,dpth,var1,1,0)

            CALL interp_1d(N,v(i-1,j,:)
     & ,0.5*(z_r(i-1,j,:)+z_r(i-1,j-1,:))
     & ,0.5*(z_w(i-1,j,:)+z_w(i-1,j-1,:))
     & ,dpth,var2,1,0)

            CALL interp_1d(N,u(i,j,:)
     & ,0.5*(z_r(i-1,j,:)+z_r(i,j,:))
     & ,0.5*(z_w(i-1,j,:)+z_w(i,j,:))
     & ,dpth,var3,1,0)

            CALL interp_1d(N,u(i,j-1,:)
     & ,0.5*(z_r(i,j-1,:)+z_r(i-1,j-1,:))
     & ,0.5*(z_w(i,j-1,:)+z_w(i-1,j-1,:))
     & ,dpth,var4,1,0)

          cff = 0.25*(f(i,j) + f(i-1,j) + f(i,j-1) + f(i-1,j-1))
          cff2 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
          cff3 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

            absvrt(i,j)= cff
     & + (var1-var2) * cff2
     & - (var3-var4) * cff3

         enddo
       enddo

!-------------------------------------------------------------------------
! COMPUTE VORTICITY
!-------------------------------------------------------------------------

       do i=imin+1,imax
        do j=jmin+1,jmax

            dpth=0.25*(z_r(i,j,iz+1)+z_r(i-1,j,iz+1)
     & + z_r(i-1,j-1,iz+1)+z_r(i,j-1,iz+1))

         do k=1,N-1
         work(i,j,k) = 2.*(v(i,j,k+1)+v(i-1,j,k+1)-v(i,j,k)-v(i-1,j,k))
     & / ( z_r(i,j,k+1)+z_r(i-1,j,k+1)-z_r(i,j,k)-z_r(i-1,j,k)
     & + z_r(i,j-1,k+1)+z_r(i-1,j-1,k+1)-z_r(i,j-1,k)-z_r(i-1,j-1,k))
         enddo

            CALL interp_1d(N,work(i,j,:)
     & ,0.25*(z_w(i,j,1:N-1)+z_w(i-1,j-1,1:N-1)
     & +z_w(i-1,j,1:N-1)+z_w(i,j-1,1:N-1))
     & ,0.25*(z_r(i,j,1:N)+z_r(i-1,j-1,1:N)
     & +z_r(i-1,j,1:N)+z_r(i,j-1,1:N))
     & ,dpth,vz(i,j),1,0)

         do k=1,N-1
         work(i,j,k) = 2.*(u(i,j,k+1)+u(i,j-1,k+1)-u(i,j,k)-u(i,j-1,k))
     & / ( z_r(i,j,k+1)+z_r(i-1,j,k+1)-z_r(i,j,k)-z_r(i-1,j,k)
     & + z_r(i,j-1,k+1)+z_r(i-1,j-1,k+1)-z_r(i,j-1,k)-z_r(i-1,j-1,k))
         enddo

            CALL interp_1d(N,work(i,j,:)
     & ,0.25*(z_w(i,j,1:N-1)+z_w(i-1,j-1,1:N-1)
     & +z_w(i-1,j,1:N-1)+z_w(i,j-1,1:N-1))
     & ,0.25*(z_r(i,j,1:N)+z_r(i-1,j-1,1:N)
     & +z_r(i-1,j,1:N)+z_r(i,j-1,1:N))
     & ,dpth,uz(i,j),1,0)

         enddo
       enddo

!-------------------------------------------------------------------------
! HEIGHT GRADIENT
!-------------------------------------------------------------------------

        do j=jmin+1,jmax
          do i=imin+1,imax

            dx(i,j) = 0.5* (z_r(i,j,iz+1) + z_r(i,j-1,iz+1)
     & - z_r(i-1,j,iz+1)- z_r(i-1,j-1,iz+1))
     & * 0.25*(pm(i,j)+pm(i,j-1)+ pm(i-1,j)+pm(i-1,j-1))

            dy(i,j) = 0.5* (z_r(i,j,iz+1) + z_r(i-1,j,iz+1)
     & - z_r(i,j-1,iz+1)- z_r(i-1,j-1,iz+1))
     & * 0.25*(pn(i,j)+pn(i,j-1)+ pn(i-1,j)+pn(i-1,j-1))

            cff = 1./sqrt(1 + dx(i,j)**2 + dy(i,j)**2)

            dx(i,j) = - cff * dx(i,j)
            dy(i,j) = - cff * dy(i,j)
            dz(i,j) = cff

          enddo
        enddo

!-------------------------------------------------------------------------
! COMPUTE VORTICITY
!-------------------------------------------------------------------------

        do j=jmin,jmax
          do i=imin,imax
              dbdt(i,j) = g*( alpha(i,j)*Tdiab(i,j)
     & - beta(i,j)*Sdiab(i,j) )
          enddo
        enddo

        do j=jmin+1,jmax
          do i=imin+1,imax

            Jdiab(i,j) = - 0.25 *
     & (dbdt(i,j)+dbdt(i-1,j)+dbdt(i,j-1)+dbdt(i-1,j-1))
     & * ( absvrt(i,j) * dz(i,j)
     & + uz(i,j) * dy(i,j)
     & - vz(i,j) * dx(i,j) )


          enddo
        enddo

      return
      end
# 69 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/old/get_J1_sol2.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute PV
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_J1_sol2(Lm,Mm,N, stflx,ssflx, u,v, z_r,z_w,rho0,pm,
     & pn,hbls,f,J1)

      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k
      real*8 stflx(0:Lm+1,0:Mm+1), ssflx(0:Lm+1,0:Mm+1),
     & u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & J1(1:Lm+1,1:Mm+1),f(0:Lm+1,0:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & hbls(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & dpth,cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00,
     & var1, var2,var3, var4,cff3,
     & absvrt(1:Lm+1,1:Mm+1),absvrt0(1:Lm+1,1:Mm+1)

      real*8, parameter :: g=9.81


Cf2py intent(in) Lm,Mm,N, stflx,ssflx, u,v,z_r,z_w,rho0,pm,pn,hbls,f
Cf2py intent(out) J1


      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE VORTICITY
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!---------------------------------------------------------------------------------------

       do i=imin+1,imax
        do j=jmin+1,jmax

            dpth=0.25*(z_r(i,j,N)+z_r(i-1,j,N)
     & + z_r(i-1,j-1,N)+z_r(i,j-1,N))

            CALL interp_1d(N,v(i,j,:)
     & ,0.5*(z_r(i,j,:)+z_r(i,j-1,:))
     & ,0.5*(z_w(i,j,:)+z_w(i,j-1,:))
     & ,dpth,var1,1,0)

            CALL interp_1d(N,v(i-1,j,:)
     & ,0.5*(z_r(i-1,j,:)+z_r(i-1,j-1,:))
     & ,0.5*(z_w(i-1,j,:)+z_w(i-1,j-1,:))
     & ,dpth,var2,1,0)


            CALL interp_1d(N,u(i,j,:)
     & ,0.5*(z_r(i-1,j,:)+z_r(i,j,:))
     & ,0.5*(z_w(i-1,j,:)+z_w(i,j,:))
     & ,dpth,var3,1,0)

            CALL interp_1d(N,u(i,j-1,:)
     & ,0.5*(z_r(i,j-1,:)+z_r(i-1,j-1,:))
     & ,0.5*(z_w(i,j-1,:)+z_w(i-1,j-1,:))
     & ,dpth,var4,1,0)

            cff = 0.25*(f(i,j) + f(i-1,j) + f(i,j-1) + f(i-1,j-1))
            cff2 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
            cff3 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

            absvrt(i,j)= cff
     & + (var1-var2) * cff2
     & - (var3-var4) * cff3


c absvrt0(i,j)= cff
c & + (v(i,j,N)-v(i-1,j,N)) * cff2
c & - (u(i,j,N)-u(i,j-1,N)) * cff3

c write(*,*) i,j,dpth,z_r(i,j,N),absvrt(i,j), absvrt0(i,j)


         enddo
       enddo


!---------------------------------------------------------------------------------------





!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        do j=jmin+1,jmax
          do i=imin+1,imax

                cff = g/((hbls(i,j) + hbls(i-1,j)
     & + hbls(i,j-1) + hbls(i-1,j-1)))



            J1(i,j) = cff * absvrt(i,j) * (
     & (stflx(i,j)+stflx(i-1,j)+stflx(i,j-1)+stflx(i-1,j-1))
     & - (ssflx(i,j)+ssflx(i-1,j)+ssflx(i,j-1)+ssflx(i-1,j-1))
     & )


          enddo
        enddo


      return
      end
# 71 "R_tools_fort_gula.F" 2
# 1 "R_tools_fort_routines_gula/old/get_J2_sol2.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute PV
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_J2_sol2(Lm,Mm,N, T,S, u,v, z_r,z_w,rho0,pm,pn,
     & hbls,J2)


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k
      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & J2(1:Lm+1,1:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & hbls(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & rho1(0:Lm+1,0:Mm+1,N),
     & drdx(1:Lm+1,0:Mm+1), drdy(0:Lm+1,1:Mm+1),
     & qp1(0:Lm+1,0:Mm+1,N),
     & dpth,cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00,
     & dvdx, dudy,zlev,
     & cffi(0:Lm+1), cffj(0:Mm+1),
     & var1, var2,var3, var4

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


Cf2py intent(in) Lm,Mm,N, T,S, u,v,z_r,z_w,rho0,pm,pn,hbls
Cf2py intent(out) J2
# 60 "R_tools_fort_routines_gula/old/get_J2_sol2.F"
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE NEUTRAL DENSITY GRADIENTS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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


!---------------------------------------------------------------------------------------
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
      enddo ! <-- j



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE BUOYANCY GRADIENTS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        cff=g/rho0


!---------------------------------------------------------------------------------------

       do i=imin+1,imax

        do j=jmin,jmax


            zlev=0.5*(z_r(i,j,N)+z_r(i-1,j,N))
            dpth=0.5*(z_w(i,j,N)+z_w(i-1,j,N)) - zlev

            !if (dpth.gt.z_w(i,j,N)) then
            ! write(*,*) 'calling', dpth, z_r(i,j,N), z_w(i,j,N)
            !endif

            !if (dpth.gt.z_w(i-1,j,N)) then
            ! write(*,*) 'calling 2', dpth, z_r(i,j,N), z_w(i,j,N)
            !endif

            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,zlev,var1,1,0)

            CALL interp_1d(N,rho1(i-1,j,:),z_r(i-1,j,:),z_w(i-1,j,:)
     & ,zlev,var2,1,0)



            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,zlev,var3,1,0)
            CALL interp_1d(N,qp1(i-1,j,:),z_r(i-1,j,:),z_w(i-1,j,:)
     & ,zlev,var4,1,0)

            cffj(j)=-cff*( var1 - var2 ! Elementary
     & +(var3-var4) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference
     & *0.5*(pm(i,j)+pm(i-1,j))
         enddo



         do j=jmin+1,jmax

            drdx(i,j)= 0.5 * (cffj(j) + cffj(j-1))
            !!write(*,*) 'all3',dpth,cffj(j),cffj(j-1), drdx(i,j,k)

         enddo

         !write(*,*) 'all2',i,j,drdx(i,j)


       enddo


!---------------------------------------------------------------------------------------

        !write(*,*) 'bouble'

        cff=g/rho0

        do j=jmin+1,jmax

          do i=imin,imax

            zlev=0.5*(z_r(i,j,N)+z_r(i,j-1,N))
            dpth=0.5*(z_w(i,j,N)+z_w(i-1,j,N)) - zlev


            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,zlev,var1,1,0)
            CALL interp_1d(N,rho1(i,j-1,:),z_r(i,j-1,:),z_w(i,j-1,:)
     & ,zlev,var2,1,0)

            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,zlev,var3,1,0)
            CALL interp_1d(N,qp1(i,j-1,:),z_r(i,j-1,:),z_w(i,j-1,:)
     & ,zlev,var4,1,0)




            cffi(i)=-cff*( var1 - var2 ! Elementary
     & +(var3-var4) ! adiabatic
     & *dpth*(1.-qp2*dpth) )
     & *0.5*(pn(i,j)+pn(i,j-1))


         !write(*,*) 'all2',i,j,dpth,cffi(i),var1,var2

          enddo



          do i=imin+1,imax

            drdy(i,j)= 0.5 * (cffi(i) + cffi(i-1))


          enddo



        enddo





!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        do j=jmin+1,jmax
          do i=imin+1,imax

            cff = 0.5/(rho0*0.25*(hbls(i,j) + hbls(i-1,j)
     & + hbls(i,j-1) + hbls(i-1,j-1)))

            J2(i,j) = cff * (drdy(i,j) * (u(i,j-1) + u(i,j))
     & - drdx(i,j) * (v(i-1,j)+v(i,j) ))

            !!write(*,*) 'all3',i,j,cff,drdy(i,j),u(i,j-1),J2(i,j)
          enddo
        enddo






      return
      end
# 72 "R_tools_fort_gula.F" 2
# 1 "R_tools_fort_routines_gula/old/get_Jbot_sol2.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute PV
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_Jbot_sol2(Lm,Mm,N, T,S, u,v, z_r,z_w,rho0,pm,pn,
     & hbbls,rdrg,Jbot)


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k
      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & Jbot(1:Lm+1,1:Mm+1),
     & dx(1:Lm+1,1:Mm+1),
     & dy(1:Lm+1,1:Mm+1),
     & dz(1:Lm+1,1:Mm+1),
     & rd(0:Lm+1,0:Mm+1),
     & ubot(1:Lm+1,1:Mm+1),vbot(1:Lm+1,1:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & hbbls(0:Lm+1,0:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1),
     & rho1(0:Lm+1,0:Mm+1,N), drdz(0:Lm+1,0:Mm+1),
     & drdx(1:Lm+1,0:Mm+1), drdy(0:Lm+1,1:Mm+1),
     & qp1(0:Lm+1,0:Mm+1,N),
     & dpth,cff,Tt,Ts,sqrtTs, rho0, K0, dr00,
     & cff2,cff3,zlev,
     & dvdx, dudy,
     & cffi(0:Lm+1), cffj(0:Mm+1),
     & var1, var2,var3, var4


      real*8 Zob, rdrg

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
     & qp2=0.0000172


      real rho1_0, K0_Duk


# 1 "R_tools_fort_routines_gula/old/scalars.h" 1
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
# 57 "R_tools_fort_routines_gula/old/get_Jbot_sol2.F" 2

Cf2py intent(in) Lm,Mm,N, T,S, u,v,z_r,z_w,rho0,pm,pn,hbbls,rdrg
Cf2py intent(out) Jbot
# 69 "R_tools_fort_routines_gula/old/get_Jbot_sol2.F"
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE NEUTRAL DENSITY GRADIENTS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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


!---------------------------------------------------------------------------------------
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
      enddo ! <-- j



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE BUOYANCY GRADIENTS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        cff=g/rho0



!---------------------------------------------------------------------------------------




      do j=jmin,jmax
          do i=imin,imax

            dpth=z_w(i,j,N)-z_r(i,j,1)


            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,0),var1,1,0)
            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,1),var2,1,0)

            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,0),var3,1,0)
            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,z_w(i,j,1),var4,1,0)

            cff2=( var2-var1 ! Elementary
     & +(var4-var3) ! adiabatic
     & *dpth*(1.-2.*qp2*dpth) ! difference
     & )


            drdz(i,j) =- cff*cff2 / (z_w(i,j,1)-z_w(i,j,0))



          enddo
      enddo ! <-- j






!---------------------------------------------------------------------------------------

       do i=imin+1,imax
        do j=jmin,jmax

            zlev=0.5*(z_r(i,j,1)+z_r(i-1,j,1))

            dpth=0.5*(z_w(i,j,N)+z_w(i-1,j,N)) - zlev


            !if ((z_r(i,j,N).ne.0).and.(z_r(i-1,j,N).ne.0)) then
            !if (dpth.gt.z_w(i,j,N)) then
             !write(*,*) 'calling', dpth, z_r(i,j,N), z_w(i,j,N)
             !write(*,*) z_w(i,j,:)
             !write(*,*) z_w(i-1,j,:)
            !endif
            !endif

            !if (dpth.gt.z_w(i-1,j,N)) then
            ! write(*,*) 'calling 2', dpth, z_r(i,j,N), z_w(i,j,N)
            !endif


            !if ((z_r(i,j,N).ne.0).and.(z_r(i-1,j,N).ne.0)) then


            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,zlev,var1,1,0)
            CALL interp_1d(N,rho1(i-1,j,:),z_r(i-1,j,:),z_w(i-1,j,:)
     & ,zlev,var2,1,0)


            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,zlev,var3,1,0)
            CALL interp_1d(N,qp1(i-1,j,:),z_r(i-1,j,:),z_w(i-1,j,:)
     & ,zlev,var4,1,0)

            cffj(j)=-cff*( var1 - var2 ! Elementary
     & +(var3-var4) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference
     & *0.5*(pm(i,j)+pm(i-1,j))

            !else

            !cffj(j)= 9999.

            !endif

         enddo



         do j=jmin+1,jmax

            drdx(i,j)= 0.5 * (cffj(j) + cffj(j-1))
            !!write(*,*) 'all3',dpth,cffj(j),cffj(j-1), drdx(i,j,k)

         enddo

         !!write(*,*) 'all2',i,j,drdx(i,j)


       enddo


!---------------------------------------------------------------------------------------

        do j=jmin+1,jmax

          do i=imin,imax

            zlev=0.5*(z_r(i,j,1)+z_r(i,j-1,1))
            dpth=0.5*(z_w(i,j,N)+z_w(i-1,j,N)) - zlev

            !write(*,*) 'all2',z_w(i,j,N),z_r(i,j,1),i,j

            CALL interp_1d(N,rho1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,zlev,var1,1,0)
            CALL interp_1d(N,rho1(i,j-1,:),z_r(i,j-1,:),z_w(i,j-1,:)
     & ,zlev,var2,1,0)

            CALL interp_1d(N,qp1(i,j,:),z_r(i,j,:),z_w(i,j,:)
     & ,zlev,var3,1,0)
            CALL interp_1d(N,qp1(i,j-1,:),z_r(i,j-1,:),z_w(i,j-1,:)
     & ,zlev,var4,1,0)




            cffi(i)=-cff*( var1 - var2 ! Elementary
     & +(var3-var4) ! adiabatic
     & *dpth*(1.-qp2*dpth) )
     & *0.5*(pn(i,j)+pn(i,j-1))


         !!!write(*,*) 'all2',i,j,dpth,cffi(i),var1,var2

          enddo



          do i=imin+1,imax

            drdy(i,j)= 0.5 * (cffi(i) + cffi(i-1))


          enddo



        enddo




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Bottom Drag
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

       Zob=0.01


       do j=jmin+1,jmax-1
         do i=imin+1,imax-1

            Hz(i,j) = z_w(i,j,1) - z_w(i,j,0)




            cff=sqrt( 0.333333333333*(
     & u(i,j)**2 +u(i+1,j)**2
     & +u(i,j)*u(i+1,j)
     & +v(i,j)**2+v(i,j+1)**2
     & +v(i,j)*v(i,j+1)
     & ))
            rd(i,j)=rdrg + cff*(vonKar/log(Hz(i,j)/Zob))**2




          enddo
        enddo

       do j=jmin+1,jmax
            rd(imax ,j)=rd(imax-1 ,j)
            rd(imin ,j)=rd(imin+1 ,j)
        enddo

       do i=imin+1,imax
            rd(i ,jmax)=rd(i ,jmax-1)
            rd(i ,jmin)=rd(i ,jmin+1)
        enddo







!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! HEIGHT GRADIENT
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        do j=jmin+1,jmax
          do i=imin+1,imax

            dx(i,j) = 0.5* (z_r(i,j,1) + z_r(i,j-1,1)
     & - z_r(i-1,j,1)- z_r(i-1,j-1,1))
     & * 0.25*(pm(i,j)+pm(i,j-1)+ pm(i+1,j)+pm(i-1,j-1))

            dy(i,j) = 0.5* (z_r(i,j,1) + z_r(i-1,j,1)
     & - z_r(i,j-1,1)- z_r(i-1,j-1,1))
     & * 0.25*(pn(i,j)+pn(i,j-1)+ pn(i+1,j)+pn(i-1,j-1))


            cff = 1./sqrt(1 + dx(i,j)**2 + dy(i,j)**2)

            dx(i,j) = cff * dx(i,j)
            dy(i,j) = cff * dy(i,j)
            dz(i,j) = cff

            !if (cff.ne.1) then
            ! write(*,*) i,j,dx(i,j), dy(i,j), dz(i,j)
            !endif

          enddo
        enddo




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        do j=jmin+1,jmax
          do i=imin+1,imax


            cff2 = 0.25 * (drdz(i-1,j) + drdz(i,j)
     & + drdz(i-1,j-1) + drdz(i,j-1))
            cff3 =0.25 * (rd(i,j)+rd(i-1,j)
     & + rd(i,j-1)+rd(i-1,j-1))


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            cff = 1./(0.25*(hbbls(i,j) + hbbls(i-1,j)
     & + hbbls(i,j-1) + hbbls(i-1,j-1)))

            ubot(i,j) = cff3*0.5*(u(i,j-1) + u(i,j))

            vbot(i,j) = cff3*0.5*(v(i-1,j)+v(i,j) )

            Jbot(i,j) = cff * ((drdx(i,j) * vbot(i,j)
     & - drdy(i,j) * ubot(i,j)) * dz(i,j)
     & - (cff2 * vbot(i,j)) * dx(i,j)
     & + (cff2 * ubot(i,j)) * dy(i,j)
     & )



          enddo
        enddo






      return
      end
# 73 "R_tools_fort_gula.F" 2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute Barotropic equation components
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# 1 "R_tools_fort_routines_gula/get_bot.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute bottom drag
!!
!! - 16/08/17 : Define Zob as input instead of fixed Zob = 0.01 value
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_bot(Lm,Mm, u,v, Hz
     &,rdrg,ubot,vbot,Zob)


      implicit none
      integer Lm,Mm, imin,imax,jmin,jmax, i,j,k
      real*8 ubot(1:Lm+1,0:Mm+1), vbot(0:Lm+1,1:Mm+1),
     & u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & Hz(0:Lm+1,0:Mm+1), rd(0:Lm+1,0:Mm+1),
     & cff

      real*8 rdrg, Zob

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
# 25 "R_tools_fort_routines_gula/get_bot.F" 2


Cf2py intent(in) Lm,Mm,u,v, Hz,rdrg,Zob
Cf2py intent(out) ubot,vbot

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Bottom Drag
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


       do j=jmin+1,jmax-1
         do i=imin+1,imax-1





            cff=sqrt( 0.333333333333*(
     & u(i,j)**2 +u(i+1,j)**2
     & +u(i,j)*u(i+1,j)
     & +v(i,j)**2+v(i,j+1)**2
     & +v(i,j)*v(i,j+1)
     & ))
            rd(i,j)=rdrg + cff*(vonKar/log(Hz(i,j)/Zob))**2




          enddo
        enddo

       do j=jmin+1,jmax
            rd(imax ,j)=rd(imax-1 ,j)
            rd(imin ,j)=rd(imin+1 ,j)
        enddo

       do i=imin+1,imax
            rd(i ,jmax)=rd(i ,jmax-1)
            rd(i ,jmin)=rd(i ,jmin+1)
        enddo




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




        do j=jmin,jmax
          do i=imin+1,imax

            ubot(i,j) = 0.5 * (rd(i,j)+rd(i-1,j))*u(i,j)

        enddo !<- i
      enddo !<- j
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




        do j=jmin+1,jmax
          do i=imin,imax


            vbot(i,j) = 0.5 * (rd(i,j)+rd(i,j-1))*v(i,j)

        enddo !<- i
      enddo !<- j





      return
      end
# 81 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_bot_croco.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute bottom drag [for croco version]
!!
!! - modified 18/01/03
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      subroutine get_bot_croco(Lm,Mm, u,v, Hz, Hz2
     &,rdrg,rdrg2,Zob,Cdb_min,Cdb_max,dt,bustr,bvstr)

      implicit none
      integer Lm,Mm, imin,imax,jmin,jmax, i,j,k
      integer Istr,Iend,Jstr,Jend,IstrU,JstrV

      real*8 bustr(1:Lm+1,0:Mm+1), bvstr(0:Lm+1,1:Mm+1),
     & u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & Hz(0:Lm+1,0:Mm+1), wrk(0:Lm+1,0:Mm+1),
     & Hz2(0:Lm+1,0:Mm+1),
     & cff, cff1

      real*8 rdrg, rdrg2, Zob, Cdb_min, Cdb_max,dt

      !parameter (rdrg2 = 0.0, Cdb_max = 0.1, Cdb_min = 0.0001)


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
# 32 "R_tools_fort_routines_gula/get_bot_croco.F" 2


Cf2py intent(in) Lm,Mm,u,v, Hz,Hz2,rdrg,rdrg2,Zob,Cdb_min,Cdb_max,dt
Cf2py intent(out) bustr,bvstr

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

      Istr = imin
      Iend = imax
      Jstr = jmin
      Jend = jmax

      IstrU = imin+1
      JstrV = jmin+1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Bottom Drag
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!
! Set bottom stress using logarithmic or linear
! and/or quadratic formulation.
!
      if (Zob.ne.0.) then
        do j=JstrV-1,Jend
          do i=IstrU-1,Iend
            cff=vonKar/LOG((Hz2(i,j))/Zob)
            wrk(i,j)=MIN(Cdb_max,MAX(Cdb_min,cff*cff))
          enddo
        enddo
        do j=Jstr,Jend
          do i=IstrU,Iend
            cff=0.25*(v(i ,j)+v(i ,j+1)+
     & v(i-1,j)+v(i-1,j+1))
            bustr(i,j)=0.5*(wrk(i-1,j)+wrk(i,j))*u(i,j)*
     & SQRT(u(i,j)*u(i,j)+cff*cff)
          enddo
        enddo
        do j=JstrV,Jend
          do i=Istr,Iend
            cff=0.25*(u(i,j )+u(i+1,j)+
     & u(i,j-1)+u(i+1,j-1))
            bvstr(i,j)=0.5*(wrk(i,j-1)+wrk(i,j))*v(i,j)*
     & SQRT(cff*cff+v(i,j)*v(i,j))
          enddo
        enddo
      elseif (rdrg2.gt.0.) then
        do j=JstrV,Jend
          do i=Istr,Iend

            cff=0.25*(v(i,j)+v(i,j+1)+v(i-1,j)+
     & v(i-1,j+1))
            bustr(i,j)=u(i,j)*(rdrg2*sqrt(
     & u(i,j)*u(i,j)+cff*cff
     & ))
          enddo
        enddo
        do j=Jstr,Jend
          do i=IstrU,Iend
            cff=0.25*(u(i,j)+u(i+1,j)+u(i,j-1)+
     & u(i+1,j-1))
            bvstr(i,j)=v(i,j)*(rdrg2*sqrt(
     & cff*cff+v(i,j)*v(i,j)
     & ))
          enddo
        enddo
      else
        do j=Jstr,Jend
          do i=Istr,Iend
            bustr(i,j)=rdrg*u(i,j)
          enddo
        enddo
        do j=Jstr,Jend
          do i=Istr,Iend
            bvstr(i,j)=rdrg*v(i,j)
          enddo
        enddo
      endif


!
! From J. Warner code:
! Set limiting factor for bottom stress. The bottom stress is adjusted
! to not change the direction of momentum. It only should slow down
! to zero. The value of 0.75 is arbitrary limitation assigment.
!
      cff=0.75/dt
      do j=Jstr,Jend
        do i=IstrU,Iend
          cff1=cff*0.5*(Hz(i-1,j)+Hz(i,j))
          bustr(i,j)=SIGN(1.D0, bustr(i,j))*
     & MIN(ABS(bustr(i,j)),
     & ABS(u(i,j))*cff1)
        enddo
      enddo
      do j=JstrV,Jend
        do i=Istr,Iend
          cff1=cff*0.5*(Hz(i,j-1)+Hz(i,j))
          bvstr(i,j)=SIGN(1.D0, bvstr(i,j))*
     & MIN(ABS(bvstr(i,j)),
     & ABS(v(i,j))*cff1)
        enddo
      enddo



      return
      end
# 83 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_bpt.F" 1

!======================================================================
!
! Compute Bottom Pressure torque
!
! = J(Pb,H)
!
!
! - updated 16/08/19 [add umask,vmask]
!======================================================================



      subroutine get_bpt (Lm,Mm,N,T,S, z_r,z_w,rho0,pm,pn,rmask
     & ,bpt)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV



      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & rho1(0:Lm+1,0:Mm+1,N), qp1(0:Lm+1,0:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,0:N),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & rmask(0:Lm+1,0:Mm+1),
     & Tt,Ts,sqrtTs, rho0, K0, dr00,
     & cff ,cff1, cff2, cfr, HalfGRho, GRho,
     & var1, var2,var3, var4

      real*8 bpt(1:Lm+1,1:Mm+1)


      real*8 P(0:Lm+1,0:Mm+1,N),
     & ru(1:Lm+1,0:Mm+1), rv(0:Lm+1,1:Mm+1),
     & rho(0:Lm+1,0:Mm+1,N), dpth,
     & dR(0:Lm+1,0:N), dZ(0:Lm+1,0:N),
     & FC(0:Lm+2,0:Mm+2), dZx(0:Lm+1,0:Mm+1),
     & rx(0:Lm+2,0:Mm+2), dRx(0:Lm+1,0:Mm+1)

      real*8, parameter :: OneFifth=0.2, OneTwelfth=1./12., epsil=0.

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
     & qp2=0.0000172


      real rho1_0, K0_Duk

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
# 65 "R_tools_fort_routines_gula/get_bpt.F" 2



Cf2py intent(in) Lm,Mm,N, T,S,z_r,z_w,rho0,pm,pn,rmask
Cf2py intent(out) bpt


!
! A non-conservative Density-Jacobian scheme using cubic polynomial
! fits for rho and z_r as functions of nondimensianal coordinates xi,
! eta, and s (basically their respective fortran indices). The cubic
! polynomials are constructed by specifying first derivatives of
! interpolated fields on co-located (non-staggered) grid. These
! derivatives are computed using harmonic (rather that algebraic)
! averaging of elementary differences, which guarantees monotonicity
! of the resultant interpolant.
!
! In the code below, if CPP-switch is defined, the Equation
! of State (EOS) is assumed to have form
!
! rho(T,S,z) = rho1(T,S) + qp1(T,S)*dpth*[1.-qp2*dpth]
!
! where rho1 is potential density at 1 atm and qp1 is compressibility
! coefficient, which does not depend on z, and dpth=zeta-z, and qp2
! is just a constant. In this case
!
! d rho d rho1 d qp1 d z
! ------- = ------ + ----- *dpth*[..] - qp1*[1.-2.*qp2*dpth]*------
! d s,x d s,x d s,x d s,x
!
! |<--- adiabatic part --->| |<--- compressible part --->|
!
! where the first two terms constitute "adiabatic derivative" of
! density, which is subject to harmonic averaging, while the last
! term is added in later. This approach quarantees that density
! profile reconstructed by cubic polynomial maintains its positive
! statification in physical sense as long as discrete values of
! density are positively stratified.
!
! This scheme retains exact antisymmetry J(rho,z_r)=-J(z_r,rho)
! [with the exception of harmonic averaging algorithm in the case
! when CPP-switch is defined, see above]. If parameter
! OneFifth (see above) is set to zero, the scheme becomes identical
! to standard Jacobian.
!
! NOTE: This routine is an alternative form of prsgrd32 and it
! produces results identical to that if its prototype.
!


        istr=0
        istrU=1
        iend=Lm+1
        jstr=0
        jstrV=1
        jend=Mm+1

        imin=istrU
        imax=iend
        jmin=jstrV
        jmax=jend




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE DENSITY
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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





!---------------------------------------------------------------------------------------


      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

      do j=jmin,jmax


!---------------------------------------------------------------------------------------
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


!---------------------------------------------------------------------------------------

            Hz(i,j,k)=z_w(i,j,k)-z_w(i,j,k-1)




          enddo
        enddo
      enddo ! <-- j



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!
! Preliminary step (same for XI- and ETA-components:
!------------ ---- ----- --- --- --- ---------------
!
      GRho=g/rho0
      HalfGRho=0.5*GRho

      do j=jstrV-1,jend
        do k=1,N-1
          do i=istrU-1,iend
            dZ(i,k)=z_r(i,j,k+1)-z_r(i,j,k)

            dpth=z_w(i,j,N)-0.5*(z_r(i,j,k+1)+z_r(i,j,k))

            dR(i,k)=rho1(i,j,k+1)-rho1(i,j,k) ! Elementary
     & +(qp1(i,j,k+1)-qp1(i,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ! difference



          enddo
        enddo
        do i=istrU-1,iend
          dR(i,N)=dR(i,N-1)
          dR(i,0)=dR(i,1)
          dZ(i,N)=dZ(i,N-1)
          dZ(i,0)=dZ(i,1)
        enddo
        do k=N,1,-1 !--> irreversible
          do i=istrU-1,iend
            cff=2.*dZ(i,k)*dZ(i,k-1)
            dZ(i,k)=cff/(dZ(i,k)+dZ(i,k-1))

            cfr=2.*dR(i,k)*dR(i,k-1)
            if (cfr.gt.epsil) then
              dR(i,k)=cfr/(dR(i,k)+dR(i,k-1))
            else
              dR(i,k)=0.
            endif

            dpth=z_w(i,j,N)-z_r(i,j,k)
            dR(i,k)=dR(i,k) -qp1(i,j,k)*dZ(i,k)*(1.-2.*qp2*dpth)
            rho(i,j,k)=rho1(i,j,k) +qp1(i,j,k)*dpth*(1.-qp2*dpth)

          enddo
        enddo
        do i=istrU-1,iend
          P(i,j,N)=g*z_w(i,j,N) + GRho*( rho(i,j,N)
     & +0.5*(rho(i,j,N)-rho(i,j,N-1))*(z_w(i,j,N)-z_r(i,j,N))
     & /(z_r(i,j,N)-z_r(i,j,N-1)) )*(z_w(i,j,N)-z_r(i,j,N))
        enddo
        do k=N-1,1,-1
          do i=istrU-1,iend
            P(i,j,k)=P(i,j,k+1)+HalfGRho*( (rho(i,j,k+1)+rho(i,j,k))
     & *(z_r(i,j,k+1)-z_r(i,j,k))

     & -OneFifth*( (dR(i,k+1)-dR(i,k))*( z_r(i,j,k+1)-z_r(i,j,k)
     & -OneTwelfth*(dZ(i,k+1)+dZ(i,k)) )

     & -(dZ(i,k+1)-dZ(i,k))*( rho(i,j,k+1)-rho(i,j,k)
     & -OneTwelfth*(dR(i,k+1)+dR(i,k)) )
     & ))
          enddo
        enddo
      enddo !<-- j




        ru = 0.
        rv = 0.



!
! Compute XI-component of pressure gradient term:
!-------- ------------ -- -------- -------- -----
!
      do k=N,1,-1
        do j=jstr,jend
          do i=imin,imax
            FC(i,j)=(z_r(i,j,k)-z_r(i-1,j,k))

            dpth=0.5*( z_w(i,j,N)+z_w(i-1,j,N)
     & -z_r(i,j,k)-z_r(i-1,j,k))

            rx(i,j)=( rho1(i,j,k)-rho1(i-1,j,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i-1,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference




          enddo
        enddo


        if (istr.eq.1) then ! Extrapolate elementary
          do j=jstr,jend ! differences near physical
            FC(imin-1,j)=FC(imin,j) ! boundaries to compencate.
            rx(imin-1,j)=rx(imin,j) ! for reduced loop ranges.
          enddo
        endif
        if (iend.eq.Lm) then
          do j=jstr,jend
            FC(imax+1,j)=FC(imax,j)
            rx(imax+1,j)=rx(imax,j)
          enddo
        endif


        do j=jstr,jend
          do i=istrU-1,iend
            cff=2.*FC(i,j)*FC(i+1,j)
            if (cff.gt.epsil) then
              dZx(i,j)=cff/(FC(i,j)+FC(i+1,j))
            else
              dZx(i,j)=0.
            endif

            cfr=2.*rx(i,j)*rx(i+1,j)
            if (cfr.gt.epsil) then
              dRx(i,j)=cfr/(rx(i,j)+rx(i+1,j))
            else
              dRx(i,j)=0.
            endif

            dRx(i,j)=dRx(i,j) -qp1(i,j,k)*dZx(i,j)
     & *(1.-2.*qp2*(z_w(i,j,N)-z_r(i,j,k)))

          enddo !--> discard FC, rx

          do i=istrU,iend
            ru(i,j)=ru(i,j) +
     & 0.5*(Hz(i,j,k)+Hz(i-1,j,k))*2./(pn(i,j)+pn(i-1,j))*(
     & P(i-1,j,k)-P(i,j,k)-HalfGRho*(

     & (rho(i,j,k)+rho(i-1,j,k))*(z_r(i,j,k)-z_r(i-1,j,k))

     & -OneFifth*( (dRx(i,j)-dRx(i-1,j))*( z_r(i,j,k)-z_r(i-1,j,k)
     & -OneTwelfth*(dZx(i,j)+dZx(i-1,j)) )

     & -(dZx(i,j)-dZx(i-1,j))*( rho(i,j,k)-rho(i-1,j,k)
     & -OneTwelfth*(dRx(i,j)+dRx(i-1,j)) )
     & )))
          enddo
        enddo
!
! ETA-component of pressure gradient term:
!-------------- -- -------- -------- -----
!
        do j=jmin,jmax
          do i=istr,iend
            FC(i,j)=(z_r(i,j,k)-z_r(i,j-1,k))

            dpth=0.5*( z_w(i,j,N)+z_w(i,j-1,N)
     & -z_r(i,j,k)-z_r(i,j-1,k))

            rx(i,j)=( rho1(i,j,k)-rho1(i,j-1,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i,j-1,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference



          enddo
        enddo


        if (jstr.eq.1) then
          do i=istr,iend
            FC(i,jmin-1)=FC(i,jmin)
            rx(i,jmin-1)=rx(i,jmin)

          enddo
        endif
        if (jend.eq.Mm) then
          do i=istr,iend
            FC(i,jmax+1)=FC(i,jmax)
            rx(i,jmax+1)=rx(i,jmax)
          enddo
        endif


        do j=jstrV-1,jend
          do i=istr,iend
            cff=2.*FC(i,j)*FC(i,j+1)
            if (cff.gt.epsil) then
c** if ((FC(i,j).gt.0. .and. FC(i,j+1).gt.0.) .or.
c** & (FC(i,j).lt.0. .and. FC(i,j+1).lt.0.)) then
              dZx(i,j)=cff/(FC(i,j)+FC(i,j+1))
            else
              dZx(i,j)=0.
            endif

            cfr=2.*rx(i,j)*rx(i,j+1)
            if (cfr.gt.epsil) then
c** if ((rx(i,j).gt.0. .and. rx(i,j+1).gt.0.) .or.
c** & (rx(i,j).lt.0. .and. rx(i,j+1).lt.0.)) then
              dRx(i,j)=cfr/(rx(i,j)+rx(i,j+1))
            else
              dRx(i,j)=0.
            endif

            dRx(i,j)=dRx(i,j) -qp1(i,j,k)*dZx(i,j)
     & *(1.-2.*qp2*(z_w(i,j,N)-z_r(i,j,k)))

          enddo !--> discard FC, rx

          if (j.ge.jstrV) then
            do i=istr,iend

            rv(i,j)=rv(i,j) +
     & 0.5*(Hz(i,j,k)+Hz(i,j-1,k))*2./(pm(i,j)+pm(i,j-1))*(
     & P(i,j-1,k)-P(i,j,k) -HalfGRho*(

     & (rho(i,j,k)+rho(i,j-1,k))*(z_r(i,j,k)-z_r(i,j-1,k))

     & -OneFifth*( (dRx(i,j)-dRx(i,j-1))*( z_r(i,j,k)-z_r(i,j-1,k)
     & -OneTwelfth*(dZx(i,j)+dZx(i,j-1)) )

     & -(dZx(i,j)-dZx(i,j-1))*( rho(i,j,k)-rho(i,j-1,k)
     & -OneTwelfth*(dRx(i,j)+dRx(i,j-1)) )
     & )))

            enddo
          endif
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Compute rotational
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!
!
! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
!
!
! ru(i,j) = ru(i,j)*0.5*(pn(i,j)+pn(i-1,j))
! & *0.5*(pm(i,j)+pm(i-1,j))
!
!
! rv(i,j) = rv(i,j)*0.5*(pm(i,j)+pm(i,j-1))
! & *0.5*(pn(i,j)+pn(i,j-1))
!
!
!
! cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
! cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
!
! bpt(i,j) = (rv(i,j) - rv(i-1,j)) * cff1
! & - (ru(i,j) - ru(i,j-1)) * cff2
! enddo
! enddo
!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/09/13)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin,jmax
        do i=imin+1,imax

            cff=0.5*(pn(i,j)+pn(i-1,j)) * rmask(i,j)*rmask(i-1,j)
            ru(i,j)=ru(i,j)*cff

        enddo
      enddo

      do j=jmin+1,jmax
        do i=imin,imax

            cff=0.5*(pm(i,j)+pm(i,j-1)) * rmask(i,j)*rmask(i,j-1)
            rv(i,j)=rv(i,j)*cff

        enddo
      enddo

      do j=jmin+1,jmax
        do i=imin+1,imax

           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))


             bpt(i,j) = (rv(i,j) - rv(i-1,j)) * cff1
     & - (ru(i,j) - ru(i,j-1)) * cff1

        enddo
       enddo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






      return
      end
# 85 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_u_prsgrd.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute Bottom Pressure torque J(Pb,H)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






      subroutine get_u_prsgrd (Lm,Mm,N,T,S, z_r,z_w,rho0,pm,pn
     & ,ru)



      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV



      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & rho1(0:Lm+1,0:Mm+1,N), qp1(0:Lm+1,0:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,0:N),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & Tt,Ts,sqrtTs, rho0, K0, dr00,
     & cff ,cff1, cff2, cfr, HalfGRho, GRho,
     & var1, var2,var3, var4

      real*8 bpt(1:Lm+1,1:Mm+1)


      real*8 P(0:Lm+1,0:Mm+1,N),
     & ru(1:Lm+1,0:Mm+1,N), rv(0:Lm+1,1:Mm+1),
     & rho(0:Lm+1,0:Mm+1,N), dpth,
     & dR(0:Lm+1,N), dZ(0:Lm+1,N),
     & FC(0:Lm+2,0:Mm+2), dZx(0:Lm+1,0:Mm+1),
     & rx(0:Lm+2,0:Mm+2), dRx(0:Lm+1,0:Mm+1)

      real*8, parameter :: OneFifth=0.2, OneTwelfth=1./12., epsil=0.

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
     & qp2=0.0000172


      real rho1_0, K0_Duk

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
# 63 "R_tools_fort_routines_gula/get_u_prsgrd.F" 2



Cf2py intent(in) Lm,Mm,N, T,S,z_r,z_w,rho0,pm,pn
Cf2py intent(out) ru




!
! A non-conservative Density-Jacobian scheme using cubic polynomial
! fits for rho and z_r as functions of nondimensianal coordinates xi,
! eta, and s (basically their respective fortran indices). The cubic
! polynomials are constructed by specifying first derivatives of
! interpolated fields on co-located (non-staggered) grid. These
! derivatives are computed using harmonic (rather that algebraic)
! averaging of elementary differences, which guarantees monotonicity
! of the resultant interpolant.
!
! In the code below, if CPP-switch is defined, the Equation
! of State (EOS) is assumed to have form
!
! rho(T,S,z) = rho1(T,S) + qp1(T,S)*dpth*[1.-qp2*dpth]
!
! where rho1 is potential density at 1 atm and qp1 is compressibility
! coefficient, which does not depend on z, and dpth=zeta-z, and qp2
! is just a constant. In this case
!
! d rho d rho1 d qp1 d z
! ------- = ------ + ----- *dpth*[..] - qp1*[1.-2.*qp2*dpth]*------
! d s,x d s,x d s,x d s,x
!
! |<--- adiabatic part --->| |<--- compressible part --->|
!
! where the first two terms constitute "adiabatic derivative" of
! density, which is subject to harmonic averaging, while the last
! term is added in later. This approach quarantees that density
! profile reconstructed by cubic polynomial maintains its positive
! statification in physical sense as long as discrete values of
! density are positively stratified.
!
! This scheme retains exact antisymmetry J(rho,z_r)=-J(z_r,rho)
! [with the exception of harmonic averaging algorithm in the case
! when CPP-switch is defined, see above]. If parameter
! OneFifth (see above) is set to zero, the scheme becomes identical
! to standard Jacobian.
!
! NOTE: This routine is an alternative form of prsgrd32 and it
! produces results identical to that if its prototype.
!


        istr=0
        istrU=1
        iend=Lm+1
        jstr=0
        jstrV=1
        jend=Mm+1

        imin=istrU
        imax=iend
        jmin=jstrV
        jmax=jend


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE DENSITY
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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


!---------------------------------------------------------------------------------------
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


!---------------------------------------------------------------------------------------

            Hz(i,j,k)=z_w(i,j,k)-z_w(i,j,k-1)




          enddo
        enddo
      enddo ! <-- j



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!
! Preliminary step (same for XI- and ETA-components:
!------------ ---- ----- --- --- --- ---------------
!
      GRho=g/rho0
      HalfGRho=0.5*GRho

      do j=jstrV-1,jend
        do k=1,N-1
          do i=istrU-1,iend
            dZ(i,k)=z_r(i,j,k+1)-z_r(i,j,k)

            dpth=z_w(i,j,N)-0.5*(z_r(i,j,k+1)+z_r(i,j,k))

            dR(i,k)=rho1(i,j,k+1)-rho1(i,j,k) ! Elementary
     & +(qp1(i,j,k+1)-qp1(i,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ! difference



          enddo
        enddo
        do i=istrU-1,iend
          dR(i,N)=dR(i,N-1)
          dR(i,0)=dR(i,1)
          dZ(i,N)=dZ(i,N-1)
          dZ(i,0)=dZ(i,1)
        enddo
        do k=N,1,-1 !--> irreversible
          do i=istrU-1,iend
            cff=2.*dZ(i,k)*dZ(i,k-1)
            dZ(i,k)=cff/(dZ(i,k)+dZ(i,k-1))

            cfr=2.*dR(i,k)*dR(i,k-1)
            if (cfr.gt.epsil) then
              dR(i,k)=cfr/(dR(i,k)+dR(i,k-1))
            else
              dR(i,k)=0.
            endif

            dpth=z_w(i,j,N)-z_r(i,j,k)
            dR(i,k)=dR(i,k) -qp1(i,j,k)*dZ(i,k)*(1.-2.*qp2*dpth)
            rho(i,j,k)=rho1(i,j,k) +qp1(i,j,k)*dpth*(1.-qp2*dpth)

          enddo
        enddo
        do i=istrU-1,iend
          P(i,j,N)=g*z_w(i,j,N) + GRho*( rho(i,j,N)
     & +0.5*(rho(i,j,N)-rho(i,j,N-1))*(z_w(i,j,N)-z_r(i,j,N))
     & /(z_r(i,j,N)-z_r(i,j,N-1)) )*(z_w(i,j,N)-z_r(i,j,N))
        enddo
        do k=N-1,1,-1
          do i=istrU-1,iend
            P(i,j,k)=P(i,j,k+1)+HalfGRho*( (rho(i,j,k+1)+rho(i,j,k))
     & *(z_r(i,j,k+1)-z_r(i,j,k))

     & -OneFifth*( (dR(i,k+1)-dR(i,k))*( z_r(i,j,k+1)-z_r(i,j,k)
     & -OneTwelfth*(dZ(i,k+1)+dZ(i,k)) )

     & -(dZ(i,k+1)-dZ(i,k))*( rho(i,j,k+1)-rho(i,j,k)
     & -OneTwelfth*(dR(i,k+1)+dR(i,k)) )
     & ))
          enddo
        enddo
      enddo !<-- j




        ru = 0.
        rv = 0.



!
! Compute XI-component of pressure gradient term:
!-------- ------------ -- -------- -------- -----
!
      do k=N,1,-1
        do j=jstr,jend
          do i=imin,imax
            FC(i,j)=(z_r(i,j,k)-z_r(i-1,j,k))

            dpth=0.5*( z_w(i,j,N)+z_w(i-1,j,N)
     & -z_r(i,j,k)-z_r(i-1,j,k))

            rx(i,j)=( rho1(i,j,k)-rho1(i-1,j,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i-1,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference




          enddo
        enddo


        if (istr.eq.1) then ! Extrapolate elementary
          do j=jstr,jend ! differences near physical
            FC(imin-1,j)=FC(imin,j) ! boundaries to compencate.
            rx(imin-1,j)=rx(imin,j) ! for reduced loop ranges.
          enddo
        endif
        if (iend.eq.Lm) then
          do j=jstr,jend
            FC(imax+1,j)=FC(imax,j)
            rx(imax+1,j)=rx(imax,j)
          enddo
        endif


        do j=jstr,jend
          do i=istrU-1,iend
            cff=2.*FC(i,j)*FC(i+1,j)
            if (cff.gt.epsil) then
              dZx(i,j)=cff/(FC(i,j)+FC(i+1,j))
            else
              dZx(i,j)=0.
            endif

            cfr=2.*rx(i,j)*rx(i+1,j)
            if (cfr.gt.epsil) then
              dRx(i,j)=cfr/(rx(i,j)+rx(i+1,j))
            else
              dRx(i,j)=0.
            endif

            dRx(i,j)=dRx(i,j) -qp1(i,j,k)*dZx(i,j)
     & *(1.-2.*qp2*(z_w(i,j,N)-z_r(i,j,k)))

          enddo !--> discard FC, rx

          do i=istrU,iend
            ru(i,j,k)=
     & 0.5*(Hz(i,j,k)+Hz(i-1,j,k))*2./(pn(i,j)+pn(i-1,j))*(
     & P(i-1,j,k)-P(i,j,k)-HalfGRho*(

     & (rho(i,j,k)+rho(i-1,j,k))*(z_r(i,j,k)-z_r(i-1,j,k))

     & -OneFifth*( (dRx(i,j)-dRx(i-1,j))*( z_r(i,j,k)-z_r(i-1,j,k)
     & -OneTwelfth*(dZx(i,j)+dZx(i-1,j)) )

     & -(dZx(i,j)-dZx(i-1,j))*( rho(i,j,k)-rho(i-1,j,k)
     & -OneTwelfth*(dRx(i,j)+dRx(i-1,j)) )
     & )))
          enddo
        enddo
!
! ETA-component of pressure gradient term:
!-------------- -- -------- -------- -----
!
        do j=jmin,jmax
          do i=istr,iend
            FC(i,j)=(z_r(i,j,k)-z_r(i,j-1,k))

            dpth=0.5*( z_w(i,j,N)+z_w(i,j-1,N)
     & -z_r(i,j,k)-z_r(i,j-1,k))

            rx(i,j)=( rho1(i,j,k)-rho1(i,j-1,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i,j-1,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference



          enddo
        enddo


        if (jstr.eq.1) then
          do i=istr,iend
            FC(i,jmin-1)=FC(i,jmin)
            rx(i,jmin-1)=rx(i,jmin)

          enddo
        endif
        if (jend.eq.Mm) then
          do i=istr,iend
            FC(i,jmax+1)=FC(i,jmax)
            rx(i,jmax+1)=rx(i,jmax)
          enddo
        endif


        do j=jstrV-1,jend
          do i=istr,iend
            cff=2.*FC(i,j)*FC(i,j+1)
            if (cff.gt.epsil) then
c** if ((FC(i,j).gt.0. .and. FC(i,j+1).gt.0.) .or.
c** & (FC(i,j).lt.0. .and. FC(i,j+1).lt.0.)) then
              dZx(i,j)=cff/(FC(i,j)+FC(i,j+1))
            else
              dZx(i,j)=0.
            endif

            cfr=2.*rx(i,j)*rx(i,j+1)
            if (cfr.gt.epsil) then
c** if ((rx(i,j).gt.0. .and. rx(i,j+1).gt.0.) .or.
c** & (rx(i,j).lt.0. .and. rx(i,j+1).lt.0.)) then
              dRx(i,j)=cfr/(rx(i,j)+rx(i,j+1))
            else
              dRx(i,j)=0.
            endif

            dRx(i,j)=dRx(i,j) -qp1(i,j,k)*dZx(i,j)
     & *(1.-2.*qp2*(z_w(i,j,N)-z_r(i,j,k)))

          enddo !--> discard FC, rx

          if (j.ge.jstrV) then
            do i=istr,iend

            rv(i,j)=rv(i,j) +
     & 0.5*(Hz(i,j,k)+Hz(i,j-1,k))*2./(pm(i,j)+pm(i,j-1))*(
     & P(i,j-1,k)-P(i,j,k) -HalfGRho*(

     & (rho(i,j,k)+rho(i,j-1,k))*(z_r(i,j,k)-z_r(i,j-1,k))

     & -OneFifth*( (dRx(i,j)-dRx(i,j-1))*( z_r(i,j,k)-z_r(i,j-1,k)
     & -OneTwelfth*(dZx(i,j)+dZx(i,j-1)) )

     & -(dZx(i,j)-dZx(i,j-1))*( rho(i,j,k)-rho(i,j-1,k)
     & -OneTwelfth*(dRx(i,j)+dRx(i,j-1)) )
     & )))

            enddo
          endif
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Compute rotational
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



       do j=jmin+1,jmax-1
          do i=imin+1,imax-1
            do k=1,N
            ru(i,j,k) = ru(i,j,k)*0.5*(pn(i,j)+pn(i-1,j))
     & *(pm(i,j)+pm(i-1,j))
     & /(Hz(i,j,k)+Hz(i-1,j,k))
            enddo



            rv(i,j) = rv(i,j)*0.5*(pm(i,j)+pm(i,j-1))
     & *0.5*(pn(i,j)+pn(i,j-1))



           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
           cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

            bpt(i,j) = (rv(i,j) - rv(i-1,j)) * cff1
     & - (ru(i,j,1) - ru(i,j-1,1)) * cff2
        enddo
      enddo
# 474 "R_tools_fort_routines_gula/get_u_prsgrd.F"
      return
      end
# 87 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_vortplanet.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute planetary vorticity balance term
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_vortplanet(Lm,Mm, u,v,H,pm,pn,f,rot)


      implicit none
      integer Lm,Mm, imin,imax,jmin,jmax, i,j,k
      real*8 rot(1:Lm+1,1:Mm+1), H(0:Lm+1,0:Mm+1),
     & u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & f(0:Lm+1,0:Mm+1),
     & cff1,cff2


Cf2py intent(in) Lm,Mm,u,v,H,pm,pn,f
Cf2py intent(out) rot

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




      do j=jmin+1,jmax
        do i=imin+1,imax

            cff1 = 0.25*(H(i,j)+ H(i-1,j)) * u(i,j) *
     & (f(i,j) - f(i-1,j)) * 0.5 * (pm(i,j)+ pm(i-1,j))
     & + 0.25*(H(i,j-1)+ H(i-1,j-1)) * u(i,j-1) *
     & (f(i,j-1) - f(i-1,j-1)) * 0.5 * (pm(i,j-1)+ pm(i-1,j-1))


            cff2 = 0.25*(H(i,j)+ H(i,j-1)) * v(i,j) *
     & (f(i,j) - f(i,j-1)) * 0.5 * (pn(i,j)+ pn(i,j-1))
     & + 0.25*(H(i-1,j)+ H(i-1,j-1)) * v(i-1,j) *
     & (f(i-1,j) - f(i-1,j-1)) * 0.5 * (pn(i-1,j)+ pn(i-1,j-1))


            rot(i,j) = cff1 + cff2

        enddo !<- i
      enddo !<- j






!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      return
      end
# 89 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_vortstretch.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute planetary stretching term from the vorticity balance
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_vortstretch(Lm,Mm, u,v,H,pm,pn,f,rot)


      implicit none
      integer Lm,Mm, imin,imax,jmin,jmax, i,j,k
      real*8 rot(1:Lm+1,1:Mm+1),H(0:Lm+1,0:Mm+1),
     & u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & Hu(0:Lm+2), Hv(0:Mm+2),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & f(0:Lm+1,0:Mm+1),
     & dudx(0:Lm+1,0:Mm+1), dvdy(0:Lm+1,1:Mm+1),
     & cff1,cff2,cff


Cf2py intent(in) Lm,Mm,u,v,H,pm,pn,f
Cf2py intent(out) rot

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin,jmax
        do i=imin+1,imax

            !cff = 0.5*(pm(i,j)+pm(i-1,j))*0.5*(pn(i,j)+pn(i-1,j))
            cff = 0.5*(pn(i,j)+pn(i-1,j))
            Hu(i) = 0.5*(H(i,j)+ H(i-1,j)) * u(i,j) / cff

        enddo !<- i
            Hu(imin) = Hu(imin+1)
            Hu(imax+1) = Hu(imax)

        do i=imin,imax
            dudx(i,j) = f(i,j) *(Hu(i+1) - Hu(i)) !* pm(i,j)/pn(i,j)
        enddo !<- i

      enddo !<- j

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do i=imin,imax

        do j=jmin+1,jmax

            !cff = 0.5*(pm(i,j)+pm(i,j-1))*0.5*(pn(i,j)+pn(i,j-1))
            cff = 0.5*(pm(i,j)+pm(i,j-1))
            Hv(j) = 0.5*(H(i,j)+ H(i,j-1)) * v(i,j) / cff
        enddo !<- j
            Hv(jmin) = Hv(jmin+1)
            Hv(jmax+1) = Hv(jmax)

        do j=jmin,jmax
            dvdy(i,j) = f(i,j) *(Hv(j+1) - Hv(j)) !* pn(i,j)*pm(i,j)
        enddo !<- j

      enddo !<- i

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do i=imin+1,imax
        do j=jmin+1,jmax

         cff =0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
     & * 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))

              rot(i,j) = 0.25*(dudx(i,j) + dudx(i-1,j)
     & + dudx(i,j-1) + dudx(i-1,j-1)
     & + dvdy(i,j) + dvdy(i-1,j)
     & + dvdy(i,j-1) + dvdy(i-1,j-1))*cff

        enddo !<- i
      enddo !<- j






!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      return
      end
# 91 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_intvortplanet.F" 1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      subroutine get_intvortplanet (Lm,Mm,N,u,v, z_r,z_w,pm,pn,f
     & ,depth1,depth2,vrtCor)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1), FlxV(0:Lm+1,1:Mm+1),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & zu(1:Lm+1,0:Mm+1,0:N), zv(0:Lm+1,1:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & f(0:Lm+1,0:Mm+1),
     & cff, cff1, cff2, depth1, depth2

      real*8 vrtCor(1:Lm+1,1:Mm+1)


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
# 32 "R_tools_fort_routines_gula/get_intvortplanet.F" 2


Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,f,depth1,depth2
Cf2py intent(out) vrtCor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      depth1=abs(depth1)
      depth2=abs(depth2)

      if (depth2 .lt. depth1) then
            cff = depth1
            depth1 = depth2
            depth2 = cff
      end if

! write(*,*) 'depths are', depth1, depth2

      do j=jmin,jmax
        do i=imin,imax
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo

          do k=0,N,+1
           if (i.gt.0) zu(i,j,k) = -0.5*(z_w(i,j,k)+z_w(i-1,j,k))
           if (j.gt.0) zv(i,j,k) = -0.5*(z_w(i,j,k)+z_w(i,j-1,k))
          enddo
        enddo
      enddo


      do j=jmin,jmax
        do i=imin-1,imax

          FlxU(i,j) = 0
          do k=N,1,-1

            if (zu(i,j,k-1).le.depth2) then

                if ((zu(i,j,k-1).ge.depth1) .and.
     & (zu(i,j,k).le.depth1)) then
                    FlxU(i,j) = (zu(i,j,k-1)-depth1) * u(i,j,k)

                else if (zu(i,j,k-1).ge.depth1) then
                    FlxU(i,j) = FlxU(i,j) +
     & 0.5*(Hz(i,j,k)+Hz(i-1,j,k)) * u(i,j,k)
                end if


            else if ((depth2-zu(i,j,k)).gt.0) then

                FlxU(i,j) = FlxU(i,j) +
     & (depth2-zu(i,j,k)) * u(i,j,k)

            end if

          enddo

        enddo !<- i
      enddo !<- j




      do j=jmin-1,jmax
        do i=imin,imax

          FlxV(i,j) = 0
          do k=N,1,-1

            if (zv(i,j,k-1).le.depth2) then

                if ((zv(i,j,k-1).ge.depth1) .and.
     & (zv(i,j,k).le.depth1)) then
                    FlxV(i,j) = (zv(i,j,k-1)-depth1) * v(i,j,k)

                else if (zv(i,j,k-1).ge.depth1) then
                    FlxV(i,j) = FlxV(i,j) +
     & 0.5*(Hz(i,j,k)+Hz(i,j-1,k)) * v(i,j,k)
                end if


            else if ((depth2-zv(i,j,k)).gt.0) then

                FlxV(i,j) = FlxV(i,j) +
     & (depth2-zv(i,j,k)) * v(i,j,k)

            end if

          enddo
        enddo !<- i
      enddo !<- j




      do j=jmin+1,jmax
        do i=imin+1,imax

            cff1 = 0.5*FlxU(i,j) *
     & (f(i,j) - f(i-1,j)) * 0.5 * (pm(i,j)+ pm(i-1,j))
     & + 0.5*FlxU(i,j-1) *
     & (f(i,j-1) - f(i-1,j-1)) * 0.5 * (pm(i,j-1)+ pm(i-1,j-1))


            cff2 = 0.5*FlxV(i,j) *
     & (f(i,j) - f(i,j-1)) * 0.5 * (pn(i,j)+ pn(i,j-1))
     & + 0.5*FlxV(i-1,j) *
     & (f(i-1,j) - f(i-1,j-1)) * 0.5 * (pn(i-1,j)+ pn(i-1,j-1))


            vrtCor(i,j) = cff1 + cff2

        enddo !<- i
      enddo !<- j
# 177 "R_tools_fort_routines_gula/get_intvortplanet.F"
      return
      end
# 93 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_vortstretch_sol2.F" 1



!======================================================================
!
! Compute only stretching part of the barotropic vorticity balance equation
!
! = f div(U)
!
! This is vortplantot_sol2 - vortadv_uvgrid - vortplanet
!
! This does not include contribution from !
!
!======================================================================


      subroutine get_vortstretch_sol2(Lm,Mm,u,v, Hz,pm,pn,f
     & ,vrtCor)

      implicit none

      integer Lm,Mm, imin,imax,jmin,jmax, i,j


      real*8 u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & FlxU(1:Lm+1,0:Mm+1), FlxV(0:Lm+1,1:Mm+1),
     & Hz(0:Lm+1,0:Mm+1), gamma,
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & f(0:Lm+1,0:Mm+1), fomn(0:Lm+1,0:Mm+1),
     & dmde(0:Lm+1,0:Mm+1), dndx(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2


      real*8 wrkCor(0:Lm+1,0:Mm+1,2)

      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1)

      real*8 vrtCor(1:Lm+1,1:Mm+1)


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
# 45 "R_tools_fort_routines_gula/get_vortstretch_sol2.F" 2

      parameter (gamma=0.25)

Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,f
Cf2py intent(out) vrtCor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jmin,jmax
        do i=imin,imax
           fomn(i,j)=1./(pm(i,j)*pn(i,j))
        enddo
      enddo


! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
! dmde(i,j) = 0.5/pm(i,j+1)-0.5/pm(i,j-1)
! dndx(i,j) = 0.5/pn(i+1,j)-0.5/pn(i-1,j)
! enddo
! enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





        do j=jmin+1,jmax-1
          do i=imin+1,imax-1
            cff=0.5*Hz(i,j)*(
     & fomn(i,j)
! & +0.5*( (v(i,j)+v(i,j+1))*dndx(i,j)
! & -(u(i,j)+u(i+1,j))*dmde(i,j))
     & )
            UFx(i,j)=cff*(v(i,j)+v(i,j+1))
            VFe(i,j)=cff*(u(i,j)+u(i+1,j))
          enddo
        enddo

        do j=jmin+1,jmax-1
          do i=imin+1,imax-1
                wrkCor(i,j,1) = 0.5*(UFx(i,j)+UFx(i-1,j))
                wrkCor(i,j,2) = -0.5*(VFe(i,j)+VFe(i,j-1))
          enddo
        enddo



! Divide all diagnostic terms by (pm*pn).
! There after the unit of these terms are :
! s-2

!
! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
!
! cff=0.25*(pm(i,j)+pm(i-1,j))
! & *(pn(i,j)+pn(i-1,j))
!
!
! wrkCor(i,j,1)=wrkCor(i,j,1)*cff
!
!
! cff=0.25*(pm(i,j)+pm(i,j-1))
! & *(pn(i,j)+pn(i,j-1))
!
!
! wrkCor(i,j,2)=wrkCor(i,j,2)*cff
!
!
! enddo
! enddo
!
!
!
! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
!
! cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
! cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
!
!
! vrtCor(i,j) = (wrkCor(i,j,2) - wrkCor(i-1,j,2)) * cff1
! & - (wrkCor(i,j,1) - wrkCor(i,j-1,1)) * cff2
!
! vrtCor(i,j) = vrtCor(i,j)
! & * 0.25*(f(i,j) + f(i-1,j) + f(i,j-1) + f(i-1,j-1))
! & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
! & * 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
!
!
! enddo
! enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/09/13)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin+1,jmax-1
        do i=imin+1,imax-1

            cff=0.5*(pn(i,j)+pn(i-1,j))
            wrkCor(i,j,1)=wrkCor(i,j,1)*cff

            cff=0.5*(pm(i,j)+pm(i,j-1))
            wrkCor(i,j,2)=wrkCor(i,j,2)*cff

        enddo
      enddo

      do j=jmin+1,jmax-1
        do i=imin+1,imax-1

           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
     & * 0.25*(f(i,j) + f(i-1,j) + f(i,j-1) + f(i-1,j-1))


            vrtCor(i,j) = (wrkCor(i,j,2) - wrkCor(i-1,j,2)) * cff1
     & - (wrkCor(i,j,1) - wrkCor(i,j-1,1)) * cff1

        enddo
       enddo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      return
      end
# 95 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_vortplantot.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute planetary stretching term from the vorticity balance
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_vortplantot(Lm,Mm, u,v,H,pm,pn,f,rot)


      implicit none
      integer Lm,Mm, imin,imax,jmin,jmax, i,j,k
      real*8 rot(1:Lm+1,1:Mm+1),H(0:Lm+1,0:Mm+1),
     & u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & Hu(0:Lm+2), Hv(0:Mm+2),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & f(0:Lm+1,0:Mm+1),
     & dudx(0:Lm+1,0:Mm+1), dvdy(0:Lm+1,1:Mm+1),
     & cff1,cff2,cff


Cf2py intent(in) Lm,Mm,u,v,H,pm,pn,f
Cf2py intent(out) rot

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin,jmax

        do i=imin+1,imax
            cff = 0.5*(pn(i,j)+pn(i-1,j))
            Hu(i) = 0.5*(f(i,j)+ f(i-1,j))
     & *0.5*(H(i,j)+ H(i-1,j)) * u(i,j)/ cff
        enddo !<- i
            Hu(imin) = Hu(imin+1)
            Hu(imax+1) = Hu(imax)

        do i=imin,imax
            dudx(i,j) = (Hu(i+1) - Hu(i))
        enddo !<- i

      enddo !<- j

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do i=imin,imax

        do j=jmin+1,jmax
            cff = 0.5*(pm(i,j)+pm(i,j-1))
            Hv(j) = 0.5*(f(i,j)+ f(i,j-1))
     & * 0.5 * (H(i,j)+ H(i,j-1)) * v(i,j)/ cff
        enddo !<- j
            Hv(jmin) = Hv(jmin+1)
            Hv(jmax+1) = Hv(jmax)

        do j=jmin,jmax
            dvdy(i,j) = (Hv(j+1) - Hv(j))
        enddo !<- j

      enddo !<- i

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do i=imin+1,imax
        do j=jmin+1,jmax

         cff =0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
     & * 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))

              rot(i,j) = 0.25*(dudx(i,j) + dudx(i-1,j)
     & + dudx(i,j-1) + dudx(i-1,j-1)
     & + dvdy(i,j) + dvdy(i-1,j)
     & + dvdy(i,j-1) + dvdy(i-1,j-1))*cff

        enddo !<- i
      enddo !<- j






!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      return
      end
# 97 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_vortplantot_sol2.F" 1



!======================================================================
!
! Compute Coriolis part of the barotropic vorticity balance equation
!
! = div(fU)
!
!
! This includes contribution from !
!
! - updated 16/08/18
!======================================================================


      subroutine get_vortplantot_sol2 (Lm,Mm,N,u,v, Hz,pm,pn,f
     & ,rmask,vrtCor)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & rmask(0:Lm+1,0:Mm+1),
     & umask(0:Lm+1,0:Mm+1), vmask(0:Lm+1,0:Mm+1),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & f(0:Lm+1,0:Mm+1), fomn(0:Lm+1,0:Mm+1),
     & dmde(0:Lm+1,0:Mm+1), dndx(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2


      real*8 wrkCor(0:Lm+1,0:Mm+1,2)

      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1)

      real*8 vrtCor(1:Lm+1,1:Mm+1)

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
# 46 "R_tools_fort_routines_gula/get_vortplantot_sol2.F" 2

      parameter (gamma=0.25)

Cf2py intent(in) Lm,Mm,N, u,v,Hz,pm,pn,f,rmask
Cf2py intent(out) vrtCor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jmin,jmax
        do i=imin+1,imax
            umask(i,j) = rmask(i,j)*rmask(i-1,j)
        enddo
      enddo

      do j=jmin+1,jmax
        do i=imin,imax
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
        enddo
      enddo


      do j=jmin,jmax
        do i=imin,imax
           fomn(i,j)=f(i,j)/(pm(i,j)*pn(i,j))
        enddo
      enddo


      do j=jmin+1,jmax-1
        do i=imin+1,imax-1
            dmde(i,j) = 0.5/pm(i,j+1)-0.5/pm(i,j-1)
            dndx(i,j) = 0.5/pn(i+1,j)-0.5/pn(i-1,j)
         enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




      do k=1,N

        do j=jmin+1,jmax-1
          do i=imin+1,imax-1
            cff=0.5*Hz(i,j,k)*(
     & fomn(i,j)
     & +0.5*( (v(i,j,k)+v(i,j+1,k))*dndx(i,j)
     & -(u(i,j,k)+u(i+1,j,k))*dmde(i,j))
     & )
            UFx(i,j)=cff*(v(i,j,k)+v(i,j+1,k))
            VFe(i,j)=cff*(u(i,j,k)+u(i+1,j,k))
          enddo
        enddo

        do j=jmin+1,jmax-1
          do i=imin+1,imax-1


              if (k.eq.1) then
                wrkCor(i,j,1) = 0.5*(UFx(i,j)+UFx(i-1,j))
              else
                wrkCor(i,j,1) = wrkCor(i,j,1) +
     & 0.5*(UFx(i,j)+UFx(i-1,j))
              endif

          enddo
        enddo
        do j=jmin+1,jmax-1
          do i=imin+1,imax-1


              if (k.eq.1) then
                wrkCor(i,j,2) = -0.5*(VFe(i,j)+VFe(i,j-1))
              else
                wrkCor(i,j,2) = wrkCor(i,j,2)
     & -0.5*(VFe(i,j)+VFe(i,j-1))
              endif

          enddo
        enddo


      enddo





! Divide all diagnostic terms by (pm*pn).
! There after the unit of these terms are :
! s-2


! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
!
! cff=0.25*(pm(i,j)+pm(i-1,j))
! & *(pn(i,j)+pn(i-1,j))
!
!
! wrkCor(i,j,1)=wrkCor(i,j,1)*cff
!
!
! cff=0.25*(pm(i,j)+pm(i,j-1))
! & *(pn(i,j)+pn(i,j-1))
!
!
! wrkCor(i,j,2)=wrkCor(i,j,2)*cff
!
!
! enddo
! ! enddo
!
!
!
! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
!
! cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
! cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
!
! vrtCor(i,j) = (wrkCor(i,j,2) - wrkCor(i-1,j,2)) * cff1
! & - (wrkCor(i,j,1) - wrkCor(i,j-1,1)) * cff2
!
!
! enddo
! enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/09/13)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin+1,jmax-1
        do i=imin+1,imax-1

            cff=0.5*(pn(i,j)+pn(i-1,j)) * umask(i,j)
            wrkCor(i,j,1)=wrkCor(i,j,1)*cff

            cff=0.5*(pm(i,j)+pm(i,j-1)) * vmask(i,j)
            wrkCor(i,j,2)=wrkCor(i,j,2)*cff

        enddo
      enddo

      do j=jmin+1,jmax-1
        do i=imin+1,imax-1

           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

           cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
     & * 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))

            vrtCor(i,j) = (wrkCor(i,j,2) - wrkCor(i-1,j,2)) * cff1
     & - (wrkCor(i,j,1) - wrkCor(i,j-1,1)) * cff2

        enddo
       enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      return
      end
# 99 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_vortplantot_sol2_test.F" 1



!======================================================================
!
! Compute Coriolis part of the barotropic vorticity balance equation
!
! = div(fU)
!
!
! This includes contribution from !
!
! - updated 16/08/18
!======================================================================


      subroutine get_vortplantot_sol2_test (Lm,Mm,N,u,v, Hz,pm,pn,f
     & ,rmask,wrkCor,vrtCor)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & rmask(0:Lm+1,0:Mm+1),
     & umask(0:Lm+1,0:Mm+1), vmask(0:Lm+1,0:Mm+1),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & f(0:Lm+1,0:Mm+1), fomn(0:Lm+1,0:Mm+1),
     & dmde(0:Lm+1,0:Mm+1), dndx(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2

      real*8 wrkCor(0:Lm+1,0:Mm+1,2)

      real*8 vrtCor(0:Lm+1,0:Mm+1)

      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1)


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
# 46 "R_tools_fort_routines_gula/get_vortplantot_sol2_test.F" 2

      parameter (gamma=0.25)

Cf2py intent(in) Lm,Mm,N, u,v,Hz,pm,pn,f,rmask
Cf2py intent(out) wrkCor,vrtCor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jmin,jmax
        do i=imin+1,imax
            umask(i,j) = rmask(i,j)*rmask(i-1,j)
        enddo
      enddo

      do j=jmin+1,jmax
        do i=imin,imax
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
        enddo
      enddo


      do j=jmin,jmax
        do i=imin,imax
           fomn(i,j)=f(i,j)/(pm(i,j)*pn(i,j))
        enddo
      enddo


      do j=jmin+1,jmax-1
        do i=imin+1,imax-1
            dmde(i,j) = 0.5/pm(i,j+1)-0.5/pm(i,j-1)
            dndx(i,j) = 0.5/pn(i+1,j)-0.5/pn(i-1,j)
         enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




      do k=1,N

        do j=jmin+1,jmax-1
          do i=imin+1,imax-1
            cff=0.5*Hz(i,j,k)*(
     & fomn(i,j)
     & )
            UFx(i,j)=cff*(v(i,j,k)+v(i,j+1,k))
            VFe(i,j)=cff*(u(i,j,k)+u(i+1,j,k))

          enddo
        enddo

        do j=jmin+1,jmax-1
          do i=imin+1,imax-1


              if (k.eq.1) then
                wrkCor(i,j,1) = 0.5*(UFx(i,j)+UFx(i-1,j))
     & * umask(i,j)
              else
                wrkCor(i,j,1) = wrkCor(i,j,1) +
     & 0.5*(UFx(i,j)+UFx(i-1,j))
     & * umask(i,j)
              endif

          enddo
        enddo
        do j=jmin+1,jmax-1
          do i=imin+1,imax-1


              if (k.eq.1) then
                wrkCor(i,j,2) = -0.5*(VFe(i,j)+VFe(i,j-1))
     & * vmask(i,j)
              else
                wrkCor(i,j,2) = wrkCor(i,j,2)
     & -0.5*(VFe(i,j)+VFe(i,j-1))
     & * vmask(i,j)
              endif

          enddo
        enddo


      enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/09/13)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin+1,jmax-1
        do i=imin+1,imax-1

            cff=0.5*(pn(i,j)+pn(i-1,j)) * umask(i,j)
            wrkCor(i,j,1)=wrkCor(i,j,1)*cff

            cff=0.5*(pm(i,j)+pm(i,j-1)) * vmask(i,j)
            wrkCor(i,j,2)=wrkCor(i,j,2)*cff

        enddo
      enddo

      do j=jmin+1,jmax-1
        do i=imin+1,imax-1

          cff = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

          vrtCor(i,j) = (wrkCor(i,j,2) - wrkCor(i-1,j,2)) * cff
     & - (wrkCor(i,j,1) - wrkCor(i,j-1,1)) * cff

        enddo
       enddo



      return
      end
# 101 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_adv_sol2.F" 1
!======================================================================
!
! Compute Advective part of the barotropic vorticity balance equation
!
!
!
! - updated 16/08/19 [add umask,vmask]
!======================================================================



      subroutine get_adv_sol2 (Lm,Mm,N,u,v, z_r,z_w,pm,pn,rmask
     & ,adv)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & rmask(0:Lm+1,0:Mm+1),
     & umask(0:Lm+1,0:Mm+1), vmask(0:Lm+1,0:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2


      real*8 wrkXadv(0:Lm+1,0:Mm+1,2),wrkYadv(0:Lm+1,0:Mm+1,2)

      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1)

      real*8 adv(1:Lm+1,1:Mm+1), vrtXadv(1:Lm+1,1:Mm+1),
     & vrtYadv(1:Lm+1,1:Mm+1)

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
# 42 "R_tools_fort_routines_gula/get_adv_sol2.F" 2

      parameter (gamma=0.25)

Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,rmask
Cf2py intent(out) adv

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

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





      do k=1,N

!
! Add in horizontal advection of momentum: Compute diagonal [UFx,VFe]
! and off-diagonal [UFe,VFx] components of tensor of momentum flux
! due to horizontal advection; after that add divergence of these
! terms to r.h.s.
!





        do j=jmin,jmax
         do i=imin+2,imax-1

            wrk1(i,j)=u(i-1,j,k)-2.*u(i,j,k)
     & +u(i+1,j,k)
            wrk2(i,j)=FlxU(i-1,j,k)-2.*FlxU(i,j,k)
     & +FlxU(i+1,j,k)
          enddo
        enddo



        do j=jmin,jmax
         do i=imin+2,imax-2


            cff=FlxU(i,j,k)+FlxU(i+1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i+1,j))
            UFx(i,j)=0.25*( cff*(u(i,j,k)+u(i+1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i ,j)
     & +min(cff,0.)*wrk1(i+1,j)
     & ))

          enddo
        enddo
# 144 "R_tools_fort_routines_gula/get_adv_sol2.F"
        do j=jmin+2,jmax-1
         do i=imin,imax

            wrk1(i,j)=v(i,j-1,k)-2.*v(i,j,k)+v(i,j+1,k)
            wrk2(i,j)=FlxV(i,j-1,k)-2.*FlxV(i,j,k)+FlxV(i,j+1,k)
          enddo
        enddo


        do j=jmin+2,jmax-2
         do i=imin,imax


            cff=FlxV(i,j,k)+FlxV(i,j+1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j+1))
            VFe(i,j)=0.25*( cff*(v(i,j,k)+v(i,j+1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j )
     & +min(cff,0.)*wrk1(i,j+1)
     & ))

          enddo
        enddo
# 174 "R_tools_fort_routines_gula/get_adv_sol2.F"
       do j=jmin+1,jmax-1
         do i=imin,imax
            wrk1(i,j)=u(i,j-1,k)-2.*u(i,j,k)
     & +u(i,j+1,k)
          enddo
        enddo

        do j=jmin,jmax
         do i=imin+1,imax-1
           wrk2(i,j)=FlxV(i-1,j,k)-2.*FlxV(i,j,k)+FlxV(i+1,j,k)
          enddo
        enddo

        do j=jmin+2,jmax-1
         do i=imin+2,imax-1

            cff=FlxV(i,j,k)+FlxV(i-1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i-1,j))
            UFe(i,j)=0.25*( cff*(u(i,j,k)+u(i,j-1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j-1)
     & +min(cff,0.)*wrk1(i,j )
     & ))

          enddo
        enddo







        do j=jmin+1,jmax
         do i=imin+1,imax-1

            wrk1(i,j)=v(i-1,j,k)-2.*v(i,j,k)
     & +v(i+1,j,k)
          enddo
        enddo


        do j=jmin+1,jmax-1
         do i=imin+1,imax

           wrk2(i,j)=FlxU(i,j-1,k)-2.*FlxU(i,j,k)+FlxU(i,j+1,k)
          enddo
        enddo

        do j=jmin+2,jmax-1
         do i=imin+2,imax-1

            cff=FlxU(i,j,k)+FlxU(i,j-1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j-1))
            VFx(i,j)=0.25*( cff*(v(i,j,k)+v(i-1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i-1,j)
     & +min(cff,0.)*wrk1(i ,j)
     & ))


          enddo
        enddo




      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

              if (k.eq.1) then
                wrkXadv(i,j,1) = -UFx(i,j)+UFx(i-1,j)
                wrkYadv(i,j,1) = -UFe(i,j+1)+UFe(i,j)
              else
                wrkXadv(i,j,1) = wrkXadv(i,j,1) - UFx(i,j)+UFx(i-1,j)
                wrkYadv(i,j,1) = wrkYadv(i,j,1) - UFe(i,j+1)+UFe(i,j)
              endif


          enddo
        enddo

      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

              if (k.eq.1) then
                wrkXadv(i,j,2) = -VFx(i+1,j)+VFx(i,j)
                wrkYadv(i,j,2) = -VFe(i,j)+VFe(i,j-1)
              else
                wrkXadv(i,j,2) = wrkXadv(i,j,2) -VFx(i+1,j)+VFx(i,j)
                wrkYadv(i,j,2) = wrkYadv(i,j,2) -VFe(i,j)+VFe(i,j-1)
              endif

          enddo
        enddo
      enddo





! ! Divide all diagnostic terms by (pm*pn).
! ! There after the unit of these terms are :
! ! s-2
!
!
! do j=jmin+2,jmax-1
! do i=imin+2,imax-1
!
! cff=0.25*(pm(i,j)+pm(i-1,j))
! & *(pn(i,j)+pn(i-1,j))
!
!
! wrkXadv(i,j,1)=wrkXadv(i,j,1)*cff
! wrkYadv(i,j,1)=wrkYadv(i,j,1)*cff
!
!
! cff=0.25*(pm(i,j)+pm(i,j-1))
! & *(pn(i,j)+pn(i,j-1))
!
!
! wrkXadv(i,j,2)=wrkXadv(i,j,2)*cff
! wrkYadv(i,j,2)=wrkYadv(i,j,2)*cff
!
! enddo
! enddo
!
!
!
! do j=jmin+2,jmax-2
! do i=imin+2,imax-2
!
! cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
! cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
!
! vrtXadv(i,j) = (wrkXadv(i,j,2) - wrkXadv(i-1,j,2)) * cff1
! & - (wrkXadv(i,j,1) - wrkXadv(i,j-1,1)) * cff2
!
!
! vrtYadv(i,j) = (wrkYadv(i,j,2) - wrkYadv(i-1,j,2)) * cff1
! & - (wrkYadv(i,j,1) - wrkYadv(i,j-1,1)) * cff2
!
!
! adv(i,j) = vrtXadv(i,j) + vrtYadv(i,j)
!
!
! enddo
! enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin,jmax
        do i=imin+1,imax
            umask(i,j) = rmask(i,j)*rmask(i-1,j)
        enddo
      enddo

      do j=jmin+1,jmax
        do i=imin,imax
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/09/13)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

            cff=0.5*(pn(i,j)+pn(i-1,j))

            wrkXadv(i,j,1)=wrkXadv(i,j,1)*cff * umask(i,j)
            wrkYadv(i,j,1)=wrkYadv(i,j,1)*cff * umask(i,j)

            cff=0.5*(pm(i,j)+pm(i,j-1))

            wrkXadv(i,j,2)=wrkXadv(i,j,2)*cff * vmask(i,j)
            wrkYadv(i,j,2)=wrkYadv(i,j,2)*cff * vmask(i,j)

          enddo
        enddo



      do j=jmin+2,jmax-2
        do i=imin+2,imax-2

           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

            vrtXadv(i,j) = (wrkXadv(i,j,2) - wrkXadv(i-1,j,2)) * cff1
     & - (wrkXadv(i,j,1) - wrkXadv(i,j-1,1)) * cff1


            vrtYadv(i,j) = (wrkYadv(i,j,2) - wrkYadv(i-1,j,2)) * cff1
     & - (wrkYadv(i,j,1) - wrkYadv(i,j-1,1)) * cff1


            adv(i,j) = vrtXadv(i,j) + vrtYadv(i,j)


          enddo
         enddo





      return
      end
# 103 "R_tools_fort_gula.F" 2



# 1 "R_tools_fort_routines_gula/get_uvgrid.F" 1
!======================================================================
!
! Compute UV_GRID part of the barotropic vorticity balance equation
!
!
!
! - updated 16/08/19 [add umask,vmask]
!======================================================================


      subroutine get_uvgrid (Lm,Mm,N,u,v, z_r,z_w,pm,pn,f,rmask
     & ,vrtCor)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & rmask(0:Lm+1,0:Mm+1),
     & umask(0:Lm+1,0:Mm+1), vmask(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & f(0:Lm+1,0:Mm+1),
     & dmde(0:Lm+1,0:Mm+1), dndx(0:Lm+1,0:Mm+1),
     & cff, cff1, cff2


      real*8 wrkCor(0:Lm+1,0:Mm+1,2)

      real*8 UFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1)

      real*8 vrtCor(1:Lm+1,1:Mm+1)


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
# 40 "R_tools_fort_routines_gula/get_uvgrid.F" 2

      parameter (gamma=0.25)

Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,f,rmask
Cf2py intent(out) vrtCor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jmin,jmax
        do i=imin,imax
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo
        enddo
      enddo


      do j=jmin+1,jmax-1
        do i=imin+1,imax-1
            dmde(i,j) = 0.5/pm(i,j+1)-0.5/pm(i,j-1)
            dndx(i,j) = 0.5/pn(i+1,j)-0.5/pn(i-1,j)
         enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




      do k=1,N

        do j=jmin+1,jmax-1
          do i=imin+1,imax-1
            cff=0.5*Hz(i,j,k)*(
     & 0.5*( (v(i,j,k)+v(i,j+1,k))*dndx(i,j)
     & -(u(i,j,k)+u(i+1,j,k))*dmde(i,j))
     & )
            UFx(i,j)=cff*(v(i,j,k)+v(i,j+1,k))
            VFe(i,j)=cff*(u(i,j,k)+u(i+1,j,k))
          enddo
        enddo

        do j=jmin+1,jmax-1
          do i=imin+1,imax-1


              if (k.eq.1) then
                wrkCor(i,j,1) = 0.5*(UFx(i,j)+UFx(i-1,j))
              else
                wrkCor(i,j,1) = wrkCor(i,j,1) +
     & 0.5*(UFx(i,j)+UFx(i-1,j))
              endif

          enddo
        enddo
        do j=jmin+1,jmax-1
          do i=imin+1,imax-1


              if (k.eq.1) then
                wrkCor(i,j,2) = -0.5*(VFe(i,j)+VFe(i,j-1))
              else
                wrkCor(i,j,2) = wrkCor(i,j,2)
     & -0.5*(VFe(i,j)+VFe(i,j-1))
              endif

          enddo
        enddo


      enddo





! ! Divide all diagnostic terms by (pm*pn).
! ! There after the unit of these terms are :
! ! s-2
! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
! cff=0.25*(pm(i,j)+pm(i-1,j))
! & *(pn(i,j)+pn(i-1,j))
!
! wrkCor(i,j,1)=wrkCor(i,j,1)*cff
! cff=0.25*(pm(i,j)+pm(i,j-1))
! & *(pn(i,j)+pn(i,j-1))
! wrkCor(i,j,2)=wrkCor(i,j,2)*cff
! enddo
! enddo
! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
! cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
! cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
! vrtCor(i,j) = (wrkCor(i,j,2) - wrkCor(i-1,j,2)) * cff1
! & - (wrkCor(i,j,1) - wrkCor(i,j-1,1)) * cff2
! enddo
! enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin,jmax
        do i=imin+1,imax
            umask(i,j) = rmask(i,j)*rmask(i-1,j)
        enddo
      enddo

      do j=jmin+1,jmax
        do i=imin,imax
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/09/13)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin,jmax-1
        do i=imin+1,imax-1

            cff=0.5*(pn(i,j)+pn(i-1,j)) * umask(i,j)
            wrkCor(i,j,1)=wrkCor(i,j,1)*cff

        enddo
      enddo

      do j=jmin+1,jmax-1
        do i=imin,imax-1

            cff=0.5*(pm(i,j)+pm(i,j-1)) * vmask(i,j)
            wrkCor(i,j,2)=wrkCor(i,j,2)*cff

        enddo
      enddo

      do j=jmin+1,jmax-1
        do i=imin+1,imax-1

           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

           cff2 = cff1

            vrtCor(i,j) = (wrkCor(i,j,2) - wrkCor(i-1,j,2)) * cff1
     & - (wrkCor(i,j,1) - wrkCor(i,j-1,1)) * cff2

        enddo
       enddo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      return
      end
# 107 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_adv_mix.F" 1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute Advective part of the barotropic vorticity balance equation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      subroutine get_adv_mix (Lm,Mm,N,u,v, z_r,z_w,pm,pn,rmask
     & ,adv)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & rmask(0:Lm+1,0:Mm+1),
     & umask(0:Lm+1,0:Mm+1), vmask(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2


      real*8 wrkXadv(0:Lm+1,0:Mm+1,2),wrkYadv(0:Lm+1,0:Mm+1,2)

      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1)

      real*8 adv(1:Lm+1,1:Mm+1), vrtXadv(1:Lm+1,1:Mm+1),
     & vrtYadv(1:Lm+1,1:Mm+1)

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
# 41 "R_tools_fort_routines_gula/get_adv_mix.F" 2

      parameter (gamma=0.25)

Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,rmask
Cf2py intent(out) adv

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

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





      do k=1,N

!
! Add in horizontal advection of momentum: Compute diagonal [UFx,VFe]
! and off-diagonal [UFe,VFx] components of tensor of momentum flux
! due to horizontal advection; after that add divergence of these
! terms to r.h.s.
!





        do j=jmin,jmax
         do i=imin+2,imax-1

            wrk1(i,j)=u(i-1,j,k)-2.*u(i,j,k)
     & +u(i+1,j,k)
            wrk2(i,j)=FlxU(i-1,j,k)-2.*FlxU(i,j,k)
     & +FlxU(i+1,j,k)
          enddo
        enddo





        do j=jmin,jmax
         do i=imin+2,imax-2
            cff=FlxU(i,j,k)+FlxU(i+1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i+1,j))
         UFx(i,j)=0.25*( cff*(u(i,j,k)+u(i+1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i ,j)
     & +min(cff,0.)*wrk1(i+1,j)
     & ))

        UFx(i,j) = UFx(i,j) - 0.25*( u(i,j,k)+u(i+1,j,k)
     & -0.125*(wrk1(i,j)+wrk1(i+1,j))
     & )*( FlxU(i,j,k)+FlxU(i+1,j,k)
     & -0.125*(wrk2(i,j)+wrk2(i+1,j)))

          enddo
        enddo
# 152 "R_tools_fort_routines_gula/get_adv_mix.F"
        do j=jmin+2,jmax-1
         do i=imin,imax

            wrk1(i,j)=v(i,j-1,k)-2.*v(i,j,k)+v(i,j+1,k)

            wrk2(i,j)=FlxV(i,j-1,k)-2.*FlxV(i,j,k)+FlxV(i,j+1,k)

          enddo
        enddo


        do j=jmin+2,jmax-2
         do i=imin,imax


            cff=FlxV(i,j,k)+FlxV(i,j+1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j+1))

            VFe(i,j)=0.25*( cff*(v(i,j,k)+v(i,j+1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j )
     & +min(cff,0.)*wrk1(i,j+1)
     & ))


        VFe(i,j)=VFe(i,j) - 0.25*( v(i,j,k)+v(i,j+1,k)
     & -0.125*(wrk1(i,j)+wrk1(i,j+1))
     & )*( FlxV(i,j,k)+FlxV(i,j+1,k)
     & -0.125*(wrk2(i,j)+wrk2(i,j+1)))


          enddo
        enddo
# 194 "R_tools_fort_routines_gula/get_adv_mix.F"
       do j=jmin+1,jmax-1
         do i=imin,imax
            wrk1(i,j)=u(i,j-1,k)-2.*u(i,j,k)
     & +u(i,j+1,k)
          enddo
        enddo

        do j=jmin,jmax
         do i=imin+1,imax-1
           wrk2(i,j)=FlxV(i-1,j,k)-2.*FlxV(i,j,k)+FlxV(i+1,j,k)
          enddo
        enddo

        do j=jmin+2,jmax-1
         do i=imin+2,imax-1

            cff=FlxV(i,j,k)+FlxV(i-1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i-1,j))
            UFe(i,j)=0.25*( cff*(u(i,j,k)+u(i,j-1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j-1)
     & +min(cff,0.)*wrk1(i,j )
     & ))

            UFe(i,j)=UFe(i,j) - 0.25*( u(i,j,k)+u(i,j-1,k)
     & -0.125*(wrk1(i,j)+wrk1(i,j-1))
     & )*( FlxV(i,j,k)+FlxV(i-1,j,k)
     & -0.125*(wrk2(i,j)+wrk2(i-1,j)))

          enddo
        enddo







        do j=jmin+1,jmax
         do i=imin+1,imax-1

            wrk1(i,j)=v(i-1,j,k)-2.*v(i,j,k)
     & +v(i+1,j,k)
          enddo
        enddo

        do j=jmin+1,jmax-1
         do i=imin+1,imax

           wrk2(i,j)=FlxU(i,j-1,k)-2.*FlxU(i,j,k)+FlxU(i,j+1,k)
          enddo
        enddo

        do j=jmin+2,jmax-1
         do i=imin+2,imax-1


            cff=FlxU(i,j,k)+FlxU(i,j-1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j-1))
            VFx(i,j)=0.25*( cff*(v(i,j,k)+v(i-1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i-1,j)
     & +min(cff,0.)*wrk1(i ,j)
     & ))

            VFx(i,j)=VFx(i,j) - 0.25*( v(i,j,k)+v(i-1,j,k)
     & -0.125*(wrk1(i,j)+wrk1(i-1,j))
     & )*( FlxU(i,j,k)+FlxU(i,j-1,k)
     & -0.125*(wrk2(i,j)+wrk2(i,j-1)))

          enddo
        enddo




      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

              if (k.eq.1) then
                wrkXadv(i,j,1) = -UFx(i,j)+UFx(i-1,j)
                wrkYadv(i,j,1) = -UFe(i,j+1)+UFe(i,j)
              else
                wrkXadv(i,j,1) = wrkXadv(i,j,1) - UFx(i,j)+UFx(i-1,j)
                wrkYadv(i,j,1) = wrkYadv(i,j,1) - UFe(i,j+1)+UFe(i,j)
              endif


          enddo
        enddo

      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

              if (k.eq.1) then
                wrkXadv(i,j,2) = -VFx(i+1,j)+VFx(i,j)
                wrkYadv(i,j,2) = -VFe(i,j)+VFe(i,j-1)
              else
                wrkXadv(i,j,2) = wrkXadv(i,j,2) -VFx(i+1,j)+VFx(i,j)
                wrkYadv(i,j,2) = wrkYadv(i,j,2) -VFe(i,j)+VFe(i,j-1)
              endif

          enddo
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin,jmax
        do i=imin+1,imax
            umask(i,j) = rmask(i,j)*rmask(i-1,j)
        enddo
      enddo

      do j=jmin+1,jmax
        do i=imin,imax
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/03/15)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

            cff=0.5*(pn(i,j)+pn(i-1,j))

            wrkXadv(i,j,1)=wrkXadv(i,j,1)*cff * umask(i,j)
            wrkYadv(i,j,1)=wrkYadv(i,j,1)*cff * umask(i,j)

            cff=0.5*(pm(i,j)+pm(i,j-1))

            wrkXadv(i,j,2)=wrkXadv(i,j,2)*cff * vmask(i,j)
            wrkYadv(i,j,2)=wrkYadv(i,j,2)*cff * vmask(i,j)

          enddo
        enddo



      do j=jmin+2,jmax-2
        do i=imin+2,imax-2

           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

            vrtXadv(i,j) = (wrkXadv(i,j,2) - wrkXadv(i-1,j,2)) * cff1
     & - (wrkXadv(i,j,1) - wrkXadv(i,j-1,1)) * cff1


            vrtYadv(i,j) = (wrkYadv(i,j,2) - wrkYadv(i-1,j,2)) * cff1
     & - (wrkYadv(i,j,1) - wrkYadv(i,j-1,1)) * cff1


            adv(i,j) = vrtXadv(i,j) + vrtYadv(i,j)


          enddo
         enddo





      return
      end
# 109 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_adv_4th.F" 1






!======================================================================
!
! Compute Centered part of Advection part of the barotropic vorticity balance equation
!
!
!
! - updated 16/08/19 [add umask,vmask]
!======================================================================


      subroutine get_adv_4th (Lm,Mm,N,u,v, z_r,z_w,pm,pn,rmask
     & ,adv)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & rmask(0:Lm+1,0:Mm+1),
     & umask(0:Lm+1,0:Mm+1), vmask(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2


      real*8 wrkXadv(0:Lm+1,0:Mm+1,2),wrkYadv(0:Lm+1,0:Mm+1,2)

      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1)

      real*8 adv(1:Lm+1,1:Mm+1), vrtXadv(1:Lm+1,1:Mm+1),
     & vrtYadv(1:Lm+1,1:Mm+1)

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
# 47 "R_tools_fort_routines_gula/get_adv_4th.F" 2

      parameter (gamma=0.25)

Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,rmask
Cf2py intent(out) adv

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

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





      do k=1,N

!
! Add in horizontal advection of momentum: Compute diagonal [UFx,VFe]
! and off-diagonal [UFe,VFx] components of tensor of momentum flux
! due to horizontal advection; after that add divergence of these
! terms to r.h.s.
!





        do j=jmin,jmax
         do i=imin+2,imax-1

            wrk1(i,j)=u(i-1,j,k)-2.*u(i,j,k)
     & +u(i+1,j,k)
            wrk2(i,j)=FlxU(i-1,j,k)-2.*FlxU(i,j,k)
     & +FlxU(i+1,j,k)
          enddo
        enddo





        do j=jmin,jmax
         do i=imin+2,imax-1
# 140 "R_tools_fort_routines_gula/get_adv_4th.F"
            UFx(i,j)=0.25*( u(i,j,k)+u(i+1,j,k)
     & -0.125*(wrk1(i,j)+wrk1(i+1,j))
     & )*( FlxU(i,j,k)+FlxU(i+1,j,k)
     & -0.125*(wrk2(i,j)+wrk2(i+1,j)))


          enddo
        enddo
# 160 "R_tools_fort_routines_gula/get_adv_4th.F"
        do j=jmin+2,jmax-1
         do i=imin,imax

            wrk1(i,j)=v(i,j-1,k)-2.*v(i,j,k)+v(i,j+1,k)

            wrk2(i,j)=FlxV(i,j-1,k)-2.*FlxV(i,j,k)+FlxV(i,j+1,k)

          enddo
        enddo


        do j=jmin+2,jmax-1
         do i=imin,imax
# 183 "R_tools_fort_routines_gula/get_adv_4th.F"
            VFe(i,j)=0.25*( v(i,j,k)+v(i,j+1,k)
     & -0.125*(wrk1(i,j)+wrk1(i,j+1))
     & )*( FlxV(i,j,k)+FlxV(i,j+1,k)
     & -0.125*(wrk2(i,j)+wrk2(i,j+1)))


          enddo
        enddo
# 201 "R_tools_fort_routines_gula/get_adv_4th.F"
       do j=jmin+1,jmax-1
         do i=imin,imax
            wrk1(i,j)=u(i,j-1,k)-2.*u(i,j,k)
     & +u(i,j+1,k)
          enddo
        enddo

        do j=jmin,jmax
         do i=imin+1,imax-1
           wrk2(i,j)=FlxV(i-1,j,k)-2.*FlxV(i,j,k)+FlxV(i+1,j,k)
          enddo
        enddo

        do j=jmin+1,jmax-1
         do i=imin+1,imax-1
# 224 "R_tools_fort_routines_gula/get_adv_4th.F"
            UFe(i,j)=0.25*( u(i,j,k)+u(i,j-1,k)
     & -0.125*(wrk1(i,j)+wrk1(i,j-1))
     & )*( FlxV(i,j,k)+FlxV(i-1,j,k)
     & -0.125*(wrk2(i,j)+wrk2(i-1,j)))

          enddo
        enddo







        do j=jmin,jmax
         do i=imin+1,imax-1

            wrk1(i,j)=v(i-1,j,k)-2.*v(i,j,k)
     & +v(i+1,j,k)
          enddo
        enddo

        do j=jmin+1,jmax-1
         do i=imin,imax

           wrk2(i,j)=FlxU(i,j-1,k)-2.*FlxU(i,j,k)+FlxU(i,j+1,k)
          enddo
        enddo

        do j=jmin+1,jmax-1
         do i=imin+1,imax-1
# 264 "R_tools_fort_routines_gula/get_adv_4th.F"
            VFx(i,j)=0.25*( v(i,j,k)+v(i-1,j,k)
     & -0.125*(wrk1(i,j)+wrk1(i-1,j))
     & )*( FlxU(i,j,k)+FlxU(i,j-1,k)
     & -0.125*(wrk2(i,j)+wrk2(i,j-1)))

          enddo
        enddo




      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

              if (k.eq.1) then
                wrkXadv(i,j,1) = -UFx(i,j)+UFx(i-1,j)
                wrkYadv(i,j,1) = -UFe(i,j+1)+UFe(i,j)
              else
                wrkXadv(i,j,1) = wrkXadv(i,j,1) - UFx(i,j)+UFx(i-1,j)
                wrkYadv(i,j,1) = wrkYadv(i,j,1) - UFe(i,j+1)+UFe(i,j)
              endif


          enddo
        enddo

      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

              if (k.eq.1) then
                wrkXadv(i,j,2) = -VFx(i+1,j)+VFx(i,j)
                wrkYadv(i,j,2) = -VFe(i,j)+VFe(i,j-1)
              else
                wrkXadv(i,j,2) = wrkXadv(i,j,2) -VFx(i+1,j)+VFx(i,j)
                wrkYadv(i,j,2) = wrkYadv(i,j,2) -VFe(i,j)+VFe(i,j-1)
              endif

          enddo
        enddo
      enddo


!
! ! Divide all diagnostic terms by (pm*pn).
! ! There after the unit of these terms are :
! ! s-2
!
!
! do j=jmin+2,jmax-1
! do i=imin+2,imax-1
!
! cff=0.25*(pm(i,j)+pm(i-1,j))
! & *(pn(i,j)+pn(i-1,j))
!
!
! wrkXadv(i,j,1)=wrkXadv(i,j,1)*cff
! wrkYadv(i,j,1)=wrkYadv(i,j,1)*cff
!
!
! cff=0.25*(pm(i,j)+pm(i,j-1))
! & *(pn(i,j)+pn(i,j-1))
!
!
! wrkXadv(i,j,2)=wrkXadv(i,j,2)*cff
! wrkYadv(i,j,2)=wrkYadv(i,j,2)*cff
!
! enddo
! enddo
!
!
!
! do j=jmin+2,jmax-2
! do i=imin+2,imax-2
!
! cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
! cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
!
! vrtXadv(i,j) = (wrkXadv(i,j,2) - wrkXadv(i-1,j,2)) * cff1
! & - (wrkXadv(i,j,1) - wrkXadv(i,j-1,1)) * cff2
!
!
! vrtYadv(i,j) = (wrkYadv(i,j,2) - wrkYadv(i-1,j,2)) * cff1
! & - (wrkYadv(i,j,1) - wrkYadv(i,j-1,1)) * cff2
!
!
! adv(i,j) = vrtXadv(i,j) + vrtYadv(i,j)
!
!
! enddo
! enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin,jmax
        do i=imin+1,imax
            umask(i,j) = rmask(i,j)*rmask(i-1,j)
        enddo
      enddo

      do j=jmin+1,jmax
        do i=imin,imax
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/09/13)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

            cff=0.5*(pn(i,j)+pn(i-1,j))

            wrkXadv(i,j,1)=wrkXadv(i,j,1)*cff * umask(i,j)
            wrkYadv(i,j,1)=wrkYadv(i,j,1)*cff * umask(i,j)

            cff=0.5*(pm(i,j)+pm(i,j-1))

            wrkXadv(i,j,2)=wrkXadv(i,j,2)*cff * vmask(i,j)
            wrkYadv(i,j,2)=wrkYadv(i,j,2)*cff * vmask(i,j)

          enddo
        enddo



      do j=jmin+2,jmax-2
        do i=imin+2,imax-2

           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

            vrtXadv(i,j) = (wrkXadv(i,j,2) - wrkXadv(i-1,j,2)) * cff1
     & - (wrkXadv(i,j,1) - wrkXadv(i,j-1,1)) * cff1


            vrtYadv(i,j) = (wrkYadv(i,j,2) - wrkYadv(i-1,j,2)) * cff1
     & - (wrkYadv(i,j,1) - wrkYadv(i,j-1,1)) * cff1


            adv(i,j) = vrtXadv(i,j) + vrtYadv(i,j)


          enddo
         enddo




      return
      end
# 111 "R_tools_fort_gula.F" 2


# 1 "R_tools_fort_routines_gula/get_fwb.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute bottom vorticity stretching (r'$- f \vec{u_b} . \vec{\nabla} h $')
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






      subroutine get_fwb(Lm,Mm,N,u,v, z_r,z_w,pm,pn,f
     & ,fwb)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N), f(0:Lm+1,0:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4

      real*8 Wxi(1:Lm+1,0:Mm+1),Weta(0:Lm+1,1:Mm+1)

      real*8 fwb(0:Lm+1,0:Mm+1)

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
# 36 "R_tools_fort_routines_gula/get_fwb.F" 2




Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,f
Cf2py intent(out) fwb

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



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!
! Compute and add contributions due to (quasi-)horizontal motions
! along S=const surfaces by multiplying horizontal velocity
! components by slops S-coordinate surfaces:
!

        do j=jmin,jmax
          do i=imin+1,imax
            Wxi(i,j)=u(i,j,1)*(z_r(i,j,1)-z_r(i-1,j,1))
     & *(pm(i,j)+pm(i-1,j))
          enddo
        enddo
        do j=jmin+1,jmax
          do i=imin,imax
            Weta(i,j)=v(i,j,1)*(z_r(i,j,1)-z_r(i,j-1,1))
     & *(pn(i,j)+pn(i,j-1))
          enddo
        enddo
        do j=jmin+1,jmax-1
          do i=imin+1,imax-1
            fwb(i,j)=f(i,j)*0.25*(Wxi(i,j)+Wxi(i+1,j)
     & +Weta(i,j)+Weta(i,j+1))
          enddo
        enddo



! do j=jmin,jmax
! do i=imin+1,imax
! Wxi(i,j)=u(i,j,N)*(z_r(i,j,N)-z_r(i-1,j,N))
! & *(pm(i,j)+pm(i-1,j))
! enddo
! enddo
! do j=jmin+1,jmax
! do i=imin,imax
! Weta(i,j)=v(i,j,N)*(z_r(i,j,N)-z_r(i,j-1,N))
! & *(pn(i,j)+pn(i,j-1))
! enddo
! enddo
! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
! fws(i,j)=-f(i,j)*0.25*(Wxi(i,j)+Wxi(i+1,j)
! & +Weta(i,j)+Weta(i,j+1))
! enddo
! enddo






      return
      end
# 114 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_fws.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute bottom and surface vorticity stretching
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






      subroutine get_fws(Lm,Mm,N,u,v, z_r,z_w,pm,pn,f
     & ,fws)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N), f(0:Lm+1,0:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4

      real*8 Wxi(1:Lm+1,0:Mm+1),Weta(0:Lm+1,1:Mm+1)

      real*8 fws(0:Lm+1,0:Mm+1)

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
# 36 "R_tools_fort_routines_gula/get_fws.F" 2




Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,f
Cf2py intent(out) fws

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



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!
! Compute and add contributions due to (quasi-)horizontal motions
! along S=const surfaces by multiplying horizontal velocity
! components by slops S-coordinate surfaces:
!

! do j=jmin,jmax
! do i=imin+1,imax
! Wxi(i,j)=u(i,j,1)*(z_r(i,j,1)-z_r(i-1,j,1))
! & *(pm(i,j)+pm(i-1,j))
! enddo
! enddo
! do j=jmin+1,jmax
! do i=imin,imax
! Weta(i,j)=v(i,j,1)*(z_r(i,j,1)-z_r(i,j-1,1))
! & *(pn(i,j)+pn(i,j-1))
! enddo
! enddo
! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
! fwb(i,j)=-f(i,j)*0.25*(Wxi(i,j)+Wxi(i+1,j)
! & +Weta(i,j)+Weta(i,j+1))
! enddo
! enddo



        do j=jmin,jmax
          do i=imin+1,imax
            Wxi(i,j)=u(i,j,N)*(z_r(i,j,N)-z_r(i-1,j,N))
     & *(pm(i,j)+pm(i-1,j))
          enddo
        enddo
        do j=jmin+1,jmax
          do i=imin,imax
            Weta(i,j)=v(i,j,N)*(z_r(i,j,N)-z_r(i,j-1,N))
     & *(pn(i,j)+pn(i,j-1))
          enddo
        enddo
        do j=jmin+1,jmax-1
          do i=imin+1,imax-1
            fws(i,j)=f(i,j)*0.25*(Wxi(i,j)+Wxi(i+1,j)
     & +Weta(i,j)+Weta(i,j+1))
          enddo
        enddo






      return
      end
# 116 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_fdivub.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute difference between:
!
! -f wb (where wb is vertical velocity)
! and
! bottom vorticity stretching (r'$- f \vec{u_b} . \vec{\nabla} h $')
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






      subroutine get_fdivub(Lm,Mm,N,u,v, z_r,z_w,pm,pn,f
     & ,fdivu)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N), f(0:Lm+1,0:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4

      real*8 Wrk(0:Lm+1,0:N),Wvlc(0:Lm+1,0:Mm+1,N),
     & Wxi(1:Lm+1,0:Mm+1),Weta(0:Lm+1,1:Mm+1)

      real*8 fdivu(0:Lm+1,0:Mm+1)

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
# 42 "R_tools_fort_routines_gula/get_fdivub.F" 2




Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,f
Cf2py intent(out) fdivu

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
          fdivu(i,j)= f(i,j)*( -0.125*Wrk(i,2) +0.75*Wrk(i,1)
     & +0.375*Wrk(i,0) )
        enddo
      enddo
!
# 166 "R_tools_fort_routines_gula/get_fdivub.F"
      return
      end
# 118 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_fwdivub.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute difference between:
!
! -f wb (where wb is vertical velocity)
! and
! bottom vorticity stretching (r'$- f \vec{u_b} . \vec{\nabla} h $')
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






      subroutine get_fwdivub(Lm,Mm,N,u,v, z_r,z_w,pm,pn,f
     & ,fdivu)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N), f(0:Lm+1,0:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4

      real*8 Wrk(0:Lm+1,0:N),Wvlc(0:Lm+1,0:Mm+1,N),
     & Wxi(1:Lm+1,0:Mm+1),Weta(0:Lm+1,1:Mm+1)

      real*8 fdivu(0:Lm+1,0:Mm+1)

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
# 42 "R_tools_fort_routines_gula/get_fwdivub.F" 2




Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,f
Cf2py intent(out) fdivu

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
          fdivu(i,j)= f(i,j)*( -0.125*Wrk(i,2) +0.75*Wrk(i,1)
     & +0.375*Wrk(i,0) )
        enddo
      enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        do j=jmin,jmax
          do i=imin+1,imax
            Wxi(i,j)=u(i,j,1)*(z_r(i,j,1)-z_r(i-1,j,1))
     & *(pm(i,j)+pm(i-1,j))
          enddo
        enddo
        do j=jmin+1,jmax
          do i=imin,imax
            Weta(i,j)=v(i,j,1)*(z_r(i,j,1)-z_r(i,j-1,1))
     & *(pn(i,j)+pn(i,j-1))
          enddo
        enddo
        do j=jmin+1,jmax-1
          do i=imin+1,imax-1
            fdivu(i,j)=fdivu(i,j)+f(i,j)*0.25*(Wxi(i,j)+Wxi(i+1,j)
     & +Weta(i,j)+Weta(i,j+1))
          enddo
        enddo
# 178 "R_tools_fort_routines_gula/get_fwdivub.F"
      return
      end
# 120 "R_tools_fort_gula.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute Barotropic equation components for vertically averaged flow
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 1 "R_tools_fort_routines_gula/get_bpt_mean.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute Bottom Pressure torque J(Pb,H)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






      subroutine get_bpt_mean (Lm,Mm,N,T,S, z_r,z_w,rho0,pm,pn
     & ,bpt)



      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV



      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & rho1(0:Lm+1,0:Mm+1,N), qp1(0:Lm+1,0:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,0:N),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & Tt,Ts,sqrtTs, rho0, K0, dr00,
     & cff ,cff1, cff2, cfr, HalfGRho, GRho,
     & var1, var2,var3, var4

      real*8 bpt(1:Lm+1,1:Mm+1)


      real*8 P(0:Lm+1,0:Mm+1,N),
     & ru(1:Lm+1,0:Mm+1), rv(0:Lm+1,1:Mm+1),
     & rho(0:Lm+1,0:Mm+1,N), dpth,
     & dR(0:Lm+1,N), dZ(0:Lm+1,N),
     & FC(0:Lm+2,0:Mm+2), dZx(0:Lm+1,0:Mm+1),
     & rx(0:Lm+2,0:Mm+2), dRx(0:Lm+1,0:Mm+1)

      real*8, parameter :: OneFifth=0.2, OneTwelfth=1./12., epsil=0.

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
     & qp2=0.0000172


      real rho1_0, K0_Duk

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
# 63 "R_tools_fort_routines_gula/get_bpt_mean.F" 2



Cf2py intent(in) Lm,Mm,N, T,S,z_r,z_w,rho0,pm,pn
Cf2py intent(out) bpt




!
! A non-conservative Density-Jacobian scheme using cubic polynomial
! fits for rho and z_r as functions of nondimensianal coordinates xi,
! eta, and s (basically their respective fortran indices). The cubic
! polynomials are constructed by specifying first derivatives of
! interpolated fields on co-located (non-staggered) grid. These
! derivatives are computed using harmonic (rather that algebraic)
! averaging of elementary differences, which guarantees monotonicity
! of the resultant interpolant.
!
! In the code below, if CPP-switch is defined, the Equation
! of State (EOS) is assumed to have form
!
! rho(T,S,z) = rho1(T,S) + qp1(T,S)*dpth*[1.-qp2*dpth]
!
! where rho1 is potential density at 1 atm and qp1 is compressibility
! coefficient, which does not depend on z, and dpth=zeta-z, and qp2
! is just a constant. In this case
!
! d rho d rho1 d qp1 d z
! ------- = ------ + ----- *dpth*[..] - qp1*[1.-2.*qp2*dpth]*------
! d s,x d s,x d s,x d s,x
!
! |<--- adiabatic part --->| |<--- compressible part --->|
!
! where the first two terms constitute "adiabatic derivative" of
! density, which is subject to harmonic averaging, while the last
! term is added in later. This approach quarantees that density
! profile reconstructed by cubic polynomial maintains its positive
! statification in physical sense as long as discrete values of
! density are positively stratified.
!
! This scheme retains exact antisymmetry J(rho,z_r)=-J(z_r,rho)
! [with the exception of harmonic averaging algorithm in the case
! when CPP-switch is defined, see above]. If parameter
! OneFifth (see above) is set to zero, the scheme becomes identical
! to standard Jacobian.
!
! NOTE: This routine is an alternative form of prsgrd32 and it
! produces results identical to that if its prototype.
!


        istr=0
        istrU=1
        iend=Lm+1
        jstr=0
        jstrV=1
        jend=Mm+1

        imin=istrU
        imax=iend
        jmin=jstrV
        jmax=jend


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE DENSITY
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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
# 159 "R_tools_fort_routines_gula/get_bpt_mean.F"
      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

      do j=jmin,jmax


!---------------------------------------------------------------------------------------
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


!---------------------------------------------------------------------------------------

            Hz(i,j,k)=(z_w(i,j,k)-z_w(i,j,k-1))
     & / (z_w(i,j,N) - z_w(i,j,0))





          enddo
        enddo
      enddo ! <-- j



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!
! Preliminary step (same for XI- and ETA-components:
!------------ ---- ----- --- --- --- ---------------
!
      GRho=g/rho0
      HalfGRho=0.5*GRho

      do j=jstrV-1,jend
        do k=1,N-1
          do i=istrU-1,iend
            dZ(i,k)=z_r(i,j,k+1)-z_r(i,j,k)

            dpth=z_w(i,j,N)-0.5*(z_r(i,j,k+1)+z_r(i,j,k))

            dR(i,k)=rho1(i,j,k+1)-rho1(i,j,k) ! Elementary
     & +(qp1(i,j,k+1)-qp1(i,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ! difference



          enddo
        enddo
        do i=istrU-1,iend
          dR(i,N)=dR(i,N-1)
          dR(i,0)=dR(i,1)
          dZ(i,N)=dZ(i,N-1)
          dZ(i,0)=dZ(i,1)
        enddo
        do k=N,1,-1 !--> irreversible
          do i=istrU-1,iend
            cff=2.*dZ(i,k)*dZ(i,k-1)
            dZ(i,k)=cff/(dZ(i,k)+dZ(i,k-1))

            cfr=2.*dR(i,k)*dR(i,k-1)
            if (cfr.gt.epsil) then
              dR(i,k)=cfr/(dR(i,k)+dR(i,k-1))
            else
              dR(i,k)=0.
            endif

            dpth=z_w(i,j,N)-z_r(i,j,k)
            dR(i,k)=dR(i,k) -qp1(i,j,k)*dZ(i,k)*(1.-2.*qp2*dpth)
            rho(i,j,k)=rho1(i,j,k) +qp1(i,j,k)*dpth*(1.-qp2*dpth)

          enddo
        enddo
        do i=istrU-1,iend
          P(i,j,N)=g*z_w(i,j,N) + GRho*( rho(i,j,N)
     & +0.5*(rho(i,j,N)-rho(i,j,N-1))*(z_w(i,j,N)-z_r(i,j,N))
     & /(z_r(i,j,N)-z_r(i,j,N-1)) )*(z_w(i,j,N)-z_r(i,j,N))
        enddo
        do k=N-1,1,-1
          do i=istrU-1,iend
            P(i,j,k)=P(i,j,k+1)+HalfGRho*( (rho(i,j,k+1)+rho(i,j,k))
     & *(z_r(i,j,k+1)-z_r(i,j,k))

     & -OneFifth*( (dR(i,k+1)-dR(i,k))*( z_r(i,j,k+1)-z_r(i,j,k)
     & -OneTwelfth*(dZ(i,k+1)+dZ(i,k)) )

     & -(dZ(i,k+1)-dZ(i,k))*( rho(i,j,k+1)-rho(i,j,k)
     & -OneTwelfth*(dR(i,k+1)+dR(i,k)) )
     & ))
          enddo
        enddo
      enddo !<-- j




        ru = 0.
        rv = 0.



!
! Compute XI-component of pressure gradient term:
!-------- ------------ -- -------- -------- -----
!
      do k=N,1,-1
        do j=jstr,jend
          do i=imin,imax
            FC(i,j)=(z_r(i,j,k)-z_r(i-1,j,k))

            dpth=0.5*( z_w(i,j,N)+z_w(i-1,j,N)
     & -z_r(i,j,k)-z_r(i-1,j,k))

            rx(i,j)=( rho1(i,j,k)-rho1(i-1,j,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i-1,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference




          enddo
        enddo


        if (istr.eq.1) then ! Extrapolate elementary
          do j=jstr,jend ! differences near physical
            FC(imin-1,j)=FC(imin,j) ! boundaries to compencate.
            rx(imin-1,j)=rx(imin,j) ! for reduced loop ranges.
          enddo
        endif
        if (iend.eq.Lm) then
          do j=jstr,jend
            FC(imax+1,j)=FC(imax,j)
            rx(imax+1,j)=rx(imax,j)
          enddo
        endif


        do j=jstr,jend
          do i=istrU-1,iend
            cff=2.*FC(i,j)*FC(i+1,j)
            if (cff.gt.epsil) then
              dZx(i,j)=cff/(FC(i,j)+FC(i+1,j))
            else
              dZx(i,j)=0.
            endif

            cfr=2.*rx(i,j)*rx(i+1,j)
            if (cfr.gt.epsil) then
              dRx(i,j)=cfr/(rx(i,j)+rx(i+1,j))
            else
              dRx(i,j)=0.
            endif

            dRx(i,j)=dRx(i,j) -qp1(i,j,k)*dZx(i,j)
     & *(1.-2.*qp2*(z_w(i,j,N)-z_r(i,j,k)))

          enddo !--> discard FC, rx

          do i=istrU,iend
            ru(i,j)=ru(i,j) +
     & 0.5*(Hz(i,j,k)+Hz(i-1,j,k))*2./(pn(i,j)+pn(i-1,j))*(
     & P(i-1,j,k)-P(i,j,k)-HalfGRho*(

     & (rho(i,j,k)+rho(i-1,j,k))*(z_r(i,j,k)-z_r(i-1,j,k))

     & -OneFifth*( (dRx(i,j)-dRx(i-1,j))*( z_r(i,j,k)-z_r(i-1,j,k)
     & -OneTwelfth*(dZx(i,j)+dZx(i-1,j)) )

     & -(dZx(i,j)-dZx(i-1,j))*( rho(i,j,k)-rho(i-1,j,k)
     & -OneTwelfth*(dRx(i,j)+dRx(i-1,j)) )
     & )))
          enddo
        enddo
!
! ETA-component of pressure gradient term:
!-------------- -- -------- -------- -----
!
        do j=jmin,jmax
          do i=istr,iend
            FC(i,j)=(z_r(i,j,k)-z_r(i,j-1,k))

            dpth=0.5*( z_w(i,j,N)+z_w(i,j-1,N)
     & -z_r(i,j,k)-z_r(i,j-1,k))

            rx(i,j)=( rho1(i,j,k)-rho1(i,j-1,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i,j-1,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference



          enddo
        enddo


        if (jstr.eq.1) then
          do i=istr,iend
            FC(i,jmin-1)=FC(i,jmin)
            rx(i,jmin-1)=rx(i,jmin)

          enddo
        endif
        if (jend.eq.Mm) then
          do i=istr,iend
            FC(i,jmax+1)=FC(i,jmax)
            rx(i,jmax+1)=rx(i,jmax)
          enddo
        endif


        do j=jstrV-1,jend
          do i=istr,iend
            cff=2.*FC(i,j)*FC(i,j+1)
            if (cff.gt.epsil) then
c** if ((FC(i,j).gt.0. .and. FC(i,j+1).gt.0.) .or.
c** & (FC(i,j).lt.0. .and. FC(i,j+1).lt.0.)) then
              dZx(i,j)=cff/(FC(i,j)+FC(i,j+1))
            else
              dZx(i,j)=0.
            endif

            cfr=2.*rx(i,j)*rx(i,j+1)
            if (cfr.gt.epsil) then
c** if ((rx(i,j).gt.0. .and. rx(i,j+1).gt.0.) .or.
c** & (rx(i,j).lt.0. .and. rx(i,j+1).lt.0.)) then
              dRx(i,j)=cfr/(rx(i,j)+rx(i,j+1))
            else
              dRx(i,j)=0.
            endif

            dRx(i,j)=dRx(i,j) -qp1(i,j,k)*dZx(i,j)
     & *(1.-2.*qp2*(z_w(i,j,N)-z_r(i,j,k)))

          enddo !--> discard FC, rx

          if (j.ge.jstrV) then
            do i=istr,iend

            rv(i,j)=rv(i,j) +
     & 0.5*(Hz(i,j,k)+Hz(i,j-1,k))*2./(pm(i,j)+pm(i,j-1))*(
     & P(i,j-1,k)-P(i,j,k) -HalfGRho*(

     & (rho(i,j,k)+rho(i,j-1,k))*(z_r(i,j,k)-z_r(i,j-1,k))

     & -OneFifth*( (dRx(i,j)-dRx(i,j-1))*( z_r(i,j,k)-z_r(i,j-1,k)
     & -OneTwelfth*(dZx(i,j)+dZx(i,j-1)) )

     & -(dZx(i,j)-dZx(i,j-1))*( rho(i,j,k)-rho(i,j-1,k)
     & -OneTwelfth*(dRx(i,j)+dRx(i,j-1)) )
     & )))

            enddo
          endif
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Compute rotational
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!
!
! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
!
!
! ru(i,j) = ru(i,j)*0.5*(pn(i,j)+pn(i-1,j))
! & *0.5*(pm(i,j)+pm(i-1,j))
!
!
! rv(i,j) = rv(i,j)*0.5*(pm(i,j)+pm(i,j-1))
! & *0.5*(pn(i,j)+pn(i,j-1))
!
!
!
! cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
! cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
!
! bpt(i,j) = (rv(i,j) - rv(i-1,j)) * cff1
! & - (ru(i,j) - ru(i,j-1)) * cff2
! enddo
! enddo
!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/09/13)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin+1,jmax-1
        do i=imin+1,imax-1

            cff=0.5*(pn(i,j)+pn(i-1,j))
            ru(i,j)=ru(i,j)*cff

            cff=0.5*(pm(i,j)+pm(i,j-1))
            rv(i,j)=rv(i,j)*cff


           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))


             bpt(i,j) = (rv(i,j) - rv(i-1,j)) * cff1
     & - (ru(i,j) - ru(i,j-1)) * cff1

        enddo
       enddo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






      return
      end
# 126 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_vortplantot_sol2_mean.F" 1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute Coriolis part of the barotropic vorticity balance equation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      subroutine get_vortplantot_sol2_mean (Lm,Mm,N,u,v, z_r,z_w,pm,pn,f
     & ,vrtCor)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & f(0:Lm+1,0:Mm+1), fomn(0:Lm+1,0:Mm+1),
     & dmde(0:Lm+1,0:Mm+1), dndx(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2


      real*8 wrkCor(0:Lm+1,0:Mm+1,2)

      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1)

      real*8 vrtCor(1:Lm+1,1:Mm+1)


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
# 40 "R_tools_fort_routines_gula/get_vortplantot_sol2_mean.F" 2

      parameter (gamma=0.25)

Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,f
Cf2py intent(out) vrtCor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jmin,jmax
        do i=imin,imax
          do k=1,N,+1
           Hz(i,j,k) = (z_w(i,j,k) - z_w(i,j,k-1))
     & / (z_w(i,j,N) - z_w(i,j,0))
          enddo
           fomn(i,j)=f(i,j)/(pm(i,j)*pn(i,j))
        enddo
      enddo


      do j=jmin+1,jmax-1
        do i=imin+1,imax-1
            dmde(i,j) = 0.5/pm(i,j+1)-0.5/pm(i,j-1)
            dndx(i,j) = 0.5/pn(i+1,j)-0.5/pn(i-1,j)
         enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




      do k=1,N

        do j=jmin+1,jmax-1
          do i=imin+1,imax-1
            cff=0.5*Hz(i,j,k)*(
     & fomn(i,j)
     & +0.5*( (v(i,j,k)+v(i,j+1,k))*dndx(i,j)
     & -(u(i,j,k)+u(i+1,j,k))*dmde(i,j))
     & )
            UFx(i,j)=cff*(v(i,j,k)+v(i,j+1,k))
            VFe(i,j)=cff*(u(i,j,k)+u(i+1,j,k))
          enddo
        enddo

        do j=jmin+1,jmax-1
          do i=imin+1,imax-1


              if (k.eq.1) then
                wrkCor(i,j,1) = 0.5*(UFx(i,j)+UFx(i-1,j))
              else
                wrkCor(i,j,1) = wrkCor(i,j,1) +
     & 0.5*(UFx(i,j)+UFx(i-1,j))
              endif

          enddo
        enddo
        do j=jmin+1,jmax-1
          do i=imin+1,imax-1


              if (k.eq.1) then
                wrkCor(i,j,2) = -0.5*(VFe(i,j)+VFe(i,j-1))
              else
                wrkCor(i,j,2) = wrkCor(i,j,2)
     & -0.5*(VFe(i,j)+VFe(i,j-1))
              endif

          enddo
        enddo


      enddo





! Divide all diagnostic terms by (pm*pn).
! There after the unit of these terms are :
! s-2


! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
!
! cff=0.25*(pm(i,j)+pm(i-1,j))
! & *(pn(i,j)+pn(i-1,j))
!
!
! wrkCor(i,j,1)=wrkCor(i,j,1)*cff
!
!
! cff=0.25*(pm(i,j)+pm(i,j-1))
! & *(pn(i,j)+pn(i,j-1))
!
!
! wrkCor(i,j,2)=wrkCor(i,j,2)*cff
!
!
! enddo
! ! enddo
!
!
!
! do j=jmin+1,jmax-1
! do i=imin+1,imax-1
!
! cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
! cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
!
! vrtCor(i,j) = (wrkCor(i,j,2) - wrkCor(i-1,j,2)) * cff1
! & - (wrkCor(i,j,1) - wrkCor(i,j-1,1)) * cff2
!
!
! enddo
! enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/09/13)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin+1,jmax-1
        do i=imin+1,imax-1

            cff=0.5*(pn(i,j)+pn(i-1,j))
            wrkCor(i,j,1)=wrkCor(i,j,1)*cff

            cff=0.5*(pm(i,j)+pm(i,j-1))
            wrkCor(i,j,2)=wrkCor(i,j,2)*cff

        enddo
      enddo

      do j=jmin+1,jmax-1
        do i=imin+1,imax-1

           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

           cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
     & * 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))

            vrtCor(i,j) = (wrkCor(i,j,2) - wrkCor(i-1,j,2)) * cff1
     & - (wrkCor(i,j,1) - wrkCor(i,j-1,1)) * cff2

        enddo
       enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      return
      end
# 128 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_adv_sol2_mean.F" 1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute Advective part of the barotropic vorticity balance equation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      subroutine get_adv_sol2_mean (Lm,Mm,N,u,v, z_r,z_w,pm,pn
     & ,adv)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2


      real*8 wrkXadv(0:Lm+1,0:Mm+1,2),wrkYadv(0:Lm+1,0:Mm+1,2)

      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1)

      real*8 adv(1:Lm+1,1:Mm+1), vrtXadv(1:Lm+1,1:Mm+1),
     & vrtYadv(1:Lm+1,1:Mm+1)

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
# 39 "R_tools_fort_routines_gula/get_adv_sol2_mean.F" 2

      parameter (gamma=0.25)

Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn
Cf2py intent(out) adv

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jmin,jmax
        do i=imin,imax
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
     & / (z_w(i,j,N) - z_w(i,j,0))
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





      do k=1,N

!
! Add in horizontal advection of momentum: Compute diagonal [UFx,VFe]
! and off-diagonal [UFe,VFx] components of tensor of momentum flux
! due to horizontal advection; after that add divergence of these
! terms to r.h.s.
!





        do j=jmin,jmax
         do i=imin+2,imax-1

            wrk1(i,j)=u(i-1,j,k)-2.*u(i,j,k)
     & +u(i+1,j,k)
            wrk2(i,j)=FlxU(i-1,j,k)-2.*FlxU(i,j,k)
     & +FlxU(i+1,j,k)
          enddo
        enddo



        do j=jmin,jmax
         do i=imin+2,imax-2


            cff=FlxU(i,j,k)+FlxU(i+1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i+1,j))
            UFx(i,j)=0.25*( cff*(u(i,j,k)+u(i+1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i ,j)
     & +min(cff,0.)*wrk1(i+1,j)
     & ))

          enddo
        enddo
# 142 "R_tools_fort_routines_gula/get_adv_sol2_mean.F"
        do j=jmin+2,jmax-1
         do i=imin,imax

            wrk1(i,j)=v(i,j-1,k)-2.*v(i,j,k)+v(i,j+1,k)
            wrk2(i,j)=FlxV(i,j-1,k)-2.*FlxV(i,j,k)+FlxV(i,j+1,k)
          enddo
        enddo


        do j=jmin+2,jmax-2
         do i=imin,imax


            cff=FlxV(i,j,k)+FlxV(i,j+1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j+1))
            VFe(i,j)=0.25*( cff*(v(i,j,k)+v(i,j+1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j )
     & +min(cff,0.)*wrk1(i,j+1)
     & ))

          enddo
        enddo
# 172 "R_tools_fort_routines_gula/get_adv_sol2_mean.F"
       do j=jmin+1,jmax-1
         do i=imin,imax
            wrk1(i,j)=u(i,j-1,k)-2.*u(i,j,k)
     & +u(i,j+1,k)
          enddo
        enddo

        do j=jmin,jmax
         do i=imin+1,imax-1
           wrk2(i,j)=FlxV(i-1,j,k)-2.*FlxV(i,j,k)+FlxV(i+1,j,k)
          enddo
        enddo

        do j=jmin+2,jmax-1
         do i=imin+2,imax-1

            cff=FlxV(i,j,k)+FlxV(i-1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i-1,j))
            UFe(i,j)=0.25*( cff*(u(i,j,k)+u(i,j-1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j-1)
     & +min(cff,0.)*wrk1(i,j )
     & ))

          enddo
        enddo







        do j=jmin+1,jmax
         do i=imin+1,imax-1

            wrk1(i,j)=v(i-1,j,k)-2.*v(i,j,k)
     & +v(i+1,j,k)
          enddo
        enddo


        do j=jmin+1,jmax-1
         do i=imin+1,imax

           wrk2(i,j)=FlxU(i,j-1,k)-2.*FlxU(i,j,k)+FlxU(i,j+1,k)
          enddo
        enddo

        do j=jmin+2,jmax-1
         do i=imin+2,imax-1

            cff=FlxU(i,j,k)+FlxU(i,j-1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j-1))
            VFx(i,j)=0.25*( cff*(v(i,j,k)+v(i-1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i-1,j)
     & +min(cff,0.)*wrk1(i ,j)
     & ))


          enddo
        enddo




      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

              if (k.eq.1) then
                wrkXadv(i,j,1) = -UFx(i,j)+UFx(i-1,j)
                wrkYadv(i,j,1) = -UFe(i,j+1)+UFe(i,j)
              else
                wrkXadv(i,j,1) = wrkXadv(i,j,1) - UFx(i,j)+UFx(i-1,j)
                wrkYadv(i,j,1) = wrkYadv(i,j,1) - UFe(i,j+1)+UFe(i,j)
              endif


          enddo
        enddo

      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

              if (k.eq.1) then
                wrkXadv(i,j,2) = -VFx(i+1,j)+VFx(i,j)
                wrkYadv(i,j,2) = -VFe(i,j)+VFe(i,j-1)
              else
                wrkXadv(i,j,2) = wrkXadv(i,j,2) -VFx(i+1,j)+VFx(i,j)
                wrkYadv(i,j,2) = wrkYadv(i,j,2) -VFe(i,j)+VFe(i,j-1)
              endif

          enddo
        enddo
      enddo





! ! Divide all diagnostic terms by (pm*pn).
! ! There after the unit of these terms are :
! ! s-2
!
!
! do j=jmin+2,jmax-1
! do i=imin+2,imax-1
!
! cff=0.25*(pm(i,j)+pm(i-1,j))
! & *(pn(i,j)+pn(i-1,j))
!
!
! wrkXadv(i,j,1)=wrkXadv(i,j,1)*cff
! wrkYadv(i,j,1)=wrkYadv(i,j,1)*cff
!
!
! cff=0.25*(pm(i,j)+pm(i,j-1))
! & *(pn(i,j)+pn(i,j-1))
!
!
! wrkXadv(i,j,2)=wrkXadv(i,j,2)*cff
! wrkYadv(i,j,2)=wrkYadv(i,j,2)*cff
!
! enddo
! enddo
!
!
!
! do j=jmin+2,jmax-2
! do i=imin+2,imax-2
!
! cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
! cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
!
! vrtXadv(i,j) = (wrkXadv(i,j,2) - wrkXadv(i-1,j,2)) * cff1
! & - (wrkXadv(i,j,1) - wrkXadv(i,j-1,1)) * cff2
!
!
! vrtYadv(i,j) = (wrkYadv(i,j,2) - wrkYadv(i-1,j,2)) * cff1
! & - (wrkYadv(i,j,1) - wrkYadv(i,j-1,1)) * cff2
!
!
! adv(i,j) = vrtXadv(i,j) + vrtYadv(i,j)
!
!
! enddo
! enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/09/13)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

            cff=0.5*(pn(i,j)+pn(i-1,j))

            wrkXadv(i,j,1)=wrkXadv(i,j,1)*cff
            wrkYadv(i,j,1)=wrkYadv(i,j,1)*cff

            cff=0.5*(pm(i,j)+pm(i,j-1))

            wrkXadv(i,j,2)=wrkXadv(i,j,2)*cff
            wrkYadv(i,j,2)=wrkYadv(i,j,2)*cff

          enddo
        enddo



      do j=jmin+2,jmax-2
        do i=imin+2,imax-2

           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

            vrtXadv(i,j) = (wrkXadv(i,j,2) - wrkXadv(i-1,j,2)) * cff1
     & - (wrkXadv(i,j,1) - wrkXadv(i,j-1,1)) * cff1


            vrtYadv(i,j) = (wrkYadv(i,j,2) - wrkYadv(i-1,j,2)) * cff1
     & - (wrkYadv(i,j,1) - wrkYadv(i,j-1,1)) * cff1


            adv(i,j) = vrtXadv(i,j) + vrtYadv(i,j)


          enddo
         enddo





      return
      end
# 130 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_adv_sol3_mean.F" 1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute Advective part of the barotropic vorticity balance equation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      subroutine get_adv_sol3_mean (Lm,Mm,N,u,v, z_r,z_w,pm,pn
     & ,adv)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2


      real*8 wrkXadv(0:Lm+1,0:Mm+1,2),wrkYadv(0:Lm+1,0:Mm+1,2)

      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1)

      real*8 adv(1:Lm+1,1:Mm+1), vrtXadv(1:Lm+1,1:Mm+1),
     & vrtYadv(1:Lm+1,1:Mm+1)

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
# 39 "R_tools_fort_routines_gula/get_adv_sol3_mean.F" 2

      parameter (gamma=0.25)

Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn
Cf2py intent(out) adv

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

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





      do k=1,N

!
! Add in horizontal advection of momentum: Compute diagonal [UFx,VFe]
! and off-diagonal [UFe,VFx] components of tensor of momentum flux
! due to horizontal advection; after that add divergence of these
! terms to r.h.s.
!





        do j=jmin,jmax
         do i=imin+2,imax-1

            wrk1(i,j)=u(i-1,j,k)-2.*u(i,j,k)
     & +u(i+1,j,k)
            wrk2(i,j)=FlxU(i-1,j,k)-2.*FlxU(i,j,k)
     & +FlxU(i+1,j,k)
          enddo
        enddo



        do j=jmin,jmax
         do i=imin+2,imax-2


            cff=FlxU(i,j,k)+FlxU(i+1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i+1,j))
            UFx(i,j)=0.25*( cff*(u(i,j,k)+u(i+1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i ,j)
     & +min(cff,0.)*wrk1(i+1,j)
     & ))

          enddo
        enddo
# 141 "R_tools_fort_routines_gula/get_adv_sol3_mean.F"
        do j=jmin+2,jmax-1
         do i=imin,imax

            wrk1(i,j)=v(i,j-1,k)-2.*v(i,j,k)+v(i,j+1,k)
            wrk2(i,j)=FlxV(i,j-1,k)-2.*FlxV(i,j,k)+FlxV(i,j+1,k)
          enddo
        enddo


        do j=jmin+2,jmax-2
         do i=imin,imax


            cff=FlxV(i,j,k)+FlxV(i,j+1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j+1))
            VFe(i,j)=0.25*( cff*(v(i,j,k)+v(i,j+1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j )
     & +min(cff,0.)*wrk1(i,j+1)
     & ))

          enddo
        enddo
# 171 "R_tools_fort_routines_gula/get_adv_sol3_mean.F"
       do j=jmin+1,jmax-1
         do i=imin,imax
            wrk1(i,j)=u(i,j-1,k)-2.*u(i,j,k)
     & +u(i,j+1,k)
          enddo
        enddo

        do j=jmin,jmax
         do i=imin+1,imax-1
           wrk2(i,j)=FlxV(i-1,j,k)-2.*FlxV(i,j,k)+FlxV(i+1,j,k)
          enddo
        enddo

        do j=jmin+2,jmax-1
         do i=imin+2,imax-1

            cff=FlxV(i,j,k)+FlxV(i-1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i-1,j))
            UFe(i,j)=0.25*( cff*(u(i,j,k)+u(i,j-1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j-1)
     & +min(cff,0.)*wrk1(i,j )
     & ))

          enddo
        enddo







        do j=jmin+1,jmax
         do i=imin+1,imax-1

            wrk1(i,j)=v(i-1,j,k)-2.*v(i,j,k)
     & +v(i+1,j,k)
          enddo
        enddo


        do j=jmin+1,jmax-1
         do i=imin+1,imax

           wrk2(i,j)=FlxU(i,j-1,k)-2.*FlxU(i,j,k)+FlxU(i,j+1,k)
          enddo
        enddo

        do j=jmin+2,jmax-1
         do i=imin+2,imax-1

            cff=FlxU(i,j,k)+FlxU(i,j-1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j-1))
            VFx(i,j)=0.25*( cff*(v(i,j,k)+v(i-1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i-1,j)
     & +min(cff,0.)*wrk1(i ,j)
     & ))


          enddo
        enddo




      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

              if (k.eq.1) then
                wrkXadv(i,j,1) = -UFx(i,j)+UFx(i-1,j)
                wrkYadv(i,j,1) = -UFe(i,j+1)+UFe(i,j)
              else
                wrkXadv(i,j,1) = wrkXadv(i,j,1) - UFx(i,j)+UFx(i-1,j)
                wrkYadv(i,j,1) = wrkYadv(i,j,1) - UFe(i,j+1)+UFe(i,j)
              endif


          enddo
        enddo

      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

              if (k.eq.1) then
                wrkXadv(i,j,2) = -VFx(i+1,j)+VFx(i,j)
                wrkYadv(i,j,2) = -VFe(i,j)+VFe(i,j-1)
              else
                wrkXadv(i,j,2) = wrkXadv(i,j,2) -VFx(i+1,j)+VFx(i,j)
                wrkYadv(i,j,2) = wrkYadv(i,j,2) -VFe(i,j)+VFe(i,j-1)
              endif

          enddo
        enddo
      enddo





! ! Divide all diagnostic terms by (pm*pn).
! ! There after the unit of these terms are :
! ! s-2
!
!
! do j=jmin+2,jmax-1
! do i=imin+2,imax-1
!
! cff=0.25*(pm(i,j)+pm(i-1,j))
! & *(pn(i,j)+pn(i-1,j))
!
!
! wrkXadv(i,j,1)=wrkXadv(i,j,1)*cff
! wrkYadv(i,j,1)=wrkYadv(i,j,1)*cff
!
!
! cff=0.25*(pm(i,j)+pm(i,j-1))
! & *(pn(i,j)+pn(i,j-1))
!
!
! wrkXadv(i,j,2)=wrkXadv(i,j,2)*cff
! wrkYadv(i,j,2)=wrkYadv(i,j,2)*cff
!
! enddo
! enddo
!
!
!
! do j=jmin+2,jmax-2
! do i=imin+2,imax-2
!
! cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
! cff2 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))
!
! vrtXadv(i,j) = (wrkXadv(i,j,2) - wrkXadv(i-1,j,2)) * cff1
! & - (wrkXadv(i,j,1) - wrkXadv(i,j-1,1)) * cff2
!
!
! vrtYadv(i,j) = (wrkYadv(i,j,2) - wrkYadv(i-1,j,2)) * cff1
! & - (wrkYadv(i,j,1) - wrkYadv(i,j-1,1)) * cff2
!
!
! adv(i,j) = vrtXadv(i,j) + vrtYadv(i,j)
!
!
! enddo
! enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NEW COMPUTATION OF ROT (added 14/09/13)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

            cff=0.5*(pn(i,j)+pn(i-1,j))
     & / (0.5*(z_w(i,j,N) - z_w(i,j,0)
     & + z_w(i-1,j,N) - z_w(i-1,j,0)))


            wrkXadv(i,j,1)=wrkXadv(i,j,1)*cff
            wrkYadv(i,j,1)=wrkYadv(i,j,1)*cff

            cff=0.5*(pm(i,j)+pm(i,j-1))
     & / (0.5*(z_w(i,j,N) - z_w(i,j,0)
     & + z_w(i,j-1,N) - z_w(i,j-1,0)))

            wrkXadv(i,j,2)=wrkXadv(i,j,2)*cff
            wrkYadv(i,j,2)=wrkYadv(i,j,2)*cff

          enddo
        enddo



      do j=jmin+2,jmax-2
        do i=imin+2,imax-2

           cff1 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

            vrtXadv(i,j) = (wrkXadv(i,j,2) - wrkXadv(i-1,j,2)) * cff1
     & - (wrkXadv(i,j,1) - wrkXadv(i,j-1,1)) * cff1


            vrtYadv(i,j) = (wrkYadv(i,j,2) - wrkYadv(i-1,j,2)) * cff1
     & - (wrkYadv(i,j,1) - wrkYadv(i,j-1,1)) * cff1


            adv(i,j) = vrtXadv(i,j) + vrtYadv(i,j)


          enddo
         enddo





      return
      end
# 132 "R_tools_fort_gula.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute stuff
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# 1 "R_tools_fort_routines_gula/get_rotv.F" 1

!======================================================================
!
! Compute rotational in a flux consistent form
!
!
!
! - updated 16/08/19
!======================================================================

      subroutine get_rotv(Lm,Mm, u,v,pm,pn,rot)


      implicit none
      integer Lm,Mm, imin,imax,jmin,jmax, i,j
      real*8 rot(1:Lm+1,1:Mm+1),
     & u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dvdx, dudy, cff


Cf2py intent(in) Lm,Mm,u,v,pm,pn
Cf2py intent(out) rot

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1


!!!!!!!!!!!!!!!!!!!!!

      do j=jmin,jmax
        do i=imin+1,imax

         u(i,j) = u(i,j) * 2./ (pm(i,j) + pm(i-1,j))

        enddo !<- i
      enddo !<- j

!!!!!!!!!!!!!!!!!!!!!

      do j=jmin+1,jmax
        do i=imin,imax

         v(i,j) = v(i,j) * 2./ (pn(i,j) + pn(i,j-1))

        enddo !<- i
      enddo !<- j


        do j=jmin+1,jmax
          do i=imin+1,imax

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

          cff = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
     & * 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

            dvdx = (v(i,j) - v(i-1,j)) * cff
            dudy = (u(i,j) - u(i,j-1)) * cff

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            rot(i,j) = dvdx - dudy

        enddo !<- i
      enddo !<- j



      return
      end
# 139 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_hbbls_from_AKt.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!get_hbbls_from_AKt.F
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_hbbls_from_AKt(Lm,Mm,N1,N2,AKt,z_w,hbbls)


      implicit none
      integer Lm,Mm,N,N1,N2, imin,imax,jmin,jmax, i,j,k
      real*8 AKt(0:Lm+1,0:Mm+1,0:N1), z_w(0:Lm+1,0:Mm+1,0:N2),
     & hbbls(0:Lm+1,0:Mm+1)




Cf2py intent(in) Lm,Mm,N1,N2,AKt,z_w
Cf2py intent(out) hbbls




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

      N = min(N1,N2)

      do j=jmin,jmax
        do i=imin,imax

            k=1

            do while((AKt(i,j,k)>1e-4).and.(k.lt.N))
                k = k+1
            enddo

            hbbls(i,j) = z_w(i,j,max(1,k-1)) - z_w(i,j,0)


        enddo
      enddo






      return
      end
# 141 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_absvrt.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute PV
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_absvrt(Lm,Mm,N,u,v, z_r,z_w,pm,
     & pn,f,absvrt, absvrt0,dpth,var1,var2,var3,var4)


      implicit none
      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k
      real*8 stflx(0:Lm+1,0:Mm+1), ssflx(0:Lm+1,0:Mm+1),
     & u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & J1(1:Lm+1,1:Mm+1),f(0:Lm+1,0:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & hbls(0:Lm+1,0:Mm+1),cff3,
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & dpth(1:Lm+1,1:Mm+1),cff,cff2, Tt,Ts,sqrtTs, rho0, K0, dr00,
     & var1(1:Lm+1,1:Mm+1), var2(1:Lm+1,1:Mm+1),
     & var3(1:Lm+1,1:Mm+1), var4(1:Lm+1,1:Mm+1),
     & absvrt(1:Lm+1,1:Mm+1),absvrt0(1:Lm+1,1:Mm+1)



Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,f
Cf2py intent(out) absvrt, absvrt0, dpth,var1,var2,var3,var4


      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE VORTICITY
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!---------------------------------------------------------------------------------------

       do i=imin+1,imax
        do j=jmin+1,jmax

            dpth(i,j)=0.25*(z_r(i,j,N)+z_r(i-1,j,N)
     & + z_r(i-1,j-1,N)+z_r(i,j-1,N))

            CALL interp_1d(N,v(i,j,:)
     & ,0.5*(z_r(i,j,:)+z_r(i,j-1,:))
     & ,0.5*(z_w(i,j,:)+z_w(i,j-1,:))
     & ,dpth(i,j),var1(i,j),1,0)

            CALL interp_1d(N,v(i-1,j,:)
     & ,0.5*(z_r(i-1,j,:)+z_r(i-1,j-1,:))
     & ,0.5*(z_w(i-1,j,:)+z_w(i-1,j-1,:))
     & ,dpth(i,j),var2(i,j),1,0)


            CALL interp_1d(N,u(i,j,:)
     & ,0.5*(z_r(i-1,j,:)+z_r(i,j,:))
     & ,0.5*(z_w(i-1,j,:)+z_w(i,j,:))
     & ,dpth(i,j),var3(i,j),1,0)

            CALL interp_1d(N,u(i,j-1,:)
     & ,0.5*(z_r(i,j-1,:)+z_r(i-1,j-1,:))
     & ,0.5*(z_w(i,j-1,:)+z_w(i-1,j-1,:))
     & ,dpth(i,j),var4(i,j),1,0)

            cff = 0.25*(f(i,j) + f(i-1,j) + f(i,j-1) + f(i-1,j-1))
            cff2 = 0.25*(pm(i,j) + pm(i-1,j) + pm(i,j-1) + pm(i-1,j-1))
            cff3 = 0.25*(pn(i,j) + pn(i-1,j) + pn(i,j-1) + pn(i-1,j-1))

            absvrt(i,j)= cff
     & + (var1(i,j)-var2(i,j)) * cff2
     & - (var3(i,j)-var4(i,j)) * cff3


            absvrt0(i,j)= cff
     & + (v(i,j,N)-v(i-1,j,N)) * cff2
     & - (u(i,j,N)-u(i,j-1,N)) * cff3

          write(*,*) i,j,dpth(i,j),z_r(i,j,N),absvrt(i,j), absvrt0(i,j)

          write(*,*) i,j,v(i,j,:)
          write(*,*) i,j,0.5*(z_r(i,j,:)+z_r(i,j-1,:))
          write(*,*) i,j,0.5*(z_w(i,j,:)+z_w(i,j-1,:))
          write(*,*) i,j,var1(i,j)

         enddo

       enddo


!---------------------------------------------------------------------------------------





      return
      end
# 143 "R_tools_fort_gula.F" 2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# 1 "R_tools_fort_routines_gula/get_tracer_evolution.F" 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      subroutine get_tracer_evolution (Lm,Mm,N,u,v, z_r,z_w
     & ,pm,pn,dt
     & ,t,stflx, srflx, ghat, swr_frac, W, Akt
     & ,TXadv,TYadv,TVadv,THdiff,TVmix,TForc)


      integer Lm,Mm,N,NT, i,j,k
     & ,istr,iend,jstr,jend
     & ,imin,imax,jmin,jmax
     & ,itemp,isalt,dt,itrc

      parameter (NT=2)
      parameter (itemp=1,isalt=2)
      parameter (epsil=1.E-16)

      ! INPUTS
      real*8 t(0:Lm+1,0:Mm+1,N,NT)


      real*8 stflx(0:Lm+1,0:Mm+1,NT)
     & ,srflx(0:Lm+1,0:Mm+1)
     & ,ghat(0:Lm+1,0:Mm+1,N)

      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N)
     & ,z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N)
     & ,pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1)
     & ,W(0:Lm+1,0:Mm+1,0:N)
     & ,Akt(0:Lm+1,0:Mm+1,0:N)

      ! OUTPUTS
      real*8 TXadv(0:Lm+1,0:Mm+1,N,NT)
     & ,TYadv(0:Lm+1,0:Mm+1,N,NT)
     & ,TVadv(0:Lm+1,0:Mm+1,N,NT)
     & ,THdiff(0:Lm+1,0:Mm+1,N,NT)
     & ,TVmix(0:Lm+1,0:Mm+1,N,NT)
     & ,TForc(0:Lm+1,0:Mm+1,N,NT)


      ! LOCAL
      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1)
     & ,FX(0:Lm+1,0:Mm+1), FE(0:Lm+1,0:Mm+1)
     & ,WORK(0:Lm+1,0:Mm+1)
     & ,FC(0:Lm+1,0:N), DC(0:Lm+1,0:N)
     & ,CF(0:Lm+1,0:N)
     & ,Hz(0:Lm+1,0:Mm+1,N)
     & ,dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1)
     & ,FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N)
     & ,swr_frac(0:Lm+1,0:Mm+1,0:N)
     & ,tnew(0:Lm+1,0:Mm+1,N,NT)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn,dt,t,stflx, srflx, ghat, swr_frac,W, Akt
Cf2py intent(out) TXadv,TYadv,TVadv,THdiff,TVmix,TForc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        istr=1
        iend=Lm
        jstr=1
        jend=Mm


        imin=0
        imax=Lm+1
        jmin=0
        jmax=Mm+1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      do j=jmin,jmax
        do i=imin,imax
          do k=0,N,+1
              W(i,j,k)=W(i,j,k)/(pm(i,j)*pn(i,j))
            enddo
          enddo
        enddo


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



      !!call get_swr_frac (Lm,Mm,N, Hz, swr_frac )


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


! This part of the code is valid only if the ROMS simulations has been run with UPSTREAM scheme
! So it is not valid if was used
!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! First centered scheme
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      do itrc=1,NT
        do k=1,N

# 1 "R_tools_fort_routines_gula/compute_horiz_tracer_fluxes_centered.h" 1
! This code segment computes horizontal fluxes for tracer variables.
! Basically it interpolates tracer values from their native locations
! on C grid to horizontal velocity points. Curently three options are
! supported: 4-point symmetric fourth-order method (default); 3-point
! upstream-biased parabolic interpolation (UPSTREAM); and 4-point
! scheme where arithmetic averaging of elementary differences is
! replaced by harmonic averaging (AKIMA), resulting in mid-point
! values bounded by two nearest values at native location, regardless
! of grid-scale roughness of the interpolated field, while still
! retaining asymptotic fourth-order behavior for smooth fields.
! This code is extracted into a special module ibecause it is used
! twice, in predictor and corrector substeps for tracer variables.
!
# 24 "R_tools_fort_routines_gula/compute_horiz_tracer_fluxes_centered.h"
c--# define CONST_TRACERS







          do j=jstr,jend
            do i=imin,imax
              FX(i,j)=(t(i,j,k,itrc)-t(i-1,j,k,itrc))
            enddo
          enddo !--> discard imin,imax

          if (istr.eq.1) then
            do j=jstr,jend
              FX(istr-1,j)=FX(istr,j)
            enddo
          endif
          if (iend.eq.Lm) then
            do j=jstr,jend
              FX(iend+2,j)=FX(iend+1,j)
            enddo
          endif

          do j=jstr,jend
            do i=istr-1,iend+1
# 61 "R_tools_fort_routines_gula/compute_horiz_tracer_fluxes_centered.h"
              WORK(i,j)=0.5*(FX(i+1,j)+FX(i,j))

            enddo
          enddo !--> discard FX
          do j=jstr,jend
            do i=istr,iend+1






              FX(i,j)=0.5*( t(i,j,k,itrc)+t(i-1,j,k,itrc)
     & -0.333333333333*( WORK(i,j)-WORK(i-1,j))
     & )*FlxU(i,j,k)

            enddo !--> discard curv,WORK, keep FX
          enddo


          do j=jmin,jmax
            do i=istr,iend
              FE(i,j)=(t(i,j,k,itrc)-t(i,j-1,k,itrc))
            enddo
          enddo !--> discard jmin,jmax

          do j=jstr-1,jend+1
            do i=istr,iend
# 99 "R_tools_fort_routines_gula/compute_horiz_tracer_fluxes_centered.h"
              WORK(i,j)=0.5*(FE(i,j+1)+FE(i,j))

            enddo
          enddo !--> discard FE

          do j=jstr,jend+1
            do i=istr,iend






              FE(i,j)=0.5*( t(i,j,k,itrc)+t(i,j-1,k,itrc)
     & -0.333333333333*(WORK(i,j)-WORK(i,j-1))
     & )*FlxV(i,j,k)

            enddo
          enddo !--> discard curv,WORK, keep FE
# 140 "R_tools_fort_routines_gula/get_tracer_evolution.F" 2

          do j=jstr,jend
            do i=istr,iend

              THdiff(i,j,k,itrc) = FX(i+1,j)-FX(i,j)
     & + FE(i,j+1)-FE(i,j)

            enddo
          enddo !--> discard FX,FE
        enddo
      enddo




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! then using UPSTREAM scheme
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do itrc=1,NT
        do k=1,N

# 1 "R_tools_fort_routines_gula/compute_horiz_tracer_fluxes_upstream.h" 1
! This code segment computes horizontal fluxes for tracer variables.
! Basically it interpolates tracer values from their native locations
! on C grid to horizontal velocity points. Curently three options are
! supported: 4-point symmetric fourth-order method (default); 3-point
! upstream-biased parabolic interpolation (UPSTREAM); and 4-point
! scheme where arithmetic averaging of elementary differences is
! replaced by harmonic averaging (AKIMA), resulting in mid-point
! values bounded by two nearest values at native location, regardless
! of grid-scale roughness of the interpolated field, while still
! retaining asymptotic fourth-order behavior for smooth fields.
! This code is extracted into a special module ibecause it is used
! twice, in predictor and corrector substeps for tracer variables.
!






c--# define CONST_TRACERS







          do j=jstr,jend
            do i=imin,imax
              FX(i,j)=(t(i,j,k,itrc)-t(i-1,j,k,itrc))
            enddo
          enddo !--> discard imin,imax

          if (istr.eq.1) then
            do j=jstr,jend
              FX(istr-1,j)=FX(istr,j)
            enddo
          endif
          if (iend.eq.Lm) then
            do j=jstr,jend
              FX(iend+2,j)=FX(iend+1,j)
            enddo
          endif

          do j=jstr,jend
            do i=istr-1,iend+1

              WORK(i,j)=FX(i+1,j)-FX(i,j)
# 59 "R_tools_fort_routines_gula/compute_horiz_tracer_fluxes_upstream.h"
            enddo
          enddo !--> discard FX
          do j=jstr,jend
            do i=istr,iend+1

              FX(i,j)=0.5*(t(i,j,k,itrc)+t(i-1,j,k,itrc))
     & *FlxU(i,j,k)
     & -0.166666666666*( WORK(i-1,j)*max(FlxU(i,j,k),0.)
     & +WORK(i ,j)*min(FlxU(i,j,k),0.))





            enddo !--> discard WORK,WORK, keep FX
          enddo


          do j=jmin,jmax
            do i=istr,iend
              FE(i,j)=(t(i,j,k,itrc)-t(i,j-1,k,itrc))
            enddo
          enddo !--> discard jmin,jmax

          do j=jstr-1,jend+1
            do i=istr,iend

              WORK(i,j)=FE(i,j+1)-FE(i,j)
# 97 "R_tools_fort_routines_gula/compute_horiz_tracer_fluxes_upstream.h"
            enddo
          enddo !--> discard FE

          do j=jstr,jend+1
            do i=istr,iend

              FE(i,j)=0.5*(t(i,j,k,itrc)+t(i,j-1,k,itrc))
     & *FlxV(i,j,k)
     & -0.166666666666*( WORK(i,j-1)*max(FlxV(i,j,k),0.)
     & +WORK(i,j )*min(FlxV(i,j,k),0.))





            enddo
          enddo !--> discard WORK,WORK, keep FE
# 164 "R_tools_fort_routines_gula/get_tracer_evolution.F" 2

          do j=jstr,jend
            do i=istr,iend
              tnew(i,j,k,itrc)=Hz(i,j,k)*t(i,j,k,itrc)
     & -dt*pm(i,j)*pn(i,j)*( FX(i+1,j)-FX(i,j)
     & +FE(i,j+1)-FE(i,j)
     & )
              TXadv(i,j,k,itrc) = -(FX(i+1,j)-FX(i,j))
              TYadv(i,j,k,itrc) = -(FE(i,j+1)-FE(i,j))


              THdiff(i,j,k,itrc) = THdiff(i,j,k,itrc)
     & -( FX(i+1,j)-FX(i,j)
     & + FE(i,j+1)-FE(i,j)
     & )
            enddo
          enddo !--> discard FX,FE
        enddo
      enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jstr,jend
        do itrc=1,NT


# 1 "R_tools_fort_routines_gula/compute_vert_tracer_fluxes.h" 1
! This is "compute_vert_tracer_fluxes.h" -- module which computes
! vertical fluxes for tracer equations. In the case of SPLINES two
! versions of top and bottom boundary conditions are supported:
! Neumann (setting first derivative to zero at the top and bottom
! boundaries) and LINEAR CONTINUATION (assumption that the tracer
! distributions are linear within the top-most and botom-most grid
! boxes).
!




c--#define SPLINES
# 56 "R_tools_fort_routines_gula/compute_vert_tracer_fluxes.h"
          do k=1,N-1
            do i=istr,iend
              FC(i,k)=t(i,j,k+1,itrc)-t(i,j,k,itrc)
            enddo
          enddo
          do i=istr,iend
            FC(i,0)=FC(i,1)
            FC(i,N)=FC(i,N-1)
          enddo
          do k=1,N
            do i=istr,iend
              cff=2.*FC(i,k)*FC(i,k-1)
              if (cff.gt.epsil) then
                CF(i,k)=cff/(FC(i,k)+FC(i,k-1))
              else
                CF(i,k)=0.
              endif
            enddo
          enddo !--> discard FC
          do k=1,N-1
            do i=istr,iend
              FC(i,k)=0.5*( t(i,j,k,itrc)+t(i,j,k+1,itrc)
     & -0.333333333333*(CF(i,k+1)-CF(i,k)) )*W(i,j,k)
            enddo
          enddo !--> discard CF
          do i=istr,iend
            FC(i,0)=0.
            FC(i,N)=0.
          enddo
# 193 "R_tools_fort_routines_gula/get_tracer_evolution.F" 2

          do k=1,N ! Apply vertical advective fluxes.
            do i=istr,iend

               tnew(i,j,k,itrc)=tnew(i,j,k,itrc)-dt*pm(i,j)
     & *pn(i,j)*(FC(i,k)-FC(i,k-1))

              TVadv(i,j,k,itrc) = -(FC(i,k)-FC(i,k-1))

            enddo
          enddo !--> discard FC

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!
! Add surface and bottom fluxes
!

          do i=istr,iend
            cff = 1.
            if (itrc.eq.itemp .and.
     & (tnew(i,j,N,itrc) .le. -1.8) ) cff = 0.

             tnew(i,j,N,itrc)=tnew(i,j,N,itrc)+dt*stflx(i,j,itrc)*cff

            TForc(i,j,N,itrc)= stflx(i,j,itrc)*cff/(pm(i,j)*pn(i,j))
            do k=N-1,1,-1
                TForc(i,j,k,itrc)= 0.
            enddo

          enddo
# 242 "R_tools_fort_routines_gula/get_tracer_evolution.F"
        !write(*,*) 'sans deconner...',j,jend




!
! Add the solar radiation flux in temperature equation. Also compute
! the nonlocal transport flux for unstable (convective) forcing
! conditions into matrix DC when using the Large et al. 1994 KPP
! scheme.
!


          if (itrc.eq.itemp) then
            do k=N-1,1,-1
              do i=istr,iend
                cff=srflx(i,j)*swr_frac(i,j,k)

     & -ghat(i,j,k)*(stflx(i,j,itemp)-srflx(i,j))

                 tnew(i,j,k+1,itemp)=tnew(i,j,k+1,itemp) -dt*cff
                 tnew(i,j,k ,itemp)=tnew(i,j,k ,itemp) +dt*cff
                TForc(i,j,k+1,itrc)= TForc(i,j,k+1,itrc)
     & -cff/(pm(i,j)*pn(i,j))
                TForc(i,j,k,itrc)= TForc(i,j,k,itrc)
     & +cff/(pm(i,j)*pn(i,j))
              enddo
            enddo




c??
c?? WARNING: the above implies that light (if any) reaching all the
c?? way to the bottom is entirely absorbed within the botom-most grid
c?? box, rather that reflected/scatered back to the water column. It
c?? is not clear, however, how to deal with this situation correctly
c??
c?? do i=istr,iend
c?? cff=srflx(i,j)*swr_frac(i,j,0)
c??# ifdef
c?? & -ghat(i,j,0)*(stflx(i,j,itemp)-srflx(i,j))
c??# endif
c?? t(i,j,1,itemp)=t(i,j,1,itemp) -dt*cff
c?? enddo



          elseif (itrc.eq.isalt) then
            do k=N-1,1,-1
              do i=istr,iend
                cff=-dt*ghat(i,j,k)*stflx(i,j,isalt)
                 tnew(i,j,k+1,isalt)=tnew(i,j,k+1,isalt) -cff
                 tnew(i,j,k ,isalt)=tnew(i,j,k ,isalt) +cff
                TForc(i,j,k+1,itrc)= TForc(i,j,k+1,itrc)
     & -cff/(dt*pm(i,j)*pn(i,j))
                TForc(i,j,k,itrc)= TForc(i,j,k,itrc)
     & +cff/(dt*pm(i,j)*pn(i,j))

              enddo
            enddo



          endif



        !write(*,*) 'Forc OK',j,jend

! Perform implicit time step for vertical diffusion,
!
! dq(k) 1 [ q(k+1)-q(k) q(k)-q(k-1) ]
! ------ = ----- * [ Akt(k)* ----------- - Akt(k-1)* ----------- ]
! dt Hz(k) [ dz(k) dz(k-1) ]
!
! where q(k) represents tracer field t(:,:,k,:,itrc). Doing so
! implies solution of a tri-diagonal system
!
! -FC(k-1)*q_new(k-1) +[Hz(k)+FC(k-1)+FC(k)]*q_new(k)
! -FC(k)*q_new(k+1) = Hz(k)*q_old(k)
!
! dt*Akt(k)
! where FC(k) = ----------- is normalized diffusivity coefficient
! dz(k)
!
! defined at W-points; q_new(k) is the new-time-step (unknown) tracer
! field; q_old(k) is old-time-step tracer (known). As long as
! vertical diffusivity Akt(k) is nonnegative, the tri-diagonal matrix
! is diagonally dominant which guarantees stability of a Gaussian
! elimination procedure, (e.g., Richtmeyer annd Morton, 1967).
! Top and bottom boundary conditions are assumed to be no-flux,
! effectively Akt(N)=Akt(0)=0, hence FC(N)=FC(1)=0. This leads to
! equations for top and bottom grid boxes;
!
! -FC(N-1)*q_new(N-1) +[Hz(N)+FC(N-1)]*q_new(N) = Hz(N)*q_old(N)
!
! [Hz(1)+FC(1)]*q_new(1) -FC(1)*q_new(2) = Hz(1)*q_old(1)
!
! The FC(N)=FC(0)=0 boundary conditions does not mean that physical
! boundary conditions are no flux: the forcing fluxes have been
! applied explicitly above. Instead, the no-flux condition should
! be interpreted as that the implicit step merely redistributes the
! tracer concentration throughout the water column. At this moment
! the content of array t(:,:,:,itrc) has meaning of Hz*tracer.
! After the implicit step it becomes just tracer.
!
          do k=1,N
            do i=istr,iend
               TVmix(i, j, k, itrc)=tnew(i,j,k,itrc)
            enddo
         enddo


          do i=istr,iend
            FC(i,1)=dt*Akt(i,j,1)/(z_r(i,j,2)-z_r(i,j,1))
            cff=1./(Hz(i,j,1)+FC(i,1))
            CF(i,1)=cff*FC(i,1)
            DC(i,1)=cff*tnew(i,j,1,itrc)
          enddo
          do k=2,N-1,+1
            do i=istr,iend
              FC(i,k)=dt*Akt(i,j,k)/(z_r(i,j,k+1)-z_r(i,j,k))
              cff=1./( Hz(i,j,k) +FC(i,k)+FC(i,k-1)*(1.-CF(i,k-1)) )
              CF(i,k)=cff*FC(i,k)
              DC(i,k)=cff*(tnew(i,j,k,itrc)+FC(i,k-1)*DC(i,k-1))
            enddo
          enddo

          do i=istr,iend
             tnew(i,j,N,itrc)=( tnew(i,j,N,itrc) +FC(i,N-1)
     & *DC(i,N-1) )/(Hz(i,j,N)+FC(i,N-1)*(1.-CF(i,N-1)))
          enddo
          do k=N-1,1,-1
            do i=istr,iend
              tnew(i,j,k,itrc)=DC(i,k)+CF(i,k)*tnew(i,j,k+1,itrc)
            enddo
          enddo !--> discard FC,CF,DC

          do k=1,N
            do i=istr,iend
              TVmix(i,j,k,itrc) =
     & -(TVmix(i,j,k,itrc)-tnew(i,j,k,itrc)*Hz(i,j,k))
     & /(dt*pm(i,j)*pn(i,j))
            enddo
          enddo




       enddo ! <-- itrc

       !write(*,*) 'the end..',j,jend


      enddo ! <-- j


!
! Set lateral boundary conditions; nudge toward tracer climatology;
! apply land-sea mask and exchange periodic boundary conditions.
!
      do itrc=1,NT

!---------------------------------------------------------------
! Compute the tendency term of tracer diagnostics
! Divide all diagnostic terms by the cell volume
! (Hz(i,j,k,itrc)/(pm(i,j).*pn(i,j)). There after the unit
! of diagnostic terms will be: (unit of tracers)* s-1.
!
! Note: the Horizontal mixing term is computed in t3dmix
! where Trate is updated accordingly
!---------------------------------------------------------------

       do k=1,N
         do j=jstr,jend
           do i=istr,iend


              cff=pm(i,j)*pn(i,j)/Hz(i,j,k)
              TXadv(i,j,k,itrc)=TXadv(i,j,k,itrc)*cff
              TYadv(i,j,k,itrc)=TYadv(i,j,k,itrc)*cff
              TVadv(i,j,k,itrc)=TVadv(i,j,k,itrc)*cff
              TVmix(i,j,k,itrc)=TVmix(i,j,k,itrc)*cff
              THdiff(i,j,k,itrc)=THdiff(i,j,k,itrc)*cff
              TForc(i,j,k,itrc)=TForc(i,j,k,itrc)*cff


           enddo
         enddo

       !write(*,*) 'the end..',k,N

       enddo



      enddo ! <-- itrc

!---------------------------------------------------------------

       !write(*,*) 'cooooool'


      return
      end
# 149 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_uv_evolution.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute terms of the momentum equations in ROMS
!!
!! Note that MHdiss is the difference between upstream and 4th order centered advection (not to be included in the total)
!!
!! Note that MHmix is the sponge dissipation
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_uv_evolution (Lm,Mm,N,u,v,T,S, z_r,z_w
     & ,pm,pn,f,dt,rmask
     & ,rdrg, rho0, W, Akv, sustr, svstr
     & ,visc2, v_sponge, visctype, coord, coordmax
     & ,MXadv, MYadv, MVadv, MHdiss, MHmix, MVmix, MCor, MPrsgrd)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,dt,
     & itrc, visctype

        !INPUT
      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N)
     & ,z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N)
     & ,pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1)
     & ,rmask(0:Lm+1,0:Mm+1)
     & ,umask(1:Lm+1,0:Mm+1), vmask(0:Lm+1,1:Mm+1)
     & ,f(0:Lm+1,0:Mm+1)
     & ,W(0:Lm+1,0:Mm+1,0:N)
     & ,Akv(0:Lm+1,0:Mm+1,0:N)
     & ,sustr(1:Lm+1,0:Mm+1), svstr(0:Lm+1,1:Mm+1)

      integer coord(4), coordmax(4)

      ! OUTPUTS
      real*8 MXadv(0:Lm+1,0:Mm+1,N,2)
     & ,MYadv(0:Lm+1,0:Mm+1,N,2)
     & ,MVadv(0:Lm+1,0:Mm+1,N,2)
     & ,MHdiss(0:Lm+1,0:Mm+1,N,2)
     & ,MHmix(0:Lm+1,0:Mm+1,N,2)
     & ,MVmix(0:Lm+1,0:Mm+1,N,2)
     & ,MCor(0:Lm+1,0:Mm+1,N,2)
     & ,MPrsgrd(0:Lm+1,0:Mm+1,N,2)


      real*8 rdrg, Zob

      real*8 FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & dm_u(0:Lm+1,0:Mm+1), dn_v(0:Lm+1,0:Mm+1),
     & fomn(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2

      real*8 ru(1:Lm+1,0:Mm+1,N), rv(0:Lm+1,1:Mm+1,N)

      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1),
     & UFx_mix(0:Lm+1,0:Mm+1), UFe_mix(0:Lm+1,0:Mm+1),
     & VFx_mix(0:Lm+1,0:Mm+1), VFe_mix(0:Lm+1,0:Mm+1),
     & dmde(1:Lm,1:Mm), dndx(1:Lm,1:Mm)

      real*8 FC(0:Lm+1,0:N), DC(0:Lm+1,0:N)
     & ,CF(0:Lm+1,0:N)


      real visc2, v_sponge


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & rho1(0:Lm+1,0:Mm+1,N), qp1(0:Lm+1,0:Mm+1,N),
     & Tt,Ts,sqrtTs, rho0, K0, dr00,
     & cfr, HalfGRho, GRho

      real*8 P(0:Lm+1,0:Mm+1,N),
     & rho(0:Lm+1,0:Mm+1,N), dpth,
     & dR(0:Lm+1,N), dZ(0:Lm+1,N),
     & dZx(0:Lm+1,0:Mm+1),
     & rx(0:Lm+2,0:Mm+2), dRx(0:Lm+1,0:Mm+1),
     & FCP(0:Lm+2,0:Mm+2)

      real*8, parameter :: OneFifth=0.2, OneTwelfth=1./12., epsil=0.

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
     & qp2=0.0000172


      real rho1_0, K0_Duk



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



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
# 112 "R_tools_fort_routines_gula/get_uv_evolution.F" 2

      parameter (gamma=0.25)





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
# 122 "R_tools_fort_routines_gula/get_uv_evolution.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,u,v,T,S,z_r,z_w,pm,pn,f,dt,rmask,rdrg,rho0,W,Akv,sustr, svstr,visc2, v_sponge, visctype, coord, coordmax
Cf2py intent(out) MXadv, MYadv, MVadv, MHdiss, MHmix, Mcor, MVmix, MPrsgrd


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jstrR,jendR
        do i=istrR,iendR
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo
           fomn(i,j)=f(i,j)/(pm(i,j)*pn(i,j))
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jstrR,jendR
        do i=istrR,iendR
          do k=0,N,+1
              W(i,j,k)=W(i,j,k)/(pm(i,j)*pn(i,j))
            enddo
          enddo
        enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jstrR,jendR
        do i=istr,iendR
            dn_u(i,j) = 2./(pn(i,j)+pn(i-1,j))
            dm_u(i,j) = 2./(pm(i,j)+pm(i-1,j))
            umask(i,j) = rmask(i,j)*rmask(i-1,j)
            do k=1,N,+1
              u(i,j,k) = u(i,j,k) * umask(i,j)
              FlxU(i,j,k) = 0.5*(Hz(i,j,k)+Hz(i-1,j,k))*dn_u(i,j)
     & * u(i,j,k)
            enddo
          enddo
      enddo



      do j=jstr,jendR
        do i=istrR,iendR
            dm_v(i,j) = 2./(pm(i,j)+pm(i,j-1))
            dn_v(i,j) = 2./(pn(i,j)+pn(i,j-1))
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
            do k=1,N,+1
              v(i,j,k) = v(i,j,k) * vmask(i,j)
              FlxV(i,j,k) = 0.5*(Hz(i,j,k)+Hz(i,j-1,k))*dm_v(i,j)
     & * v(i,j,k)
            enddo
          enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      write(*,*) 'calling prsgrd.h'

# 1 "R_tools_fort_routines_gula/compute_prsgrd.h" 1






!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE DENSITY
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




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
      do j=jstrR,jendR
        do k=1,N ! NONLINEAR
          do i=istrR,iendR ! EQUATION
            Tt=T(i,j,k) ! OF STATE

            Ts=max(S(i,j,k), 0.)
            sqrtTs=sqrt(Ts)




            rho1(i,j,k)=( dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*(
     & r04+Tt*r05 ))))
     & +Ts*( r10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*(rS0+Tt*(
     & rS1+Tt*rS2 ))+Ts*r20 ))

     & *rmask(i,j)





            K0= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))



            qp1(i,j,k)= 0.1D0*(rho0+rho1(i,j,k))*(K0_Duk-K0)
     & /((K00+K0)*(K00+K0_Duk))





     & *rmask(i,j)
# 105 "R_tools_fort_routines_gula/compute_prsgrd.h"
          enddo

        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! A non-conservative Density-Jacobian scheme using cubic polynomial
! fits for rho and z_r as functions of nondimensianal coordinates xi,
! eta, and s (basically their respective fortran indices). The cubic
! polynomials are constructed by specifying first derivatives of
! interpolated fields on co-located (non-staggered) grid. These
! derivatives are computed using harmonic (rather that algebraic)
! averaging of elementary differences, which guarantees monotonicity
! of the resultant interpolant.
!
! In the code below, if CPP-switch is defined, the Equation
! of State (EOS) is assumed to have form
!
! rho(T,S,z) = rho1(T,S) + qp1(T,S)*dpth*[1.-qp2*dpth]
!
! where rho1 is potential density at 1 atm and qp1 is compressibility
! coefficient, which does not depend on z, and dpth=zeta-z, and qp2
! is just a constant. In this case
!
! d rho d rho1 d qp1 d z
! ------- = ------ + ----- *dpth*[..] - qp1*[1.-2.*qp2*dpth]*------
! d s,x d s,x d s,x d s,x
!
! |<--- adiabatic part --->| |<--- compressible part --->|
!
! where the first two terms constitute "adiabatic derivative" of
! density, which is subject to harmonic averaging, while the last
! term is added in later. This approach quarantees that density
! profile reconstructed by cubic polynomial maintains its positive
! statification in physical sense as long as discrete values of
! density are positively stratified.
!
! This scheme retains exact antisymmetry J(rho,z_r)=-J(z_r,rho)
! [with the exception of harmonic averaging algorithm in the case
! when CPP-switch is defined, see above]. If parameter
! OneFifth (see above) is set to zero, the scheme becomes identical
! to standard Jacobian.
!
! NOTE: This routine is an alternative form of prsgrd32 and it
! produces results identical to that if its prototype.
!

!


      if (istr.eq.1) then ! Restrict extended ranges one
        imin=istrU ! point inward near the physical
      else ! boundary. Note that this version
        imin=istrU-1 ! of code is suitable for MPI
      endif ! configuration too.
      if (iend.eq.Lm) then
        imax=iend
      else
        imax=iend+1
      endif





        if (jstr.eq.1) then
          jmin=jstrV
        else
          jmin=jstrV-1
        endif
        if (jend.eq.Mm) then
          jmax=jend
        else
          jmax=jend+1
        endif






!
! Preliminary step (same for XI- and ETA-components:
!------------ ---- ----- --- --- --- ---------------
!
      GRho=g/rho0
      HalfGRho=0.5*GRho

      do j=jstrV-1,jend
        do k=1,N-1
          do i=istrU-1,iend
            dZ(i,k)=z_r(i,j,k+1)-z_r(i,j,k)

            dpth=z_w(i,j,N)-0.5*(z_r(i,j,k+1)+z_r(i,j,k))

            dR(i,k)=rho1(i,j,k+1)-rho1(i,j,k) ! Elementary
     & +(qp1(i,j,k+1)-qp1(i,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ! difference







          enddo
        enddo
        do i=istrU-1,iend
          dR(i,N)=dR(i,N-1)
          dR(i,0)=dR(i,1)
          dZ(i,N)=dZ(i,N-1)
          dZ(i,0)=dZ(i,1)
        enddo
        do k=N,1,-1 !--> irreversible
          do i=istrU-1,iend
            cff=2.*dZ(i,k)*dZ(i,k-1)
            dZ(i,k)=cff/(dZ(i,k)+dZ(i,k-1))

            cfr=2.*dR(i,k)*dR(i,k-1)
            if (cfr.gt.epsil) then
              dR(i,k)=cfr/(dR(i,k)+dR(i,k-1))
            else
              dR(i,k)=0.
            endif

            dpth=z_w(i,j,N)-z_r(i,j,k)
            dR(i,k)=dR(i,k) -qp1(i,j,k)*dZ(i,k)*(1.-2.*qp2*dpth)
            rho(i,j,k)=rho1(i,j,k) +qp1(i,j,k)*dpth*(1.-qp2*dpth)




          enddo
        enddo
        do i=istrU-1,iend
          P(i,j,N)=g*z_w(i,j,N) + GRho*( rho(i,j,N)
     & +0.5*(rho(i,j,N)-rho(i,j,N-1))*(z_w(i,j,N)-z_r(i,j,N))
     & /(z_r(i,j,N)-z_r(i,j,N-1)) )*(z_w(i,j,N)-z_r(i,j,N))


        enddo
        do k=N-1,1,-1
          do i=istrU-1,iend
            P(i,j,k)=P(i,j,k+1)+HalfGRho*( (rho(i,j,k+1)+rho(i,j,k))
     & *(z_r(i,j,k+1)-z_r(i,j,k))

     & -OneFifth*( (dR(i,k+1)-dR(i,k))*( z_r(i,j,k+1)-z_r(i,j,k)
     & -OneTwelfth*(dZ(i,k+1)+dZ(i,k)) )

     & -(dZ(i,k+1)-dZ(i,k))*( rho(i,j,k+1)-rho(i,j,k)
     & -OneTwelfth*(dR(i,k+1)+dR(i,k)) )
     & ))
          enddo
        enddo
      enddo !<-- j

!
! Compute XI-component of pressure gradient term:
!-------- ------------ -- -------- -------- -----
!
      do k=N,1,-1
        do j=jstr,jend
          do i=imin,imax




            FCP(i,j)=(z_r(i,j,k)-z_r(i-1,j,k))

     & *umask(i,j)


            dpth=0.5*( z_w(i,j,N)+z_w(i-1,j,N)
     & -z_r(i,j,k)-z_r(i-1,j,k))

            rx(i,j)=( rho1(i,j,k)-rho1(i-1,j,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i-1,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference




     & *umask(i,j)




          enddo
        enddo


        if (istr.eq.1) then ! Extrapolate elementary
          do j=jstr,jend ! differences near physical
            FCP(imin-1,j)=FCP(imin,j) ! boundaries to compencate.
            rx(imin-1,j)=rx(imin,j) ! for reduced loop ranges.
          enddo
        endif
        if (iend.eq.Lm) then
          do j=jstr,jend
            FCP(imax+1,j)=FCP(imax,j)
            rx(imax+1,j)=rx(imax,j)
          enddo
        endif


        do j=jstr,jend
          do i=istrU-1,iend
            cff=2.*FCP(i,j)*FCP(i+1,j)
            if (cff.gt.epsil) then
              dZx(i,j)=cff/(FCP(i,j)+FCP(i+1,j))
            else
              dZx(i,j)=0.
            endif

            cfr=2.*rx(i,j)*rx(i+1,j)
            if (cfr.gt.epsil) then
              dRx(i,j)=cfr/(rx(i,j)+rx(i+1,j))
            else
              dRx(i,j)=0.
            endif



            dRx(i,j)=dRx(i,j) -qp1(i,j,k)*dZx(i,j)
     & *(1.-2.*qp2*(z_w(i,j,N)-z_r(i,j,k)))

          enddo !--> discard FCP, rx

          do i=istrU,iend
            ru(i,j,k)=0.5*(Hz(i,j,k)+Hz(i-1,j,k))*dn_u(i,j)*(
     & P(i-1,j,k)-P(i,j,k)-HalfGRho*(

     & (rho(i,j,k)+rho(i-1,j,k))*(z_r(i,j,k)-z_r(i-1,j,k))

     & -OneFifth*( (dRx(i,j)-dRx(i-1,j))*( z_r(i,j,k)-z_r(i-1,j,k)
     & -OneTwelfth*(dZx(i,j)+dZx(i-1,j)) )

     & -(dZx(i,j)-dZx(i-1,j))*( rho(i,j,k)-rho(i-1,j,k)
     & -OneTwelfth*(dRx(i,j)+dRx(i-1,j)) )
     & )))


              MPrsgrd(i,j,k,1) = ru(i,j,k)


          enddo
        enddo





! ETA-component of pressure gradient term:
!-------------- -- -------- -------- -----
!
        do j=jmin,jmax
          do i=istr,iend
            FCP(i,j)=(z_r(i,j,k)-z_r(i,j-1,k))

     & *vmask(i,j)


            dpth=0.5*( z_w(i,j,N)+z_w(i,j-1,N)
     & -z_r(i,j,k)-z_r(i,j-1,k))

            rx(i,j)=( rho1(i,j,k)-rho1(i,j-1,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i,j-1,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference




     & *vmask(i,j)


          enddo
        enddo


        if (jstr.eq.1) then
          do i=istr,iend
            FCP(i,jmin-1)=FCP(i,jmin)
            rx(i,jmin-1)=rx(i,jmin)

          enddo
        endif
        if (jend.eq.Mm) then
          do i=istr,iend
            FCP(i,jmax+1)=FCP(i,jmax)
            rx(i,jmax+1)=rx(i,jmax)

          enddo
        endif


        do j=jstrV-1,jend
          do i=istr,iend
            cff=2.*FCP(i,j)*FCP(i,j+1)
            if (cff.gt.epsil) then
c** if ((FCP(i,j).gt.0. .and. FCP(i,j+1).gt.0.) .or.
c** & (FCP(i,j).lt.0. .and. FCP(i,j+1).lt.0.)) then
              dZx(i,j)=cff/(FCP(i,j)+FCP(i,j+1))
            else
              dZx(i,j)=0.
            endif

            cfr=2.*rx(i,j)*rx(i,j+1)
            if (cfr.gt.epsil) then
c** if ((rx(i,j).gt.0. .and. rx(i,j+1).gt.0.) .or.
c** & (rx(i,j).lt.0. .and. rx(i,j+1).lt.0.)) then
              dRx(i,j)=cfr/(rx(i,j)+rx(i,j+1))
            else
              dRx(i,j)=0.
            endif

            dRx(i,j)=dRx(i,j) -qp1(i,j,k)*dZx(i,j)
     & *(1.-2.*qp2*(z_w(i,j,N)-z_r(i,j,k)))

          enddo !--> discard FCP, rx

          if (j.ge.jstrV) then
            do i=istr,iend
              rv(i,j,k)=0.5*(Hz(i,j,k)+Hz(i,j-1,k))*dm_v(i,j)*(
     & P(i,j-1,k)-P(i,j,k) -HalfGRho*(

     & (rho(i,j,k)+rho(i,j-1,k))*(z_r(i,j,k)-z_r(i,j-1,k))

     & -OneFifth*( (dRx(i,j)-dRx(i,j-1))*( z_r(i,j,k)-z_r(i,j-1,k)
     & -OneTwelfth*(dZx(i,j)+dZx(i,j-1)) )

     & -(dZx(i,j)-dZx(i,j-1))*( rho(i,j,k)-rho(i,j-1,k)
     & -OneTwelfth*(dRx(i,j)+dRx(i,j-1)) )
     & )))

              MPrsgrd(i,j,k,2) = rv(i,j,k)

            enddo
          endif
        enddo
      enddo
# 189 "R_tools_fort_routines_gula/get_uv_evolution.F" 2
      write(*,*) 'prsgrd.h ok'

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!
! Compute d(1/n)/d(xi) and d(1/m)/d(eta) tems, both at RHO-points.
!
      do j=jstr,jend
        do i=istr,iend
          dndx(i,j)=0.5/pn(i+1,j)-0.5/pn(i-1,j)
          dmde(i,j)=0.5/pm(i,j+1)-0.5/pm(i,j-1)
        enddo
      enddo




       write(*,*) 'dt',dt

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      do k=1,N
!


!
! Add in Coriolis and curvilinear transformation terms, if any.
!

        do j=jstrV-1,jend
          do i=istrU-1,iend
            cff=0.5*Hz(i,j,k)*(

     & fomn(i,j)


     & +0.5*( (v(i,j,k)+v(i,j+1,k))*dndx(i,j)
     & -(u(i,j,k)+u(i+1,j,k))*dmde(i,j))

     & )
            UFx(i,j)=cff*(v(i,j,k)+v(i,j+1,k))
            VFe(i,j)=cff*(u(i,j,k)+u(i+1,j,k))
          enddo
        enddo
        do j=jstr,jend
          do i=istrU,iend
            ru(i,j,k)=ru(i,j,k)+0.5*(UFx(i,j)+UFx(i-1,j))
             MCor(i,j,k,1) = 0.5*(UFx(i,j)+UFx(i-1,j))
          enddo
        enddo
        do j=jstrV,jend
          do i=istr,iend
            rv(i,j,k)=rv(i,j,k)-0.5*(VFe(i,j)+VFe(i,j-1))
            MCor(i,j,k,2) = -0.5*(VFe(i,j)+VFe(i,j-1))
          enddo
        enddo
# 262 "R_tools_fort_routines_gula/get_uv_evolution.F"
!
! Add in horizontal advection of momentum: Compute diagonal [UFx,VFe]
! and off-diagonal [UFe,VFx] components of tensor of momentum flux
! due to horizontal advection; after that add divergence of these
! terms to r.h.s.
!



        if (istr.eq.1) then ! Sort out bounding indices of
          imin=istrU ! extended ranges: note that in
        else ! the vicinity of physical
          imin=istrU-1 ! boundaries values at the
        endif ! extremal points of stencil
        if (iend.eq.Lm) then ! are not available, so an
          imax=iend ! extrapolation rule needs to
        else ! be applied. Also note that
          imax=iend+1 ! for this purpose periodic
        endif ! ghost points and MPI margins




        do j=jstr,jend
          do i=imin,imax
            wrk1(i,j)=u(i-1,j,k)-2.*u(i,j,k)
     & +u(i+1,j,k)
            wrk2(i,j)=FlxU(i-1,j,k)-2.*FlxU(i,j,k)
     & +FlxU(i+1,j,k)
          enddo
        enddo

        if (istr.eq.1) then
          do j=jstr,jend
            wrk1(istrU-1,j) =wrk1(istrU,j)
            wrk2(istrU-1,j)=wrk2(istrU,j)
          enddo
        endif
        if (iend.eq.Lm) then
          do j=jstr,jend
            wrk1(iend+1,j) =wrk1(iend,j)
            wrk2(iend+1,j)=wrk2(iend,j)
          enddo
        endif


        do j=jstr,jend
          do i=istrU-1,iend
            cff=FlxU(i,j,k)+FlxU(i+1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i+1,j))

            UFx(i,j)=0.25*( cff*(u(i,j,k)+u(i+1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i ,j)
     & +min(cff,0.)*wrk1(i+1,j)
     & ))

            UFx_mix(i,j)=UFx(i,j)-0.25*( u(i,j,k)+u(i+1,j,k)
     & -0.125*(wrk1(i,j)+wrk1(i+1,j))
     & )*( FlxU(i,j,k)+FlxU(i+1,j,k)
     & -0.125*(wrk2(i,j)+wrk2(i+1,j)))

          enddo
        enddo






        if (jstr.eq.1) then
          jmin=jstrV
        else
          jmin=jstrV-1
        endif
        if (jend.eq.Mm) then
          jmax=jend
        else
          jmax=jend+1
        endif




        do j=jmin,jmax
          do i=istr,iend
            wrk1(i,j)=v(i,j-1,k)-2.*v(i,j,k)+v(i,j+1,k)
            wrk2(i,j)=FlxV(i,j-1,k)-2.*FlxV(i,j,k)+FlxV(i,j+1,k)
          enddo
        enddo

        if (jstr.eq.1) then
          do i=istr,iend
            wrk1(i,jstrV-1)=wrk1(i,jstrV)
            wrk2(i,jstrV-1)=wrk2(i,jstrV)
          enddo
        endif
        if (jend.eq.Mm) then
          do i=istr,iend
            wrk1(i,jend+1)=wrk1(i,jend)
            wrk2(i,jend+1)=wrk2(i,jend)
          enddo
        endif


        do j=jstrV-1,jend
          do i=istr,iend
            cff=FlxV(i,j,k)+FlxV(i,j+1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j+1))
            VFe(i,j)=0.25*( cff*(v(i,j,k)+v(i,j+1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j )
     & +min(cff,0.)*wrk1(i,j+1)
     & ))

            VFe_mix(i,j)=VFe(i,j)-0.25*( v(i,j,k)+v(i,j+1,k)
     & -0.125*(wrk1(i,j)+wrk1(i,j+1))
     & )*( FlxV(i,j,k)+FlxV(i,j+1,k)
     & -0.125*(wrk2(i,j)+wrk2(i,j+1)))

          enddo
        enddo






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




        do j=jmin,jmax
          do i=istrU,iend
            wrk1(i,j)=u(i,j-1,k)-2.*u(i,j,k)
     & +u(i,j+1,k)
          enddo
        enddo

        if (jstr.eq.1) then
          do i=istrU,iend
            wrk1(i,jstr-1)=wrk1(i,jstr)
          enddo
        endif
        if (jend.eq.Mm) then
          do i=istrU,iend
            wrk1(i,jend+1)=wrk1(i,jend)
          enddo
        endif

        do j=jstr,jend+1
          do i=istrU-1,iend
           wrk2(i,j)=FlxV(i-1,j,k)-2.*FlxV(i,j,k)+FlxV(i+1,j,k)
          enddo
        enddo

        do j=jstr,jend+1
          do i=istrU,iend

            cff=FlxV(i,j,k)+FlxV(i-1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i-1,j))
            UFe(i,j)=0.25*( cff*(u(i,j,k)+u(i,j-1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j-1)
     & +min(cff,0.)*wrk1(i,j )
     & ))

            UFe_mix(i,j)=UFe(i,j)-0.25*( u(i,j,k)+u(i,j-1,k)
     & -0.125*(wrk1(i,j)+wrk1(i,j-1))
     & )*( FlxV(i,j,k)+FlxV(i-1,j,k)
     & -0.125*(wrk2(i,j)+wrk2(i-1,j)))

          enddo
        enddo






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




        do j=jstrV,jend
          do i=imin,imax
            wrk1(i,j)=v(i-1,j,k)-2.*v(i,j,k)
     & +v(i+1,j,k)
          enddo
        enddo

        if (istr.eq.1) then
          do j=jstrV,jend
            wrk1(istr-1,j)=wrk1(istr,j)
          enddo
        endif
        if (iend.eq.Lm) then
          do j=jstrV,jend
            wrk1(iend+1,j)=wrk1(iend,j)
          enddo
        endif

        do j=jstrV-1,jend
          do i=istr,iend+1
           wrk2(i,j)=FlxU(i,j-1,k)-2.*FlxU(i,j,k)+FlxU(i,j+1,k)
          enddo
        enddo

        do j=jstrV,jend
          do i=istr,iend+1

            cff=FlxU(i,j,k)+FlxU(i,j-1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j-1))
            VFx(i,j)=0.25*( cff*(v(i,j,k)+v(i-1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i-1,j)
     & +min(cff,0.)*wrk1(i ,j)
     & ))

            VFx_mix(i,j)=VFx(i,j)-0.25*( v(i,j,k)+v(i-1,j,k)
     & -0.125*(wrk1(i,j)+wrk1(i-1,j))
     & )*( FlxU(i,j,k)+FlxU(i,j-1,k)
     & -0.125*(wrk2(i,j)+wrk2(i,j-1)))

          enddo
        enddo


        do j=jstr,jend
          do i=istrU,iend
            ru(i,j,k)=ru(i,j,k)-UFx(i,j )+UFx(i-1,j)
     & -UFe(i,j+1)+UFe(i ,j)



              MXadv(i,j,k,1) = -UFx(i,j)+UFx(i-1,j)
              MYadv(i,j,k,1) = -UFe(i,j+1)+UFe(i,j)

              MHdiss(i,j,k,1) = -UFx_mix(i,j)+UFx_mix(i-1,j)
     & -UFe_mix(i,j+1)+UFe_mix(i,j)


          enddo
        enddo
        do j=jstrV,jend
          do i=istr,iend
            rv(i,j,k)=rv(i,j,k)-VFx(i+1,j)+VFx(i,j )
     & -VFe(i ,j)+VFe(i,j-1)

              MXadv(i,j,k,2) = -VFx(i+1,j)+VFx(i,j)
              MYadv(i,j,k,2) = -VFe(i,j)+VFe(i,j-1)

              MHdiss(i,j,k,2) = -VFx_mix(i+1,j)+VFx_mix(i,j)
     & -VFe_mix(i,j)+VFe_mix(i,j-1)

          enddo
        enddo

      enddo
# 569 "R_tools_fort_routines_gula/get_uv_evolution.F"
!
! Dynamic bottom drag coefficient
!

        Zob=0.01
        do j=jstrV-1,jend
          do i=istrU-1,iend



            cff=sqrt( 0.333333333333*(
     & u(i,j,1)**2 +u(i+1,j,1)**2
     & +u(i,j,1)*u(i+1,j,1)
     & +v(i,j,1)**2+v(i,j+1,1)**2
     & +v(i,j,1)*v(i,j+1,1)
     & ))

c** VFe(i,j)=rdrg + rdrg2*cff

            VFe(i,j)=rdrg + cff*(vonKar/log(Hz(i,j,1)/Zob))**2



          enddo
        enddo

! do j=jmin+1,jmax
! VFe(imax ,j)=VFe(imax-1 ,j)
! VFe(imin ,j)=VFe(imin+1 ,j)
! enddo
!
! do i=imin+1,imax
! VFe(i ,jmax)=VFe(i ,jmax-1)
! VFe(i ,jmin)=VFe(i ,jmin+1)
! enddo
!
!
# 618 "R_tools_fort_routines_gula/get_uv_evolution.F"
      do j=jstr,jend

!
! Compute and add in vertical advection terms:
!




        do i=istrU,iend
          DC(i,1)=0.5625*(Hz(i ,j,1)+Hz(i-1,j,1))
     & -0.0625*(Hz(i+1,j,1)+Hz(i-2,j,1))

          FC(i,0)=1.5*u(i,j,1)
          CF(i,1)=0.5




        enddo
        do k=1,N-1,+1 !--> irreversible
          do i=istrU,iend
            DC(i,k+1)=0.5625*(Hz(i ,j,k+1)+Hz(i-1,j,k+1))
     & -0.0625*(Hz(i+1,j,k+1)+Hz(i-2,j,k+1))

            cff=1./(2.*DC(i,k)+DC(i,k+1)*(2.-CF(i,k)))
            CF(i,k+1)=cff*DC(i,k)
            FC(i,k)=cff*( 3.*( DC(i,k )*u(i,j,k+1)
     & +DC(i,k+1)*u(i,j,k ))
     & -DC(i,k+1)*FC(i,k-1))
          enddo
        enddo !--> discard DC, keep CF,FC
        do i=istrU,iend

          FC(i,N)=(3.*u(i,j,N)-FC(i,N-1))/(2.-CF(i,N))



          DC(i,N)=0. !<-- uppermost W*U flux
        enddo
        do k=N-1,1,-1 !--> irreversible
          do i=istrU,iend
            FC(i,k)=FC(i,k)-CF(i,k+1)*FC(i,k+1)

            DC(i,k)=FC(i,k)*( 0.5625*(W(i ,j,k)+W(i-1,j,k))
     & -0.0625*(W(i+1,j,k)+W(i-2,j,k)))

            ru(i,j,k+1)=ru(i,j,k+1) -DC(i,k+1)+DC(i,k)

            MVadv(i,j,k+1,1) = -DC(i,k+1)+DC(i,k)

          enddo
        enddo !--> discard CF,FC
        do i=istrU,iend
          ru(i,j,1)=ru(i,j,1) -DC(i,1)
          MVadv(i,j,1,1) = -DC(i,1)
        enddo !--> discard DC
# 708 "R_tools_fort_routines_gula/get_uv_evolution.F"
        if (j.ge.jstrV) then

          do i=istr,iend
            DC(i,1)=0.5625*(Hz(i ,j,1)+Hz(i,j-1,1))
     & -0.0625*(Hz(i,j+1,1)+Hz(i,j-2,1))

            FC(i,0)=1.5*v(i,j,1)
            CF(i,1)=0.5




          enddo
          do k=1,N-1,+1 !--> irreversible
            do i=istr,iend
              DC(i,k+1)=0.5625*(Hz(i ,j,k+1)+Hz(i,j-1,k+1))
     & -0.0625*(Hz(i,j+1,k+1)+Hz(i,j-2,k+1))

              cff=1./(2.*DC(i,k)+DC(i,k+1)*(2.-CF(i,k)))
              CF(i,k+1)=cff*DC(i,k)
              FC(i,k)=cff*( 3.*( DC(i,k )*v(i,j,k+1)
     & +DC(i,k+1)*v(i,j,k ))
     & -DC(i,k+1)*FC(i,k-1))
            enddo
          enddo !--> discard DC, keep CF,FC
          do i=istr,iend

            FC(i,N)=(3.*v(i,j,N)-FC(i,N-1))/(2.-CF(i,N))



            DC(i,N)=0. !<-- uppermost W*V flux
          enddo
          do k=N-1,1,-1 !--> irreversible
            do i=istr,iend
              FC(i,k)=FC(i,k)-CF(i,k+1)*FC(i,k+1)

              DC(i,k)=FC(i,k)*( 0.5625*(W(i,j ,k)+W(i,j-1,k))
     & -0.0625*(W(i,j+1,k)+W(i,j-2,k)))

              rv(i,j,k+1)=rv(i,j,k+1) -DC(i,k+1)+DC(i,k)

                MVadv(i,j,k+1,2) = -DC(i,k+1)+DC(i,k)

            enddo
          enddo !--> discard CF,FC

          do i=istr,iend
            rv(i,j,1)=rv(i,j,1) -DC(i,1)

              MVadv(i,j,1,2) = -DC(i,1)


          enddo !--> discard DC
# 795 "R_tools_fort_routines_gula/get_uv_evolution.F"
        endif







! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! !Update u,v with ru,rv
! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! cff=0.25*dt
! do i=istrU,iend
! DC(i,0)=cff*(pm(i,j)+pm(i-1,j))*(pn(i,j)+pn(i-1,j))
! enddo
! do k=1,N
! do i=istrU,iend
! u(i,j,k)=u(i,j,k)+DC(i,0)*ru(i,j,k)
! enddo
! enddo
! if (j.ge.jstrV) then
! do i=istr,iend
! DC(i,0)=cff*(pm(i,j)+pm(i,j-1))*(pn(i,j)+pn(i,j-1))
! enddo
! do k=1,N
! do i=istr,iend
! v(i,j,k)=v(i,j,k)+DC(i,0)*rv(i,j,k)
! enddo
! enddo
! endif
!
!
! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!






          do i=istrU,iend
            do k=1,N
                u(i,j,k) = u(i,j,k) * 0.5 *
     & (Hz(i,j,k)+Hz(i-1,j,k))

            enddo
         enddo

         if (j.ge.jstrV) then
           do i=istr,iend
             do k=1,N
                v(i,j,k) = v(i,j,k) * 0.5 *
     & (Hz(i,j,k)+Hz(i,j-1,k))

            enddo
           enddo
         endif





!






!
! Start computation of the forcing terms for the 2D (barotropic mode)
! momentum equations: vertically integrate the just computed r.h.s
! "ru" and "rv". Also, if so prescribed, add in the difference
! between surface (wind) and bottom (drag) stresses. The computation
! of the 2D forcing terms will be finalized in "rhs2d" during the
! first barotropic time step, when the barotropically computed r.h.ss
! "rubar", "rvbar" will be subtracted from the vertically integrated
! (here) "rufrc", "rvfrc".
!


          do i=istrU,iend
           DC(i,0)=dt*0.25*(pm(i,j)+pm(i-1,j))*(pn(i,j)+pn(i-1,j))

            FC(i,N-1)=dt *2.*(Akv(i,j,N-1)+Akv(i-1,j,N-1))
     & /( Hz(i,j,N )+Hz(i-1,j,N )
     & +Hz(i,j,N-1)+Hz(i-1,j,N-1))

            cff=1./(0.5*(Hz(i,j,N)+Hz(i-1,j,N))+FC(i,N-1))

            CF(i,N-1)=cff*FC(i,N-1)

!----------------------------------------------------------------------
            DC(i,N)=cff*( u(i,j,N) +DC(i,0)*ru(i,j,N)
     & +dt*sustr(i,j))

! DC(i,N)=cff*( u(i,j,N) *0.5*(Hz(i,j,N)+Hz(i-1,j,N))
! & +DC(i,0)*ru(i,j,N)
! & +dt*sustr(i,j))
!----------------------------------------------------------------------

          enddo
          do k=N-1,2,-1 !--> forward elimination
            do i=istrU,iend
              FC(i,k-1)= dt*2.*(Akv(i,j,k-1)+Akv(i-1,j,k-1))
     & /( Hz(i,j,k )+Hz(i-1,j,k )
     & +Hz(i,j,k-1)+Hz(i-1,j,k-1))

              cff=1./( 0.5*(Hz(i,j,k)+Hz(i-1,j,k)) +FC(i,k-1)
     & +FC(i,k)*(1.-CF(i,k))
     & )
              CF(i,k-1)=cff*FC(i,k-1)

!----------------------------------------------------------------------
              DC(i,k)=cff*( u(i,j,k) +DC(i,0)*ru(i,j,k)
     & +FC(i,k)*DC(i,k+1))

! DC(i,k)=cff*( u(i,j,k) *0.5*(Hz(i,j,k)+Hz(i-1,j,k))
! & +DC(i,0)*ru(i,j,k)
! & +FC(i,k)*DC(i,k+1))

!----------------------------------------------------------------------

            enddo
          enddo
          do i=istrU,iend

!----------------------------------------------------------------------
             DC(i,1)=(u(i,j,1) +DC(i,0)*ru(i,j,1)
     & +FC(i,1)*DC(i,2))
     & /( 0.5*(Hz(i,j,1)+Hz(i-1,j,1))

     & +dt * 0.5*(VFe(i,j)+VFe(i-1,j))

     & +FC(i,1)*(1.-CF(i,1)) )

!
!
! DC(i,1)=(u(i,j,1)*0.5*(Hz(i,j,1)+Hz(i-1,j,1))
! & +DC(i,0)*ru(i,j,1)
! & +FC(i,1)*DC(i,2))
! & /( 0.5*(Hz(i,j,1)+Hz(i-1,j,1))
! # ifdef
! & +dt * 0.5*(VFe(i,j)+VFe(i-1,j))
! # endif
! & +FC(i,1)*(1.-CF(i,1)) )

!----------------------------------------------------------------------

          enddo
          cff=1./dt
          do k=2,N,+1 !--> backsubstitution
            do i=istrU,iend
              DC(i,k)=DC(i,k) +CF(i,k-1)*DC(i,k-1)
!>
              FC(i,k-1)=cff*FC(i,k-1)*(DC(i,k)-DC(i,k-1))
            enddo
          enddo !--> now FC is visc. flux
          do i=istrU,iend
            DC(i,0)=dm_u(i,j)*dn_u(i,j)
            ru(i,j,N)=ru(i,j,N) +DC(i,0)*(sustr(i,j)-FC(i,N-1))
            ru(i,j,1)=ru(i,j,1) +DC(i,0)*( FC(i,1)
     & -0.5*(VFe(i-1,j)+VFe(i,j))*DC(i,1) )



               MVmix(i,j,1,1)= DC(i,0)*( FC(i,1)
     & -0.5*(VFe(i-1,j)+VFe(i,j))*DC(i,1) )
               MVmix(i,j,N,1)= DC(i,0)*(sustr(i,j)-FC(i,N-1))



          enddo
          do k=N-1,2,-1
            do i=istrU,iend

              ru(i,j,k)=ru(i,j,k) +DC(i,0)*(FC(i,k)-FC(i,k-1))

              MVmix(i,j,k,1)=DC(i,0)*(FC(i,k)-FC(i,k-1))

            enddo
          enddo





          if (j.ge.jstrV) then

            do i=istr,iend
              DC(i,0)=dt*0.25*(pm(i,j)+pm(i,j-1))*(pn(i,j)+pn(i,j-1))

              FC(i,N-1)=dt *2.*(Akv(i,j,N-1)+Akv(i,j-1,N-1))
     & /( Hz(i,j,N )+Hz(i,j-1,N )
     & +Hz(i,j,N-1)+Hz(i,j-1,N-1))

              cff=1./(0.5*(Hz(i,j,N)+Hz(i,j-1,N))+FC(i,N-1))

              CF(i,N-1)=cff*FC(i,N-1)
              DC(i,N)=cff*( v(i,j,N) +DC(i,0)*rv(i,j,N)
     & +dt*svstr(i,j))
            enddo
            do k=N-1,2,-1 !--> forward elimination
              do i=istr,iend
                FC(i,k-1)= dt*2.*(Akv(i,j,k-1)+Akv(i,j-1,k-1))
     & /( Hz(i,j,k )+Hz(i,j-1,k )
     & +Hz(i,j,k-1)+Hz(i,j-1,k-1))

                cff=1./( 0.5*(Hz(i,j,k)+Hz(i,j-1,k)) +FC(i,k-1)
     & +FC(i,k)*(1.-CF(i,k))
     & )
                CF(i,k-1)=cff*FC(i,k-1)
                DC(i,k)=cff*( v(i,j,k) +DC(i,0)*rv(i,j,k)
     & +FC(i,k)*DC(i,k+1))
              enddo
            enddo
            do i=istr,iend
               DC(i,1)=( v(i,j,1) +DC(i,0)*rv(i,j,1)
     & +FC(i,1)*DC(i,2))
     & /( 0.5*(Hz(i,j,1)+Hz(i,j-1,1))

     & +dt*0.5*(VFe(i,j)+VFe(i,j-1))

     & +FC(i,1)*(1.-CF(i,1)) )
            enddo
            cff=1./dt
            do k=2,N,+1 !<-- backsubstitution
              do i=istr,iend
                DC(i,k)=DC(i,k) +CF(i,k-1)*DC(i,k-1)
!>
                FC(i,k-1)=cff*FC(i,k-1)*(DC(i,k)-DC(i,k-1))
              enddo
            enddo !--> now FC is visc. flux

            do i=istr,iend
              DC(i,0)=dm_v(i,j)*dn_v(i,j)
              rv(i,j,N)=rv(i,j,N) +DC(i,0)*(svstr(i,j)-FC(i,N-1))
              rv(i,j,1)=rv(i,j,1) +DC(i,0)*( FC(i,1)
     & -0.5*(VFe(i,j-1)+VFe(i,j))*DC(i,1) )


                MVmix(i,j,1,2)= DC(i,0)*( FC(i,1)
     & -0.5*(VFe(i,j-1)+VFe(i,j))*DC(i,1) )
                MVmix(i,j,N,2)= DC(i,0)*(svstr(i,j)-FC(i,N-1))


            enddo
            do k=N-1,2,-1
              do i=istr,iend
                rv(i,j,k)=rv(i,j,k) +DC(i,0)*(FC(i,k)-FC(i,k-1))

                MVmix(i,j,k,2)=DC(i,0)*(FC(i,k)-FC(i,k-1))


              enddo
            enddo
          endif




      enddo !<-- j
# 1067 "R_tools_fort_routines_gula/get_uv_evolution.F"
!---------------------------------------------------------------
! Terms in visc3d...
!---------------------------------------------------------------



      do j=jstr,jend

          do i=istrU,iend
            do k=1,N
                u(i,j,k) = u(i,j,k) / ( 0.5 *
     & (Hz(i,j,k)+Hz(i-1,j,k)))

            enddo
         enddo

         if (j.ge.jstrV) then
           do i=istr,iend
             do k=1,N
                v(i,j,k) = v(i,j,k) / ( 0.5 *
     & (Hz(i,j,k)+Hz(i,j-1,k)))

            enddo
           enddo
         endif

      enddo !<-- j


!---------------------------------------------------------------
! Terms in visc3d...
!---------------------------------------------------------------






        if (visctype.eq.0) then

! Old version with no east sponge

       CALL visc3d_GP (Lm,Mm,N,u,v,z_r,z_w
     & ,pm,pn,f,dt,rmask,umask, vmask
     & ,visc2, v_sponge, coord, coordmax
     & ,MHmix)


        elseif (visctype.eq.1) then

! New version with east sponge + new sponge sixes (1/12)


       CALL visc3d_S (Lm,Mm,N,u,v,z_r,z_w
     & ,pm,pn,f,dt,rmask,umask, vmask
     & ,visc2, v_sponge, coord, coordmax
     & ,MHmix)


! New version with east sponge + new sponge sixes (1/20)

        elseif (visctype.eq.2) then

! New version with east sponge + new sponge sixes (1/12)


       CALL visc3d_S_baham (Lm,Mm,N,u,v,z_r,z_w
     & ,pm,pn,f,dt,rmask,umask, vmask
     & ,visc2, v_sponge, coord, coordmax
     & ,MHmix)

        endif


!---------------------------------------------------------------
! Divide all diagnostic terms by the cell volume Hz/(pm*pn).
! There after the unit of diag terms are :
! (unit of velocity) * s-1 = m * s-2
!---------------------------------------------------------------

       do k=1,N
         do j=jstr,jend
           do i=istr,iend



            cff=0.5*(pm(i,j)+pm(i-1,j))
     & *(pn(i,j)+pn(i-1,j))
     & /(Hz(i,j,k)+Hz(i-1,j,k))

     & *umask(i,j)


            MXadv(i,j,k,1)=MXadv(i,j,k,1)*cff
            MYadv(i,j,k,1)=MYadv(i,j,k,1)*cff
            MVadv(i,j,k,1)=MVadv(i,j,k,1)*cff
            MCor(i,j,k,1)=MCor(i,j,k,1)*cff
            MHdiss(i,j,k,1)=MHdiss(i,j,k,1)*cff
            MHmix(i,j,k,1)=MHmix(i,j,k,1)*cff
            MVmix(i,j,k,1)=MVmix(i,j,k,1)*cff
            MPrsgrd(i,j,k,1)=MPrsgrd(i,j,k,1)*cff


            cff=0.5*(pm(i,j)+pm(i,j-1))
     & *(pn(i,j)+pn(i,j-1))
     & /(Hz(i,j,k)+Hz(i,j-1,k))

     & *vmask(i,j)


            MXadv(i,j,k,2)=MXadv(i,j,k,2)*cff
            MYadv(i,j,k,2)=MYadv(i,j,k,2)*cff
            MVadv(i,j,k,2)=MVadv(i,j,k,2)*cff
            MCor(i,j,k,2)=MCor(i,j,k,2)*cff
            MHdiss(i,j,k,2)=MHdiss(i,j,k,2)*cff
            MHmix(i,j,k,2)=MHmix(i,j,k,2)*cff
            MVmix(i,j,k,2)=MVmix(i,j,k,2)*cff
            MPrsgrd(i,j,k,2)=MPrsgrd(i,j,k,2)*cff


           enddo
         enddo

       !write(*,*) 'the end..',k,N

       enddo



!---------------------------------------------------------------






      return
      end

     &
# 151 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_uv_evolution_old.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute terms of the momentum equations in ROMS
!!
!! Note that MHdiss is the difference between upstream and 4th order centered advection (not to be included in the total)
!!
!! Note that MHmix is the sponge dissipation
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_uv_evolution_old (Lm,Mm,N,u,v,T,S, z_r,z_w
     & ,pm,pn,f,dt,rmask
     & ,rdrg, rho0, W, Akv, sustr, svstr
     & ,visc2, v_sponge, coord, coordmax
     & ,MXadv, MYadv, MVadv, MHdiss, MHmix, MVmix, MCor, MPrsgrd)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,dt,istrU,jstrV,
     & itrc

        !INPUT
      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N)
     & ,z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N)
     & ,pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1)
     & ,rmask(0:Lm+1,0:Mm+1)
     & ,umask(1:Lm+1,0:Mm+1), vmask(0:Lm+1,1:Mm+1)
     & ,f(0:Lm+1,0:Mm+1)
     & ,W(0:Lm+1,0:Mm+1,0:N)
     & ,Akv(0:Lm+1,0:Mm+1,0:N)
     & ,sustr(1:Lm+1,0:Mm+1), svstr(0:Lm+1,1:Mm+1)

      integer coord(4), coordmax(4)

      ! OUTPUTS
      real*8 MXadv(0:Lm+1,0:Mm+1,N,2)
     & ,MYadv(0:Lm+1,0:Mm+1,N,2)
     & ,MVadv(0:Lm+1,0:Mm+1,N,2)
     & ,MHdiss(0:Lm+1,0:Mm+1,N,2)
     & ,MHmix(0:Lm+1,0:Mm+1,N,2)
     & ,MVmix(0:Lm+1,0:Mm+1,N,2)
     & ,MCor(0:Lm+1,0:Mm+1,N,2)
     & ,MPrsgrd(0:Lm+1,0:Mm+1,N,2)


      real*8 rdrg, Zob

      real*8 FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & dm_u(0:Lm+1,0:Mm+1), dn_v(0:Lm+1,0:Mm+1),
     & fomn(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2

      real*8 ru(1:Lm+1,0:Mm+1,N), rv(0:Lm+1,1:Mm+1,N)

      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1),
     & UFx_mix(0:Lm+1,0:Mm+1), UFe_mix(0:Lm+1,0:Mm+1),
     & VFx_mix(0:Lm+1,0:Mm+1), VFe_mix(0:Lm+1,0:Mm+1),
     & dmde(1:Lm,1:Mm), dndx(1:Lm,1:Mm)

      real*8 FC(0:Lm+1,0:N), DC(0:Lm+1,0:N)
     & ,CF(0:Lm+1,0:N)

      real visc2, v_sponge


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & rho1(0:Lm+1,0:Mm+1,N), qp1(0:Lm+1,0:Mm+1,N),
     & Tt,Ts,sqrtTs, rho0, K0, dr00,
     & cfr, HalfGRho, GRho

      real*8 P(0:Lm+1,0:Mm+1,N),
     & rho(0:Lm+1,0:Mm+1,N), dpth,
     & dR(0:Lm+1,N), dZ(0:Lm+1,N),
     & dZx(0:Lm+1,0:Mm+1),
     & rx(0:Lm+2,0:Mm+2), dRx(0:Lm+1,0:Mm+1)

      real*8, parameter :: OneFifth=0.2, OneTwelfth=1./12., epsil=0.

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
     & qp2=0.0000172


      real rho1_0, K0_Duk



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



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
# 112 "R_tools_fort_routines_gula/get_uv_evolution_old.F" 2

      parameter (gamma=0.25)







!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,u,v,T,S,z_r,z_w,pm,pn,f,dt,rdrg,rho0,W,Akv,sustr, svstr,visc2, v_sponge
Cf2py intent(out) MXadv, MYadv, MVadv, MHdiss, MHmix, Mcor, MVmix, MPrsgrd


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        imin=0
        imax=Lm+1
        jmin=0
        jmax=Mm+1

        istr=1
        iend=Lm
        jstr=1
        jend=Mm

        istrU=2
        jstrV=2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jmin,jmax
        do i=imin,imax
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo
           fomn(i,j)=f(i,j)/(pm(i,j)*pn(i,j))
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jmin,jmax
        do i=imin,imax
          do k=0,N,+1
              W(i,j,k)=W(i,j,k)/(pm(i,j)*pn(i,j))
            enddo
          enddo
        enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        do j=jmin,jmax
          do i=imin+1,imax
            dm_u(i,j) = 2./(pm(i,j)+pm(i-1,j))
            umask(i,j) = rmask(i,j)*rmask(i-1,j)

          enddo
        enddo

        do j=jmin+1,jmax
          do i=imin,imax
            dn_v(i,j) = 2./(pn(i,j)+pn(i,j-1))
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
          enddo
        enddo



      do j=jmin,jmax
        do i=imin+1,imax
            dn_u(i,j) = 2./(pn(i,j)+pn(i-1,j))
            do k=1,N,+1
              u(i,j,k) = u(i,j,k) * umask(i,j)
              FlxU(i,j,k) = 0.5*(Hz(i,j,k)+Hz(i-1,j,k))*dn_u(i,j)
     & * u(i,j,k)
            enddo
          enddo
      enddo



      do j=jmin+1,jmax
        do i=imin,imax
            dm_v(i,j) = 2./(pm(i,j)+pm(i,j-1))
            do k=1,N,+1
              v(i,j,k) = v(i,j,k) * vmask(i,j)
              FlxV(i,j,k) = 0.5*(Hz(i,j,k)+Hz(i,j-1,k))*dm_v(i,j)
     & * v(i,j,k)
            enddo
          enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 1 "R_tools_fort_routines_gula/compute_prsgrd_old.h" 1






!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE DENSITY
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




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
      enddo ! <-- j


!
! A non-conservative Density-Jacobian scheme using cubic polynomial
! fits for rho and z_r as functions of nondimensianal coordinates xi,
! eta, and s (basically their respective fortran indices). The cubic
! polynomials are constructed by specifying first derivatives of
! interpolated fields on co-located (non-staggered) grid. These
! derivatives are computed using harmonic (rather that algebraic)
! averaging of elementary differences, which guarantees monotonicity
! of the resultant interpolant.
!
! In the code below, if CPP-switch is defined, the Equation
! of State (EOS) is assumed to have form
!
! rho(T,S,z) = rho1(T,S) + qp1(T,S)*dpth*[1.-qp2*dpth]
!
! where rho1 is potential density at 1 atm and qp1 is compressibility
! coefficient, which does not depend on z, and dpth=zeta-z, and qp2
! is just a constant. In this case
!
! d rho d rho1 d qp1 d z
! ------- = ------ + ----- *dpth*[..] - qp1*[1.-2.*qp2*dpth]*------
! d s,x d s,x d s,x d s,x
!
! |<--- adiabatic part --->| |<--- compressible part --->|
!
! where the first two terms constitute "adiabatic derivative" of
! density, which is subject to harmonic averaging, while the last
! term is added in later. This approach quarantees that density
! profile reconstructed by cubic polynomial maintains its positive
! statification in physical sense as long as discrete values of
! density are positively stratified.
!
! This scheme retains exact antisymmetry J(rho,z_r)=-J(z_r,rho)
! [with the exception of harmonic averaging algorithm in the case
! when CPP-switch is defined, see above]. If parameter
! OneFifth (see above) is set to zero, the scheme becomes identical
! to standard Jacobian.
!
! NOTE: This routine is an alternative form of prsgrd32 and it
! produces results identical to that if its prototype.
!

!

!
! Preliminary step (same for XI- and ETA-components:
!------------ ---- ----- --- --- --- ---------------
!
      GRho=g/rho0
      HalfGRho=0.5*GRho

      do j=jstrV-1,jend
        do k=1,N-1
          do i=istrU-1,iend
            dZ(i,k)=z_r(i,j,k+1)-z_r(i,j,k)

            dpth=z_w(i,j,N)-0.5*(z_r(i,j,k+1)+z_r(i,j,k))

            dR(i,k)=rho1(i,j,k+1)-rho1(i,j,k) ! Elementary
     & +(qp1(i,j,k+1)-qp1(i,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ! difference



          enddo
        enddo
        do i=istrU-1,iend
          dR(i,N)=dR(i,N-1)
          dR(i,0)=dR(i,1)
          dZ(i,N)=dZ(i,N-1)
          dZ(i,0)=dZ(i,1)
        enddo
        do k=N,1,-1 !--> irreversible
          do i=istrU-1,iend
            cff=2.*dZ(i,k)*dZ(i,k-1)
            dZ(i,k)=cff/(dZ(i,k)+dZ(i,k-1))

            cfr=2.*dR(i,k)*dR(i,k-1)
            if (cfr.gt.epsil) then
              dR(i,k)=cfr/(dR(i,k)+dR(i,k-1))
            else
              dR(i,k)=0.
            endif

            dpth=z_w(i,j,N)-z_r(i,j,k)
            dR(i,k)=dR(i,k) -qp1(i,j,k)*dZ(i,k)*(1.-2.*qp2*dpth)
            rho(i,j,k)=rho1(i,j,k) +qp1(i,j,k)*dpth*(1.-qp2*dpth)

          enddo
        enddo
        do i=istrU-1,iend
          P(i,j,N)=g*z_w(i,j,N) + GRho*( rho(i,j,N)
     & +0.5*(rho(i,j,N)-rho(i,j,N-1))*(z_w(i,j,N)-z_r(i,j,N))
     & /(z_r(i,j,N)-z_r(i,j,N-1)) )*(z_w(i,j,N)-z_r(i,j,N))


        enddo
        do k=N-1,1,-1
          do i=istrU-1,iend
            P(i,j,k)=P(i,j,k+1)+HalfGRho*( (rho(i,j,k+1)+rho(i,j,k))
     & *(z_r(i,j,k+1)-z_r(i,j,k))

     & -OneFifth*( (dR(i,k+1)-dR(i,k))*( z_r(i,j,k+1)-z_r(i,j,k)
     & -OneTwelfth*(dZ(i,k+1)+dZ(i,k)) )

     & -(dZ(i,k+1)-dZ(i,k))*( rho(i,j,k+1)-rho(i,j,k)
     & -OneTwelfth*(dR(i,k+1)+dR(i,k)) )
     & ))
          enddo
        enddo
      enddo !<-- j


!
! Compute XI-component of pressure gradient term:
!-------- ------------ -- -------- -------- -----
!
      do k=N,1,-1
        do j=jstr,jend
          do i=imin,imax
            FC(i,j)=(z_r(i,j,k)-z_r(i-1,j,k))

            dpth=0.5*( z_w(i,j,N)+z_w(i-1,j,N)
     & -z_r(i,j,k)-z_r(i-1,j,k))

            rx(i,j)=( rho1(i,j,k)-rho1(i-1,j,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i-1,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference



          enddo
        enddo



        do j=jstr,jend
          do i=istrU-1,iend
            cff=2.*FC(i,j)*FC(i+1,j)
            if (cff.gt.epsil) then
              dZx(i,j)=cff/(FC(i,j)+FC(i+1,j))
            else
              dZx(i,j)=0.
            endif

            cfr=2.*rx(i,j)*rx(i+1,j)
            if (cfr.gt.epsil) then
              dRx(i,j)=cfr/(rx(i,j)+rx(i+1,j))
            else
              dRx(i,j)=0.
            endif

            dRx(i,j)=dRx(i,j) -qp1(i,j,k)*dZx(i,j)
     & *(1.-2.*qp2*(z_w(i,j,N)-z_r(i,j,k)))

          enddo !--> discard FC, rx

          do i=istrU,iend
            ru(i,j,k)=0.5*(Hz(i,j,k)+Hz(i-1,j,k))*dn_u(i,j)*(
     & P(i-1,j,k)-P(i,j,k)-HalfGRho*(

     & (rho(i,j,k)+rho(i-1,j,k))*(z_r(i,j,k)-z_r(i-1,j,k))

     & -OneFifth*( (dRx(i,j)-dRx(i-1,j))*( z_r(i,j,k)-z_r(i-1,j,k)
     & -OneTwelfth*(dZx(i,j)+dZx(i-1,j)) )

     & -(dZx(i,j)-dZx(i-1,j))*( rho(i,j,k)-rho(i-1,j,k)
     & -OneTwelfth*(dRx(i,j)+dRx(i-1,j)) )
     & )))


              MPrsgrd(i,j,k,1) = ru(i,j,k)

          enddo
        enddo
!
! ETA-component of pressure gradient term:
!-------------- -- -------- -------- -----
!
        do j=jmin,jmax
          do i=istr,iend
            FC(i,j)=(z_r(i,j,k)-z_r(i,j-1,k))

            dpth=0.5*( z_w(i,j,N)+z_w(i,j-1,N)
     & -z_r(i,j,k)-z_r(i,j-1,k))

            rx(i,j)=( rho1(i,j,k)-rho1(i,j-1,k) ! Elementary
     & +(qp1(i,j,k)-qp1(i,j-1,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ) ! difference





          enddo
        enddo



        do j=jstrV-1,jend
          do i=istr,iend
            cff=2.*FC(i,j)*FC(i,j+1)
            if (cff.gt.epsil) then
c** if ((FC(i,j).gt.0. .and. FC(i,j+1).gt.0.) .or.
c** & (FC(i,j).lt.0. .and. FC(i,j+1).lt.0.)) then
              dZx(i,j)=cff/(FC(i,j)+FC(i,j+1))
            else
              dZx(i,j)=0.
            endif

            cfr=2.*rx(i,j)*rx(i,j+1)
            if (cfr.gt.epsil) then
c** if ((rx(i,j).gt.0. .and. rx(i,j+1).gt.0.) .or.
c** & (rx(i,j).lt.0. .and. rx(i,j+1).lt.0.)) then
              dRx(i,j)=cfr/(rx(i,j)+rx(i,j+1))
            else
              dRx(i,j)=0.
            endif

            dRx(i,j)=dRx(i,j) -qp1(i,j,k)*dZx(i,j)
     & *(1.-2.*qp2*(z_w(i,j,N)-z_r(i,j,k)))

          enddo !--> discard FC, rx

          if (j.ge.jstrV) then
            do i=istr,iend
              rv(i,j,k)=0.5*(Hz(i,j,k)+Hz(i,j-1,k))*dm_v(i,j)*(
     & P(i,j-1,k)-P(i,j,k) -HalfGRho*(

     & (rho(i,j,k)+rho(i,j-1,k))*(z_r(i,j,k)-z_r(i,j-1,k))

     & -OneFifth*( (dRx(i,j)-dRx(i,j-1))*( z_r(i,j,k)-z_r(i,j-1,k)
     & -OneTwelfth*(dZx(i,j)+dZx(i,j-1)) )

     & -(dZx(i,j)-dZx(i,j-1))*( rho(i,j,k)-rho(i,j-1,k)
     & -OneTwelfth*(dRx(i,j)+dRx(i,j-1)) )
     & )))

              MPrsgrd(i,j,k,2) = rv(i,j,k)

            enddo
          endif
        enddo
      enddo
# 214 "R_tools_fort_routines_gula/get_uv_evolution_old.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!

      do j=jmin+1,jmax-1
        do i=imin+1,imax-1
            dmde(i,j) = 0.5/pm(i,j+1)-0.5/pm(i,j-1)
            dndx(i,j) = 0.5/pn(i+1,j)-0.5/pn(i-1,j)
         enddo
      enddo

      do k=1,N
!


!
! Add in Coriolis and curvilinear transformation terms, if any.
!
        do j=jstr,jend
          do i=istr,iend
            cff=0.5*Hz(i,j,k)*(

     & fomn(i,j)


     & +0.5*( (v(i,j,k)+v(i,j+1,k))*dndx(i,j)
     & -(u(i,j,k)+u(i+1,j,k))*dmde(i,j))

     & )
            UFx(i,j)=cff*(v(i,j,k)+v(i,j+1,k))
            VFe(i,j)=cff*(u(i,j,k)+u(i+1,j,k))
          enddo
        enddo
        do j=jstr,jend
          do i=istrU,iend
            ru(i,j,k)=ru(i,j,k)+0.5*(UFx(i,j)+UFx(i-1,j))
             MCor(i,j,k,1) = 0.5*(UFx(i,j)+UFx(i-1,j))
          enddo
        enddo
        do j=jstrV,jend
          do i=istr,iend
            rv(i,j,k)=rv(i,j,k)-0.5*(VFe(i,j)+VFe(i,j-1))
            MCor(i,j,k,2) = -0.5*(VFe(i,j)+VFe(i,j-1))
          enddo
        enddo
# 275 "R_tools_fort_routines_gula/get_uv_evolution_old.F"
!
! Add in horizontal advection of momentum: Compute diagonal [UFx,VFe]
! and off-diagonal [UFe,VFx] components of tensor of momentum flux
! due to horizontal advection; after that add divergence of these
! terms to r.h.s.
!



        do j=jstr,jend
          do i=imin+2,imax-1
            wrk1(i,j)=u(i-1,j,k)-2.*u(i,j,k)
     & +u(i+1,j,k)
            wrk2(i,j)=FlxU(i-1,j,k)-2.*FlxU(i,j,k)
     & +FlxU(i+1,j,k)
          enddo
        enddo



        do j=jmin,jmax
         do i=imin+2,imax-2
            cff=FlxU(i,j,k)+FlxU(i+1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i+1,j))
            UFx(i,j)=0.25*( cff*(u(i,j,k)+u(i+1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i ,j)
     & +min(cff,0.)*wrk1(i+1,j)
     & ))

            UFx_mix(i,j)=UFx(i,j)-0.25*( u(i,j,k)+u(i+1,j,k)
     & -0.125*(wrk1(i,j)+wrk1(i+1,j))
     & )*( FlxU(i,j,k)+FlxU(i+1,j,k)
     & -0.125*(wrk2(i,j)+wrk2(i+1,j)))

          enddo
        enddo






        do j=jmin+2,jmax-1
          do i=imin,imax
            wrk1(i,j)=v(i,j-1,k)-2.*v(i,j,k)+v(i,j+1,k)
            wrk2(i,j)=FlxV(i,j-1,k)-2.*FlxV(i,j,k)+FlxV(i,j+1,k)
          enddo
        enddo


        do j=jmin+2,jmax-2
         do i=imin,imax
            cff=FlxV(i,j,k)+FlxV(i,j+1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j+1))
            VFe(i,j)=0.25*( cff*(v(i,j,k)+v(i,j+1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j )
     & +min(cff,0.)*wrk1(i,j+1)
     & ))

            VFe_mix(i,j)=VFe(i,j)-0.25*( v(i,j,k)+v(i,j+1,k)
     & -0.125*(wrk1(i,j)+wrk1(i,j+1))
     & )*( FlxV(i,j,k)+FlxV(i,j+1,k)
     & -0.125*(wrk2(i,j)+wrk2(i,j+1)))

          enddo
        enddo






        do j=jmin+1,jmax-1
          do i=imin+1,imax
            wrk1(i,j)=u(i,j-1,k)-2.*u(i,j,k)
     & +u(i,j+1,k)
          enddo
        enddo

        do j=jmin+1,jmax
          do i=imin+1,imax-1
           wrk2(i,j)=FlxV(i-1,j,k)-2.*FlxV(i,j,k)+FlxV(i+1,j,k)
          enddo
        enddo

        do j=jmin+2,jmax-1
         do i=imin+2,imax-1

            cff=FlxV(i,j,k)+FlxV(i-1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i-1,j))
            UFe(i,j)=0.25*( cff*(u(i,j,k)+u(i,j-1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j-1)
     & +min(cff,0.)*wrk1(i,j )
     & ))

            UFe_mix(i,j)=UFe(i,j)-0.25*( u(i,j,k)+u(i,j-1,k)
     & -0.125*(wrk1(i,j)+wrk1(i,j-1))
     & )*( FlxV(i,j,k)+FlxV(i-1,j,k)
     & -0.125*(wrk2(i,j)+wrk2(i-1,j)))

          enddo
        enddo






        do j=jmin+1,jmax
         do i=imin+1,imax-1
            wrk1(i,j)=v(i-1,j,k)-2.*v(i,j,k)
     & +v(i+1,j,k)
          enddo
        enddo

        do j=jmin+1,jmax-1
         do i=imin+1,imax
           wrk2(i,j)=FlxU(i,j-1,k)-2.*FlxU(i,j,k)+FlxU(i,j+1,k)
          enddo
        enddo

        do j=jmin+2,jmax-1
         do i=imin+2,imax-1

            cff=FlxU(i,j,k)+FlxU(i,j-1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j-1))
            VFx(i,j)=0.25*( cff*(v(i,j,k)+v(i-1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i-1,j)
     & +min(cff,0.)*wrk1(i ,j)
     & ))

            VFx_mix(i,j)=VFx(i,j)-0.25*( v(i,j,k)+v(i-1,j,k)
     & -0.125*(wrk1(i,j)+wrk1(i-1,j))
     & )*( FlxU(i,j,k)+FlxU(i,j-1,k)
     & -0.125*(wrk2(i,j)+wrk2(i,j-1)))

          enddo
        enddo


        do j=jmin+2,jmax-2
         do i=imin+2,imax-2
            ru(i,j,k)=ru(i,j,k)-UFx(i,j )+UFx(i-1,j)
     & -UFe(i,j+1)+UFe(i ,j)



              MXadv(i,j,k,1) = -UFx(i,j)+UFx(i-1,j)
              MYadv(i,j,k,1) = -UFe(i,j+1)+UFe(i,j)

              MHdiss(i,j,k,1) = -UFx_mix(i,j)+UFx_mix(i-1,j)
     & -UFe_mix(i,j+1)+UFe_mix(i,j)


          enddo
        enddo

        do j=jmin+2,jmax-2
         do i=imin+2,imax-2
            rv(i,j,k)=rv(i,j,k)-VFx(i+1,j)+VFx(i,j )
     & -VFe(i ,j)+VFe(i,j-1)

              MXadv(i,j,k,2) = -VFx(i+1,j)+VFx(i,j)
              MYadv(i,j,k,2) = -VFe(i,j)+VFe(i,j-1)

              MHdiss(i,j,k,2) = -VFx_mix(i+1,j)+VFx_mix(i,j)
     & -VFe_mix(i,j)+VFe_mix(i,j-1)

          enddo
        enddo

      enddo
# 479 "R_tools_fort_routines_gula/get_uv_evolution_old.F"
!
! Dynamic bottom drag coefficient
!

        Zob=0.01
       do j=jmin+1,jmax-1
         do i=imin+1,imax-1



            cff=sqrt( 0.333333333333*(
     & u(i,j,1)**2 +u(i+1,j,1)**2
     & +u(i,j,1)*u(i+1,j,1)
     & +v(i,j,1)**2+v(i,j+1,1)**2
     & +v(i,j,1)*v(i,j+1,1)
     & ))

c** VFe(i,j)=rdrg + rdrg2*cff

            VFe(i,j)=rdrg + cff*(vonKar/log(Hz(i,j,1)/Zob))**2



          enddo
        enddo

       do j=jmin+1,jmax
            VFe(imax ,j)=VFe(imax-1 ,j)
            VFe(imin ,j)=VFe(imin+1 ,j)
        enddo

       do i=imin+1,imax
            VFe(i ,jmax)=VFe(i ,jmax-1)
            VFe(i ,jmin)=VFe(i ,jmin+1)
        enddo
# 528 "R_tools_fort_routines_gula/get_uv_evolution_old.F"
      do j=jstr,jend

!
! Compute and add in vertical advection terms:
!




        do i=istrU,iend
          DC(i,1)=0.5625*(Hz(i ,j,1)+Hz(i-1,j,1))
     & -0.0625*(Hz(i+1,j,1)+Hz(i-2,j,1))

          FC(i,0)=1.5*u(i,j,1)
          CF(i,1)=0.5




        enddo
        do k=1,N-1,+1 !--> irreversible
          do i=istrU,iend
            DC(i,k+1)=0.5625*(Hz(i ,j,k+1)+Hz(i-1,j,k+1))
     & -0.0625*(Hz(i+1,j,k+1)+Hz(i-2,j,k+1))

            cff=1./(2.*DC(i,k)+DC(i,k+1)*(2.-CF(i,k)))
            CF(i,k+1)=cff*DC(i,k)
            FC(i,k)=cff*( 3.*( DC(i,k )*u(i,j,k+1)
     & +DC(i,k+1)*u(i,j,k ))
     & -DC(i,k+1)*FC(i,k-1))
          enddo
        enddo !--> discard DC, keep CF,FC
        do i=istrU,iend

          FC(i,N)=(3.*u(i,j,N)-FC(i,N-1))/(2.-CF(i,N))



          DC(i,N)=0. !<-- uppermost W*U flux
        enddo
        do k=N-1,1,-1 !--> irreversible
          do i=istrU,iend
            FC(i,k)=FC(i,k)-CF(i,k+1)*FC(i,k+1)

            DC(i,k)=FC(i,k)*( 0.5625*(W(i ,j,k)+W(i-1,j,k))
     & -0.0625*(W(i+1,j,k)+W(i-2,j,k)))

            ru(i,j,k+1)=ru(i,j,k+1) -DC(i,k+1)+DC(i,k)

            MVadv(i,j,k+1,1) = -DC(i,k+1)+DC(i,k)

          enddo
        enddo !--> discard CF,FC
        do i=istrU,iend
          ru(i,j,1)=ru(i,j,1) -DC(i,1)
          MVadv(i,j,1,1) = -DC(i,1)
        enddo !--> discard DC
# 618 "R_tools_fort_routines_gula/get_uv_evolution_old.F"
        if (j.ge.jstrV) then

          do i=istr,iend
            DC(i,1)=0.5625*(Hz(i ,j,1)+Hz(i,j-1,1))
     & -0.0625*(Hz(i,j+1,1)+Hz(i,j-2,1))

            FC(i,0)=1.5*v(i,j,1)
            CF(i,1)=0.5




          enddo
          do k=1,N-1,+1 !--> irreversible
            do i=istr,iend
              DC(i,k+1)=0.5625*(Hz(i ,j,k+1)+Hz(i,j-1,k+1))
     & -0.0625*(Hz(i,j+1,k+1)+Hz(i,j-2,k+1))

              cff=1./(2.*DC(i,k)+DC(i,k+1)*(2.-CF(i,k)))
              CF(i,k+1)=cff*DC(i,k)
              FC(i,k)=cff*( 3.*( DC(i,k )*v(i,j,k+1)
     & +DC(i,k+1)*v(i,j,k ))
     & -DC(i,k+1)*FC(i,k-1))
            enddo
          enddo !--> discard DC, keep CF,FC
          do i=istr,iend

            FC(i,N)=(3.*v(i,j,N)-FC(i,N-1))/(2.-CF(i,N))



            DC(i,N)=0. !<-- uppermost W*V flux
          enddo
          do k=N-1,1,-1 !--> irreversible
            do i=istr,iend
              FC(i,k)=FC(i,k)-CF(i,k+1)*FC(i,k+1)

              DC(i,k)=FC(i,k)*( 0.5625*(W(i,j ,k)+W(i,j-1,k))
     & -0.0625*(W(i,j+1,k)+W(i,j-2,k)))

              rv(i,j,k+1)=rv(i,j,k+1) -DC(i,k+1)+DC(i,k)

                MVadv(i,j,k+1,2) = -DC(i,k+1)+DC(i,k)

            enddo
          enddo !--> discard CF,FC

          do i=istr,iend
            rv(i,j,1)=rv(i,j,1) -DC(i,1)

              MVadv(i,j,1,2) = -DC(i,1)


          enddo !--> discard DC
# 705 "R_tools_fort_routines_gula/get_uv_evolution_old.F"
        endif





! !
! !
! ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! ! !Update u,v with ru,rv
! ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! !
! !
! ! cff=0.25*dt
! ! do i=istrU,iend
! ! DC(i,0)=cff*(pm(i,j)+pm(i-1,j))*(pn(i,j)+pn(i-1,j))
! ! enddo
! ! do k=1,N
! ! do i=istrU,iend
! ! u(i,j,k)=u(i,j,k)+DC(i,0)*ru(i,j,k)
! ! enddo
! ! enddo
! ! if (j.ge.jstrV) then
! ! do i=istr,iend
! ! DC(i,0)=cff*(pm(i,j)+pm(i,j-1))*(pn(i,j)+pn(i,j-1))
! ! enddo
! ! do k=1,N
! ! do i=istr,iend
! ! v(i,j,k)=v(i,j,k)+DC(i,0)*rv(i,j,k)
! ! enddo
! ! enddo
! ! endif
! !
! !
! ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! !





          do i=istrU,iend
            do k=1,N
                u(i,j,k) = u(i,j,k) * 0.5 *
     & (Hz(i,j,k)+Hz(i-1,j,k))

             if (j.ge.jstrV) then
                v(i,j,k) = v(i,j,k) * 0.5 *
     & (Hz(i,j,k)+Hz(i,j-1,k))

             endif

            enddo
         enddo



!
! Start computation of the forcing terms for the 2D (barotropic mode)
! momentum equations: vertically integrate the just computed r.h.s
! "ru" and "rv". Also, if so prescribed, add in the difference
! between surface (wind) and bottom (drag) stresses. The computation
! of the 2D forcing terms will be finalized in "rhs2d" during the
! first barotropic time step, when the barotropically computed r.h.ss
! "rubar", "rvbar" will be subtracted from the vertically integrated
! (here) "rufrc", "rvfrc".
!


          do i=istrU,iend
           DC(i,0)=dt*0.25*(pm(i,j)+pm(i-1,j))*(pn(i,j)+pn(i-1,j))

            FC(i,N-1)=dt *2.*(Akv(i,j,N-1)+Akv(i-1,j,N-1))
     & /( Hz(i,j,N )+Hz(i-1,j,N )
     & +Hz(i,j,N-1)+Hz(i-1,j,N-1))

            cff=1./(0.5*(Hz(i,j,N)+Hz(i-1,j,N))+FC(i,N-1))

            CF(i,N-1)=cff*FC(i,N-1)

!----------------------------------------------------------------------
            DC(i,N)=cff*( u(i,j,N) +DC(i,0)*ru(i,j,N)
     & +dt*sustr(i,j))

! DC(i,N)=cff*( u(i,j,N) *0.5*(Hz(i,j,N)+Hz(i-1,j,N))
! & +DC(i,0)*ru(i,j,N)
! & +dt*sustr(i,j))
!----------------------------------------------------------------------

          enddo
          do k=N-1,2,-1 !--> forward elimination
            do i=istrU,iend
              FC(i,k-1)= dt*2.*(Akv(i,j,k-1)+Akv(i-1,j,k-1))
     & /( Hz(i,j,k )+Hz(i-1,j,k )
     & +Hz(i,j,k-1)+Hz(i-1,j,k-1))

              cff=1./( 0.5*(Hz(i,j,k)+Hz(i-1,j,k)) +FC(i,k-1)
     & +FC(i,k)*(1.-CF(i,k))
     & )
              CF(i,k-1)=cff*FC(i,k-1)

!----------------------------------------------------------------------
              DC(i,k)=cff*( u(i,j,k) +DC(i,0)*ru(i,j,k)
     & +FC(i,k)*DC(i,k+1))

! DC(i,k)=cff*( u(i,j,k) *0.5*(Hz(i,j,k)+Hz(i-1,j,k))
! & +DC(i,0)*ru(i,j,k)
! & +FC(i,k)*DC(i,k+1))

!----------------------------------------------------------------------

            enddo
          enddo
          do i=istrU,iend

!----------------------------------------------------------------------
             DC(i,1)=(u(i,j,1) +DC(i,0)*ru(i,j,1)
     & +FC(i,1)*DC(i,2))
     & /( 0.5*(Hz(i,j,1)+Hz(i-1,j,1))

     & +dt * 0.5*(VFe(i,j)+VFe(i-1,j))

     & +FC(i,1)*(1.-CF(i,1)) )

!
!
! DC(i,1)=(u(i,j,1)*0.5*(Hz(i,j,1)+Hz(i-1,j,1))
! & +DC(i,0)*ru(i,j,1)
! & +FC(i,1)*DC(i,2))
! & /( 0.5*(Hz(i,j,1)+Hz(i-1,j,1))
! # ifdef
! & +dt * 0.5*(VFe(i,j)+VFe(i-1,j))
! # endif
! & +FC(i,1)*(1.-CF(i,1)) )

!----------------------------------------------------------------------

          enddo
          cff=1./dt
          do k=2,N,+1 !--> backsubstitution
            do i=istrU,iend
              DC(i,k)=DC(i,k) +CF(i,k-1)*DC(i,k-1)
!>
              FC(i,k-1)=cff*FC(i,k-1)*(DC(i,k)-DC(i,k-1))
            enddo
          enddo !--> now FC is visc. flux
          do i=istrU,iend
            DC(i,0)=dm_u(i,j)*dn_u(i,j)
            ru(i,j,N)=ru(i,j,N) +DC(i,0)*(sustr(i,j)-FC(i,N-1))
            ru(i,j,1)=ru(i,j,1) +DC(i,0)*( FC(i,1)
     & -0.5*(VFe(i-1,j)+VFe(i,j))*DC(i,1) )



               MVmix(i,j,1,1)= DC(i,0)*( FC(i,1)
     & -0.5*(VFe(i-1,j)+VFe(i,j))*DC(i,1) )
               MVmix(i,j,N,1)= DC(i,0)*(sustr(i,j)-FC(i,N-1))






          enddo
          do k=N-1,2,-1
            do i=istrU,iend

              ru(i,j,k)=ru(i,j,k) +DC(i,0)*(FC(i,k)-FC(i,k-1))

              MVmix(i,j,k,1)=DC(i,0)*(FC(i,k)-FC(i,k-1))

            enddo
          enddo





          if (j.ge.jstrV) then

            do i=istr,iend
              DC(i,0)=dt*0.25*(pm(i,j)+pm(i,j-1))*(pn(i,j)+pn(i,j-1))

              FC(i,N-1)=dt *2.*(Akv(i,j,N-1)+Akv(i,j-1,N-1))
     & /( Hz(i,j,N )+Hz(i,j-1,N )
     & +Hz(i,j,N-1)+Hz(i,j-1,N-1))

              cff=1./(0.5*(Hz(i,j,N)+Hz(i,j-1,N))+FC(i,N-1))

              CF(i,N-1)=cff*FC(i,N-1)
              DC(i,N)=cff*( v(i,j,N) +DC(i,0)*rv(i,j,N)
     & +dt*svstr(i,j))
            enddo
            do k=N-1,2,-1 !--> forward elimination
              do i=istr,iend
                FC(i,k-1)= dt*2.*(Akv(i,j,k-1)+Akv(i,j-1,k-1))
     & /( Hz(i,j,k )+Hz(i,j-1,k )
     & +Hz(i,j,k-1)+Hz(i,j-1,k-1))

                cff=1./( 0.5*(Hz(i,j,k)+Hz(i,j-1,k)) +FC(i,k-1)
     & +FC(i,k)*(1.-CF(i,k))
     & )
                CF(i,k-1)=cff*FC(i,k-1)
                DC(i,k)=cff*( v(i,j,k) +DC(i,0)*rv(i,j,k)
     & +FC(i,k)*DC(i,k+1))
              enddo
            enddo
            do i=istr,iend
               DC(i,1)=( v(i,j,1) +DC(i,0)*rv(i,j,1)
     & +FC(i,1)*DC(i,2))
     & /( 0.5*(Hz(i,j,1)+Hz(i,j-1,1))

     & +dt*0.5*(VFe(i,j)+VFe(i,j-1))

     & +FC(i,1)*(1.-CF(i,1)) )
            enddo
            cff=1./dt
            do k=2,N,+1 !<-- backsubstitution
              do i=istr,iend
                DC(i,k)=DC(i,k) +CF(i,k-1)*DC(i,k-1)
!>
                FC(i,k-1)=cff*FC(i,k-1)*(DC(i,k)-DC(i,k-1))
              enddo
            enddo !--> now FC is visc. flux

            do i=istr,iend
              DC(i,0)=dm_v(i,j)*dn_v(i,j)
              rv(i,j,N)=rv(i,j,N) +DC(i,0)*(svstr(i,j)-FC(i,N-1))
              rv(i,j,1)=rv(i,j,1) +DC(i,0)*( FC(i,1)
     & -0.5*(VFe(i,j-1)+VFe(i,j))*DC(i,1) )


                MVmix(i,j,1,2)= DC(i,0)*( FC(i,1)
     & -0.5*(VFe(i,j-1)+VFe(i,j))*DC(i,1) )
                MVmix(i,j,N,2)= DC(i,0)*(svstr(i,j)-FC(i,N-1))



            enddo
            do k=N-1,2,-1
              do i=istr,iend
                rv(i,j,k)=rv(i,j,k) +DC(i,0)*(FC(i,k)-FC(i,k-1))

                MVmix(i,j,k,2)=DC(i,0)*(FC(i,k)-FC(i,k-1))


              enddo
            enddo
          endif




      enddo !<-- j
# 968 "R_tools_fort_routines_gula/get_uv_evolution_old.F"
!---------------------------------------------------------------
! Terms in visc3d...
!---------------------------------------------------------------
# 988 "R_tools_fort_routines_gula/get_uv_evolution_old.F"
       CALL visc3d_S (Lm,Mm,N,u,v,z_r
     & ,pm,pn,f,dt,rmask,umask, vmask
     & ,visc2, v_sponge, coord, coordmax
     & ,MHmix)
# 1005 "R_tools_fort_routines_gula/get_uv_evolution_old.F"
!---------------------------------------------------------------
! Divide all diagnostic terms by the cell volume Hz/(pm*pn).
! There after the unit of diag terms are :
! (unit of velocity) * s-1 = m * s-2
!---------------------------------------------------------------

       do k=1,N
         do j=jstr,jend
           do i=istr,iend



            cff=0.5*(pm(i,j)+pm(i-1,j))
     & *(pn(i,j)+pn(i-1,j))
     & /(Hz(i,j,k)+Hz(i-1,j,k))

     & *umask(i,j)


            MXadv(i,j,k,1)=MXadv(i,j,k,1)*cff
            MYadv(i,j,k,1)=MYadv(i,j,k,1)*cff
            MVadv(i,j,k,1)=MVadv(i,j,k,1)*cff
            MCor(i,j,k,1)=MCor(i,j,k,1)*cff
            MHdiss(i,j,k,1)=MHdiss(i,j,k,1)*cff
            MHmix(i,j,k,1)=MHmix(i,j,k,1)*cff
            MVmix(i,j,k,1)=MVmix(i,j,k,1)*cff
            MPrsgrd(i,j,k,1)=MPrsgrd(i,j,k,1)*cff


            cff=0.5*(pm(i,j)+pm(i,j-1))
     & *(pn(i,j)+pn(i,j-1))
     & /(Hz(i,j,k)+Hz(i,j-1,k))

     & *vmask(i,j)


            MXadv(i,j,k,2)=MXadv(i,j,k,2)*cff
            MYadv(i,j,k,2)=MYadv(i,j,k,2)*cff
            MVadv(i,j,k,2)=MVadv(i,j,k,2)*cff
            MCor(i,j,k,2)=MCor(i,j,k,2)*cff
            MHdiss(i,j,k,2)=MHdiss(i,j,k,2)*cff
            MHmix(i,j,k,2)=MHmix(i,j,k,2)*cff
            MVmix(i,j,k,2)=MVmix(i,j,k,2)*cff
            MPrsgrd(i,j,k,2)=MPrsgrd(i,j,k,2)*cff


           enddo
         enddo

       !write(*,*) 'the end..',k,N

       enddo



!---------------------------------------------------------------






      return
      end

     &
# 153 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines/get_hbl.F" 1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subpart of the lmd_kpp.F routine (myroms version)
! used to compute the new hbl
! (the part used to compute the new Kv, Kt has been removed)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





c----#define WND_AT_RHO_POINTS

      subroutine get_hbl (Lm,Mm,N,alpha,beta, z_r,z_w
     & , stflx, srflx, swr_frac, sustr, svstr ,Ricr, hbls, f
     & , u, v, bvf
     & , hbl, out1, out2, out3, out4)

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
# 39 "R_tools_fort_routines/get_hbl.F" 2

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
      real*8 hbl(0:Lm+1,0:Mm+1)
     & , out1(0:Lm+1,0:Mm+1,0:N), out2(0:Lm+1,0:Mm+1,0:N)
     & , out3(0:Lm+1,0:Mm+1,0:N), out4(0:Lm+1,0:Mm+1,0:N)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,alpha,beta ,z_r,z_w,stflx,srflx, swr_frac, sustr, svstr ,Ricr,hbls, f, u, v, bvf
Cf2py intent(out) hbl, out1, out2, out3, out4

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
# 214 "R_tools_fort_routines/get_hbl.F" 2



          cff=bvf(i,j,k)*bvf(i,j,k-1)
          if (cff.gt.0.D0) then
            cff=cff/(bvf(i,j,k)+bvf(i,j,k-1))
          else
            cff=0.D0
          endif


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


          out1(i,j,k-1) = out1(i,j,k)+ Kern*Hz(i,j,k)*(
     & 0.375*( wrk1(i,k)**2+wrk1(i,k-1)**2
     & +wrk2(i,k)**2 +wrk2(i,k-1)**2 )
     & +0.25 *(wrk1(i,k-1)*wrk1(i,k)+wrk2(i,k-1)*wrk2(i,k))
     & )

          out2(i,j,k-1) = out2(i,j,k)+ Kern*Hz(i,j,k)*(
     & -Ri_inv*( cff + 0.25*(bvf(i,j,k)+bvf(i,j,k-1)))
     & )

          out3(i,j,k-1) = out3(i,j,k)+ Kern*Hz(i,j,k)*(
     & -C_Ek*f(i,j)*f(i,j)
     & )

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


         FC(i,k-1)= FC(i,k) + Kern * Hz(i,j,k) * (
     & 0.375*( wrk1(i,k)**2 + wrk1(i,k-1)**2
     & +wrk2(i,k)**2 + wrk2(i,k-1)**2 )
     & +0.25 * (wrk1(i,k-1)*wrk1(i,k)+
     & wrk2(i,k-1)*wrk2(i,k))
     & - Ri_inv * (cff + 0.25*(bvf(i,j,k)+bvf(i,j,k-1)))
     & -C_Ek*f(i,j)*f(i,j)
     & )




          Vtsq=Vtc*ws*sqrt(max(0., bvf(i,j,k-1)))
          out4(i,j,k-1) = Vtsq



          Cr(i,k-1)=FC(i,k-1) +Vtsq


          if (kbl(i).eq.0 .and. Cr(i,k-1).lt.0.) kbl(i)=k







        enddo
      enddo






      do i=istr,iend
c?? if (kbl(i).eq.N) then
c?? hbl(i,j)=z_w(i,j,N)-z_w(i,j,N-1)

        if (kbl(i).gt.0) then
          k=kbl(i)

          hbl(i,j)=z_w(i,j,N)-( z_w(i,j,k-1)*Cr(i,k)
     & -z_w(i,j,k)*Cr(i,k-1)
     & )/(Cr(i,k)-Cr(i,k-1))

c** if (Cr(i,k)*Cr(i,k-1).gt.0.D0 ) write(*,*)
c** & '### ERROR', k, Cr(i,k), Cr(i,k-1), hbl(i,j)

        else
          hbl(i,j)=z_w(i,j,N)-z_w(i,j,0)+eps
        endif
# 314 "R_tools_fort_routines/get_hbl.F"
      enddo

!======================================


      enddo !<-- j





# 1 "R_tools_fort_routines/kpp_smooth.h" 1
!
! Apply horizontal smoothing operator to hbl, while avoiding land-
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
          FX(i,j)=(hbl(i,j)-hbl(i-1,j))
        enddo
      enddo
      do j=jstr,jend
        do i=istr,iend
          FE(i,j)=(hbl(i,j)-hbl(i,j-1))
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
          hbl(i,j)=hbl(i,j)+cff1*( FX(i+1,j)-FX(i,j)
     & +FE1(i,j+1)-FE1(i,j))
        enddo !--> discard FX,FE,FE1
      enddo
# 326 "R_tools_fort_routines/get_hbl.F" 2




!======================================


      return
      end
# 155 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_pressure.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute pressure
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_pressure(Lm,Mm,N,T,S,z_r,z_w
     & ,rho0,pm,pn,rmask,P)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & rho1(0:Lm+1,0:Mm+1,N), qp1(0:Lm+1,0:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,0:N),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & rmask(0:Lm+1,0:Mm+1),
     & Tt,Ts,sqrtTs, rho0, K0, dr00,
     & cff ,cff1, cff2, cfr, HalfGRho, GRho,
     & var1, var2,var3, var4

      real*8 P(0:Lm+1,0:Mm+1,N),
     & ru(1:Lm+1,0:Mm+1), rv(0:Lm+1,1:Mm+1),
     & rho(0:Lm+1,0:Mm+1,N), dpth,
     & dR(0:Lm+1,0:N), dZ(0:Lm+1,0:N),
     & FC(0:Lm+2,0:Mm+2), dZx(0:Lm+1,0:Mm+1),
     & rx(0:Lm+2,0:Mm+2), dRx(0:Lm+1,0:Mm+1)

      real*8, parameter :: OneFifth=0.2, OneTwelfth=1./12., epsil=0.

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
     & qp2=0.0000172


      real rho1_0, K0_Duk

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
# 54 "R_tools_fort_routines_gula/get_pressure.F" 2



Cf2py intent(in) Lm,Mm,N, T,S,z_r,z_w,rho0,pm,pn,rmask
Cf2py intent(out) P


!
! A non-conservative Density-Jacobian scheme using cubic polynomial
! fits for rho and z_r as functions of nondimensianal coordinates xi,
! eta, and s (basically their respective fortran indices). The cubic
! polynomials are constructed by specifying first derivatives of
! interpolated fields on co-located (non-staggered) grid. These
! derivatives are computed using harmonic (rather that algebraic)
! averaging of elementary differences, which guarantees monotonicity
! of the resultant interpolant.
!
! In the code below, if CPP-switch is defined, the Equation
! of State (EOS) is assumed to have form
!
! rho(T,S,z) = rho1(T,S) + qp1(T,S)*dpth*[1.-qp2*dpth]
!
! where rho1 is potential density at 1 atm and qp1 is compressibility
! coefficient, which does not depend on z, and dpth=zeta-z, and qp2
! is just a constant. In this case
!
! d rho d rho1 d qp1 d z
! ------- = ------ + ----- *dpth*[..] - qp1*[1.-2.*qp2*dpth]*------
! d s,x d s,x d s,x d s,x
!
! |<--- adiabatic part --->| |<--- compressible part --->|
!
! where the first two terms constitute "adiabatic derivative" of
! density, which is subject to harmonic averaging, while the last
! term is added in later. This approach quarantees that density
! profile reconstructed by cubic polynomial maintains its positive
! statification in physical sense as long as discrete values of
! density are positively stratified.
!
! This scheme retains exact antisymmetry J(rho,z_r)=-J(z_r,rho)
! [with the exception of harmonic averaging algorithm in the case
! when CPP-switch is defined, see above]. If parameter
! OneFifth (see above) is set to zero, the scheme becomes identical
! to standard Jacobian.
!
! NOTE: This routine is an alternative form of prsgrd32 and it
! produces results identical to that if its prototype.
!


        istr=0
        istrU=1
        iend=Lm+1
        jstr=0
        jstrV=1
        jend=Mm+1

        imin=istrU
        imax=iend
        jmin=jstrV
        jmax=jend




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! COMPUTE DENSITY
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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





!---------------------------------------------------------------------------------------


      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

      do j=jmin,jmax


!---------------------------------------------------------------------------------------
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


!---------------------------------------------------------------------------------------

            Hz(i,j,k)=z_w(i,j,k)-z_w(i,j,k-1)




          enddo
        enddo
      enddo ! <-- j



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!
! Preliminary step (same for XI- and ETA-components:
!------------ ---- ----- --- --- --- ---------------
!
      GRho=g/rho0
      HalfGRho=0.5*GRho

      do j=jstrV-1,jend
        do k=1,N-1
          do i=istrU-1,iend
            dZ(i,k)=z_r(i,j,k+1)-z_r(i,j,k)

            dpth=z_w(i,j,N)-0.5*(z_r(i,j,k+1)+z_r(i,j,k))

            dR(i,k)=rho1(i,j,k+1)-rho1(i,j,k) ! Elementary
     & +(qp1(i,j,k+1)-qp1(i,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ! difference



          enddo
        enddo
        do i=istrU-1,iend
          dR(i,N)=dR(i,N-1)
          dR(i,0)=dR(i,1)
          dZ(i,N)=dZ(i,N-1)
          dZ(i,0)=dZ(i,1)
        enddo
        do k=N,1,-1 !--> irreversible
          do i=istrU-1,iend
            cff=2.*dZ(i,k)*dZ(i,k-1)
            dZ(i,k)=cff/(dZ(i,k)+dZ(i,k-1))

            cfr=2.*dR(i,k)*dR(i,k-1)
            if (cfr.gt.epsil) then
              dR(i,k)=cfr/(dR(i,k)+dR(i,k-1))
            else
              dR(i,k)=0.
            endif

            dpth=z_w(i,j,N)-z_r(i,j,k)
            dR(i,k)=dR(i,k) -qp1(i,j,k)*dZ(i,k)*(1.-2.*qp2*dpth)
            rho(i,j,k)=rho1(i,j,k) +qp1(i,j,k)*dpth*(1.-qp2*dpth)

          enddo
        enddo
        do i=istrU-1,iend
          P(i,j,N)=g*z_w(i,j,N) + GRho*( rho(i,j,N)
     & +0.5*(rho(i,j,N)-rho(i,j,N-1))*(z_w(i,j,N)-z_r(i,j,N))
     & /(z_r(i,j,N)-z_r(i,j,N-1)) )*(z_w(i,j,N)-z_r(i,j,N))
        enddo
        do k=N-1,1,-1
          do i=istrU-1,iend
            P(i,j,k)=P(i,j,k+1)+HalfGRho*( (rho(i,j,k+1)+rho(i,j,k))
     & *(z_r(i,j,k+1)-z_r(i,j,k))

     & -OneFifth*( (dR(i,k+1)-dR(i,k))*( z_r(i,j,k+1)-z_r(i,j,k)
     & -OneTwelfth*(dZ(i,k+1)+dZ(i,k)) )

     & -(dZ(i,k+1)-dZ(i,k))*( rho(i,j,k+1)-rho(i,j,k)
     & -OneTwelfth*(dR(i,k+1)+dR(i,k)) )
     & ))
          enddo
        enddo
      enddo !<-- j
# 273 "R_tools_fort_routines_gula/get_pressure.F"
      return
      end
# 157 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_pressure_croco_QH.F" 1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Compute pressure (updated 20/30/16 from croco)
!! for only
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




      subroutine get_pressure_croco_QH(Lm,Mm,N,T,S,
     & z_r,z_w,rho0,pm,pn,rmask,
     & u,v,e,cosa,sina,
     & P)

      implicit none

      integer Lm,Mm,N, istrR,iendR,jstrR,jendR, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV

      real*8 T(0:Lm+1,0:Mm+1,N), S(0:Lm+1,0:Mm+1,N),
     & rho1(0:Lm+1,0:Mm+1,N), qp1(0:Lm+1,0:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & e(0:Lm+1,0:Mm+1),
     & cosa(0:Lm+1,0:Mm+1), sina(0:Lm+1,0:Mm+1),
     & rmask(0:Lm+1,0:Mm+1),
     & Tt,Ts,sqrtTs, rho0, K0, dr00,
     & cff ,cff1, cff2, cfr, HalfGRho, GRho,
     & var1, var2,var3, var4

      real*8 P(0:Lm+1,0:Mm+1,N),
     & rho(0:Lm+1,0:Mm+1,N), dpth,
     & dR(0:Lm+1,0:N), dZ(0:Lm+1,0:N)

      real*8, parameter :: OneFifth=0.2, OneTwelfth=1./12., epsil=0.

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
     & qp2=0.0000172


      real rho1_0, K0_Duk

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
# 57 "R_tools_fort_routines_gula/get_pressure_croco_QH.F" 2



Cf2py intent(in) Lm,Mm,N,T,S,z_r,z_w,rho0,pm,pn,rmask,u,v,e,cosa,sina
Cf2py intent(out) P


!
!======================================================================
! Compute density anomaly via Equation Of State (EOS) for seawater.
! Following Jackett and McDougall, 1995, physical EOS is assumed to
! have form
!
! rho0 + rho1(T,S)
! rho(T,S,z) = ------------------------ (1)
! 1 - 0.1*|z|/K(T,S,|z|)
!
! where rho1(T,S) is sea-water density perturbation [kg/m^3] at
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
! rho(T,S.z) = rho1(T,S) + qp1(T,S)*|z| (5)
!
! where
! rho1 - rho0*K01(T,S)/K00
! qp1(T,S)= 0.1 -------------------------- (6)
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
! If so prescribed compute the Brunt-Väisäla frequency [1/s^2] at
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
! References:
! ----------
! Shchepetkin, A.F., McWilliams, J.C., 2003: A method for computing
! horizontal pressure-gradient force in an oceanic model with a
! non-aligned vertical coordinate. J. Geophys. Res. 108 (C3), 3090.
!
!======================================================================
!

        istr=1
        istrU=1
        iend=Lm
        jstr=1
        jstrV=1
        jend=Mm

        istrR=istr
        iendR=iend
        jstrR=jstr
        jendR=jend


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Compute density (copy-pasted from rho_eos.F CROCO 20/03/16)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      Tt=3.8D0
      Ts=34.5D0
      sqrtTs=sqrt(Ts)
      K0_Duk= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))



!
! compute rho as a perturbation to rho0 (at the surface)
!
      dr00=r00-rho0
!

!---------------------------------------------------------------------------------------

      do j=jstrR,jendR

!---------------------------------------------------------------------------------------
        do k=1,N
          do i=istrR,iendR
            Tt=T(i,j,k)
            Ts=max(S(i,j,k), 0.)
            sqrtTs=sqrt(Ts)
            rho1(i,j,k)=( dr00 +Tt*( r01+Tt*( r02+Tt*( r03+Tt*(
     & r04+Tt*r05 ))))
     & +Ts*( r10+Tt*( r11+Tt*( r12+Tt*(
     & r13+Tt*r14 )))
     & +sqrtTs*(rS0+Tt*(
     & rS1+Tt*rS2 ))+Ts*r20 ))

     & *rmask(i,j)


            K0= Tt*( K01+Tt*( K02+Tt*( K03+Tt*K04 )))
     & +Ts*( K10+Tt*( K11+Tt*( K12+Tt*K13 ))
     & +sqrtTs*( KS0+Tt*( KS1+Tt*KS2 )))





            qp1(i,j,k)= 0.1D0*(rho0+rho1(i,j,k))*(K0_Duk-K0)
     & /((K00+K0)*(K00+K0_Duk))




     & *rmask(i,j)

            dpth=z_w(i,j,N)-z_r(i,j,k)
            rho(i,j,k)=rho1(i,j,k) +qp1(i,j,k)*dpth*(1.-qp2*dpth)
# 234 "R_tools_fort_routines_gula/get_pressure_croco_QH.F"
            rho(i,j,k)=rho(i,j,k)*rmask(i,j)


          enddo
        enddo
      enddo ! <-- j


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!======================================================================
! Quasi-hydrostatique correction for non-traditional Coriolis force
!======================================================================
! dR = -rho0/g* e (U cos(a) - V sin(a) )
! with e = 2 Omega cos(Phi)
! a = angle between North and meridional grid axis
! --> QH pressure gradient is DPdz=-(rho+dR)*g/rho0
!-----------------------------------------------------------------------

      cff=0.5*rho0/g
      do j=jstr,jend
        do k=1,N
          do i=istr,iend
            rho1(i,j,k)=rho1(i,j,k)
     & - cff*e(i,j)* (
     & cosa(i,j)*(u(i,j,k)+u(i+1,j,k))
     & - sina(i,j)*(v(i,j,k)+v(i,j+1,k)) )

            rho1(i,j,k)=rho1(i,j,k)*rmask(i,j)

          enddo
        enddo
      enddo


      do j=jstrR,jendR ! resume j loop
        do k=1,N
          do i=istrR,iendR

            dpth=z_w(i,j,N)-z_r(i,j,k)
            rho(i,j,k)=rho1(i,j,k) +qp1(i,j,k)*dpth*(1.-qp2*dpth)
# 288 "R_tools_fort_routines_gula/get_pressure_croco_QH.F"
            rho(i,j,k)=rho(i,j,k)*rmask(i,j)

          enddo
        enddo
      enddo ! <-- j



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! Preliminary step (same for XI- and ETA-components:
!------------ ---- ----- --- --- --- ---------------
!
      GRho=g/rho0
      HalfGRho=0.5*GRho

      do j=jstrV-1,jend
        do k=1,N-1
          do i=istrU-1,iend
            dZ(i,k)=z_r(i,j,k+1)-z_r(i,j,k)

            dpth=z_w(i,j,N)-0.5*(z_r(i,j,k+1)+z_r(i,j,k))

            dR(i,k)=rho1(i,j,k+1)-rho1(i,j,k) ! Elementary
     & +(qp1(i,j,k+1)-qp1(i,j,k)) ! adiabatic
     & *dpth*(1.-qp2*dpth) ! difference



          enddo
        enddo
        do i=istrU-1,iend
          dR(i,N)=dR(i,N-1)
          dR(i,0)=dR(i,1)
          dZ(i,N)=dZ(i,N-1)
          dZ(i,0)=dZ(i,1)
        enddo
        do k=N,1,-1 !--> irreversible
          do i=istrU-1,iend
            cff=2.*dZ(i,k)*dZ(i,k-1)
            dZ(i,k)=cff/(dZ(i,k)+dZ(i,k-1))

            cfr=2.*dR(i,k)*dR(i,k-1)
            if (cfr.gt.epsil) then
              dR(i,k)=cfr/(dR(i,k)+dR(i,k-1))
            else
              dR(i,k)=0.
            endif

            dpth=z_w(i,j,N)-z_r(i,j,k)
            dR(i,k)=dR(i,k) -qp1(i,j,k)*dZ(i,k)*(1.-2.*qp2*dpth)

          enddo
        enddo

        do i=istrU-1,iend
          P(i,j,N)=g*z_w(i,j,N) + GRho*( rho(i,j,N)
     & +0.5*(rho(i,j,N)-rho(i,j,N-1))*(z_w(i,j,N)-z_r(i,j,N))
     & /(z_r(i,j,N)-z_r(i,j,N-1)) )*(z_w(i,j,N)-z_r(i,j,N))
        enddo
        do k=N-1,1,-1
          do i=istrU-1,iend
            P(i,j,k)=P(i,j,k+1)+HalfGRho*( (rho(i,j,k+1)+rho(i,j,k))
     & *(z_r(i,j,k+1)-z_r(i,j,k))
     & -OneFifth*( (dR(i,k+1)-dR(i,k))*( z_r(i,j,k+1)-z_r(i,j,k)
     & -OneTwelfth*(dZ(i,k+1)+dZ(i,k)) )
     & -(dZ(i,k+1)-dZ(i,k))*( rho(i,j,k+1)-rho(i,j,k)
     & -OneTwelfth*(dR(i,k+1)+dR(i,k)) )
     & ))
          enddo
        enddo
      enddo !<-- j
# 368 "R_tools_fort_routines_gula/get_pressure_croco_QH.F"
      return
      end
# 159 "R_tools_fort_gula.F" 2


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
# 165 "R_tools_fort_gula.F" 2

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
# 167 "R_tools_fort_gula.F" 2

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
# 169 "R_tools_fort_gula.F" 2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# 1 "R_tools_fort_routines_gula/get_kediss.F" 1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! compute Advective part of the barotropic vorticity balance equation
!!
!! Boundary conditions addded on 05/09/14
!!
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      subroutine get_kediss (Lm,Mm,N,u,v, z_r,z_w,pm,pn, rmask
     & ,kediss)

      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & FlxU(1:Lm+1,0:Mm+1,N), FlxV(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N),
     & rmask(0:Lm+1,0:Mm+1),
     & Hz(0:Lm+1,0:Mm+1,N), gamma,
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2



      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1),
     & umask(1:Lm+1,0:Mm+1), vmask(0:Lm+1,1:Mm+1)

      real*8 advu(1:Lm+1,0:Mm+1,N), advv(0:Lm+1,1:Mm+1,N)

      real*8 kediss(0:Lm+1,0:Mm+1,N)

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
# 44 "R_tools_fort_routines_gula/get_kediss.F" 2

      parameter (gamma=0.25)

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
# 50 "R_tools_fort_routines_gula/get_kediss.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Cf2py intent(in) Lm,Mm,N, u,v,z_r,z_w,pm,pn, rmask
Cf2py intent(out) kediss


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Compute grid variables and fluxes


      do j=jstrR,jendR
        do i=istrR,iendR
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo
        enddo
      enddo


      do j=jstrR,jendR
        do i=istr,iendR
            dn_u(i,j) = 2./(pn(i,j)+pn(i-1,j))
            umask(i,j) = rmask(i,j)*rmask(i-1,j)
            do k=1,N,+1
              u(i,j,k) = u(i,j,k) * umask(i,j)
              FlxU(i,j,k) = 0.5*(Hz(i,j,k)+Hz(i-1,j,k))*dn_u(i,j)
     & * u(i,j,k)
            enddo
          enddo
      enddo


      do j=jstr,jendR
        do i=istrR,iendR
            dm_v(i,j) = 2./(pm(i,j)+pm(i,j-1))
            vmask(i,j) = rmask(i,j)*rmask(i,j-1)
            do k=1,N,+1
              v(i,j,k) = v(i,j,k) * vmask(i,j)
              FlxV(i,j,k) = 0.5*(Hz(i,j,k)+Hz(i,j-1,k))*dm_v(i,j)
     & * v(i,j,k)
            enddo
          enddo
      enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do k=1,N

!
! Add in horizontal advection of momentum: Compute diagonal [UFx,VFe]
! and off-diagonal [UFe,VFx] components of tensor of momentum flux
! due to horizontal advection; after that add divergence of these
! terms to r.h.s.
!



        if (istr.eq.1) then ! Sort out bounding indices of
          imin=istrU ! extended ranges: note that in
        else ! the vicinity of physical
          imin=istrU-1 ! boundaries values at the
        endif ! extremal points of stencil
        if (iend.eq.Lm) then ! are not available, so an
          imax=iend ! extrapolation rule needs to
        else ! be applied. Also note that
          imax=iend+1 ! for this purpose periodic
        endif ! ghost points and MPI margins





        do j=jstr,jend
          do i=imin,imax
            wrk1(i,j)=u(i-1,j,k)-2.*u(i,j,k)
     & +u(i+1,j,k)
            wrk2(i,j)=FlxU(i-1,j,k)-2.*FlxU(i,j,k)
     & +FlxU(i+1,j,k)
          enddo
        enddo

        if (istr.eq.1) then
          do j=jstr,jend
            wrk1(istrU-1,j) =wrk1(istrU,j)
            wrk2(istrU-1,j)=wrk2(istrU,j)
          enddo
        endif
        if (iend.eq.Lm) then
          do j=jstr,jend
            wrk1(iend+1,j) =wrk1(iend,j)
            wrk2(iend+1,j)=wrk2(iend,j)
          enddo
        endif




        do j=jstr,jend
          do i=istrU-1,iend

         cff=FlxU(i,j,k)+FlxU(i+1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i+1,j))
         UFx(i,j)=0.25*( cff*(u(i,j,k)+u(i+1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i ,j)
     & +min(cff,0.)*wrk1(i+1,j)
     & ))

         UFx(i,j) = UFx(i,j) - 0.25*( u(i,j,k)+u(i+1,j,k)
     & -0.125*(wrk1(i,j)+wrk1(i+1,j))
     & )*( FlxU(i,j,k)+FlxU(i+1,j,k)
     & -0.125*(wrk2(i,j)+wrk2(i+1,j)))

          enddo
        enddo
# 176 "R_tools_fort_routines_gula/get_kediss.F"
        if (jstr.eq.1) then
          jmin=jstrV
        else
          jmin=jstrV-1
        endif
        if (jend.eq.Mm) then
          jmax=jend
        else
          jmax=jend+1
        endif




        do j=jmin,jmax
          do i=istr,iend
            wrk1(i,j)=v(i,j-1,k)-2.*v(i,j,k)+v(i,j+1,k)
            wrk2(i,j)=FlxV(i,j-1,k)-2.*FlxV(i,j,k)+FlxV(i,j+1,k)
          enddo
        enddo

        if (jstr.eq.1) then
          do i=istr,iend
            wrk1(i,jstrV-1)=wrk1(i,jstrV)
            wrk2(i,jstrV-1)=wrk2(i,jstrV)
          enddo
        endif
        if (jend.eq.Mm) then
          do i=istr,iend
            wrk1(i,jend+1)=wrk1(i,jend)
            wrk2(i,jend+1)=wrk2(i,jend)
          enddo
        endif




        do j=jstrV-1,jend
          do i=istr,iend

            cff=FlxV(i,j,k)+FlxV(i,j+1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j+1))

            VFe(i,j)=0.25*( cff*(v(i,j,k)+v(i,j+1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j )
     & +min(cff,0.)*wrk1(i,j+1)
     & ))

            VFe(i,j)=VFe(i,j) - 0.25*( v(i,j,k)+v(i,j+1,k)
     & -0.125*(wrk1(i,j)+wrk1(i,j+1))
     & )*( FlxV(i,j,k)+FlxV(i,j+1,k)
     & -0.125*(wrk2(i,j)+wrk2(i,j+1)))

          enddo
        enddo
# 241 "R_tools_fort_routines_gula/get_kediss.F"
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




        do j=jmin,jmax
          do i=istrU,iend
            wrk1(i,j)=u(i,j-1,k)-2.*u(i,j,k)
     & +u(i,j+1,k)
          enddo
        enddo

        if (jstr.eq.1) then
          do i=istrU,iend
            wrk1(i,jstr-1)=wrk1(i,jstr)
          enddo
        endif
        if (jend.eq.Mm) then
          do i=istrU,iend
            wrk1(i,jend+1)=wrk1(i,jend)
          enddo
        endif

        do j=jstr,jend+1
          do i=istrU-1,iend
           wrk2(i,j)=FlxV(i-1,j,k)-2.*FlxV(i,j,k)+FlxV(i+1,j,k)
          enddo
        enddo

        do j=jstr,jend+1
          do i=istrU,iend

            cff=FlxV(i,j,k)+FlxV(i-1,j,k)-0.125*( wrk2(i ,j)
     & +wrk2(i-1,j))
            UFe(i,j)=0.25*( cff*(u(i,j,k)+u(i,j-1,k))
     & -gamma*( max(cff,0.)*wrk1(i,j-1)
     & +min(cff,0.)*wrk1(i,j )
     & ))

            UFe(i,j)=UFe(i,j) - 0.25*( u(i,j,k)+u(i,j-1,k)
     & -0.125*(wrk1(i,j)+wrk1(i,j-1))
     & )*( FlxV(i,j,k)+FlxV(i-1,j,k)
     & -0.125*(wrk2(i,j)+wrk2(i-1,j)))

          enddo
        enddo







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




        do j=jstrV,jend
          do i=imin,imax
            wrk1(i,j)=v(i-1,j,k)-2.*v(i,j,k)
     & +v(i+1,j,k)
          enddo
        enddo

        if (istr.eq.1) then
          do j=jstrV,jend
            wrk1(istr-1,j)=wrk1(istr,j)
          enddo
        endif
        if (iend.eq.Lm) then
          do j=jstrV,jend
            wrk1(iend+1,j)=wrk1(iend,j)
          enddo
        endif

        do j=jstrV-1,jend
          do i=istr,iend+1
           wrk2(i,j)=FlxU(i,j-1,k)-2.*FlxU(i,j,k)+FlxU(i,j+1,k)
          enddo
        enddo
        do j=jstrV,jend
          do i=istr,iend+1
            cff=FlxU(i,j,k)+FlxU(i,j-1,k)-0.125*( wrk2(i,j )
     & +wrk2(i,j-1))
            VFx(i,j)=0.25*( cff*(v(i,j,k)+v(i-1,j,k))
     & -gamma*( max(cff,0.)*wrk1(i-1,j)
     & +min(cff,0.)*wrk1(i ,j)
     & ))

            VFx(i,j)=VFx(i,j) - 0.25*( v(i,j,k)+v(i-1,j,k)
     & -0.125*(wrk1(i,j)+wrk1(i-1,j))
     & )*( FlxU(i,j,k)+FlxU(i,j-1,k)
     & -0.125*(wrk2(i,j)+wrk2(i,j-1)))

          enddo
        enddo





        do j=jstr,jend
          do i=istrU,iend
            advu(i,j,k) = (-UFx(i,j)+UFx(i-1,j)
     & -UFe(i,j+1)+UFe(i,j))
          enddo
        enddo

        do j=jstrV,jend
          do i=istr,iend
            advv(i,j,k) = (-VFx(i+1,j)+VFx(i,j)
     & -VFe(i,j)+VFe(i,j-1))
          enddo
        enddo


      enddo !k




! Divide all diagnostic terms by the cell volume Hz/(pm*pn).
! There after the unit of diag terms are :
! (unit of velocity) * s-1 = m * s-2


        do j=jstr,jend
          do i=istrU,iend
            cff=0.5*(pm(i,j)+pm(i-1,j))
     & *(pn(i,j)+pn(i-1,j))


            do k=1,N

              advu(i,j,k)=advu(i,j,k)*cff
     & * u(i,j,k)
     & /(Hz(i,j,k)+Hz(i-1,j,k))
            enddo
           enddo
         enddo


        do j=jstrV,jend
          do i=istr,iend

            cff=0.5*(pm(i,j)+pm(i,j-1))
     & *(pn(i,j)+pn(i,j-1))

            do k=1,N

              advv(i,j,k)=advv(i,j,k)*cff
     & * v(i,j,k)
     & /(Hz(i,j,k)+Hz(i,j-1,k))

            enddo
          enddo
        enddo



!!!!!!!!!!!!!!!!!

        do k=1,N
          do i=istr,iend
            kediss(i,jstr,k)=0.
            kediss(i,jend,k)=0.
          enddo
        enddo


        do k=1,N
          do j=jstr,jend
            kediss(istr,j,k)=0.
            kediss(iend,j,k)=0.
          enddo
        enddo

       write(*,*) istr,jstr,kediss(istr,jstr,1)
       write(*,*) istrU,jstrV,kediss(istr,jstr,1)


        do j=jstrV,jend-1
          do i=istrU,iend-1
            do k=1,N

              kediss(i,j,k) = 0.5*( advu(i,j,k) + advu(i+1,j,k)
     & + advv(i,j,k) + advv(i,j+1,k) )
! # ifdef
! & *rmask(i,j)
! # endif
            enddo
          enddo
        enddo

         write(*,*) istr,jstr,kediss(istr,jstr,1)

!!!!!!!!!!!!!!!!!!


      return
      end
# 175 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_kediss_2d.F" 1



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!compute Advective part of the barotropic vorticity balance equation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      subroutine get_kediss_2d (Lm,Mm,u,v, pm,pn
     & ,kediss)

      implicit none

      integer Lm,Mm, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1), v(0:Lm+1,1:Mm+1),
     & FlxU(1:Lm+1,0:Mm+1), FlxV(0:Lm+1,1:Mm+1),
     & Hz(0:Lm+1,0:Mm+1), gamma,
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & var1, var2,var3, var4, cff, cff1, cff2



      real*8 wrk1(0:Lm+1,0:Mm+1), wrk2(0:Lm+1,0:Mm+1),
     & UFx(0:Lm+1,0:Mm+1), UFe(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1), VFe(0:Lm+1,0:Mm+1)

      real*8 advu(1:Lm+1,0:Mm+1), advv(0:Lm+1,1:Mm+1)

      real*8 kediss(0:Lm+1,0:Mm+1)

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
# 38 "R_tools_fort_routines_gula/get_kediss_2d.F" 2

      parameter (gamma=0.25)

Cf2py intent(in) Lm,Mm,N, u,v,pm,pn
Cf2py intent(out) kediss

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jmin,jmax
        do i=imin,imax
           Hz(i,j) = 1.
        enddo
      enddo



      do j=jmin,jmax
        do i=imin+1,imax
            dn_u(i,j) = 2./(pn(i,j)+pn(i-1,j))
              FlxU(i,j) = 0.5*(Hz(i,j)+Hz(i-1,j))*dn_u(i,j)
     & * u(i,j)
          enddo
      enddo



      do j=jmin+1,jmax
        do i=imin,imax
            dm_v(i,j) = 2./(pm(i,j)+pm(i,j-1))
              FlxV(i,j) = 0.5*(Hz(i,j)+Hz(i,j-1))*dm_v(i,j)
     & * v(i,j)
          enddo
      enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!
! Add in horizontal advection of momentum: Compute diagonal [UFx,VFe]
! and off-diagonal [UFe,VFx] components of tensor of momentum flux
! due to horizontal advection; after that add divergence of these
! terms to r.h.s.
!





        do j=jmin,jmax
         do i=imin+1,imax-2

            wrk1(i,j)=u(i-1,j)-2.*u(i,j)
     & +u(i+1,j)
            wrk2(i,j)=FlxU(i-1,j)-2.*FlxU(i,j)
     & +FlxU(i+1,j)
          enddo
        enddo





        do j=jmin,jmax
         do i=imin+2,imax-1
            cff=FlxU(i,j)+FlxU(i+1,j)-0.125*( wrk2(i ,j)
     & +wrk2(i+1,j))
         UFx(i,j)=0.25*( cff*(u(i,j)+u(i+1,j))
     & -gamma*( max(cff,0.)*wrk1(i ,j)
     & +min(cff,0.)*wrk1(i+1,j)
     & ))

        UFx(i,j) = UFx(i,j) - 0.25*( u(i,j)+u(i+1,j)
     & -0.125*(wrk1(i,j)+wrk1(i+1,j))
     & )*( FlxU(i,j)+FlxU(i+1,j)
     & -0.125*(wrk2(i,j)+wrk2(i+1,j)))

          enddo
        enddo
# 138 "R_tools_fort_routines_gula/get_kediss_2d.F"
        do j=jmin+1,jmax-2
         do i=imin,imax

            wrk1(i,j)=v(i,j-1)-2.*v(i,j)+v(i,j+1)

            wrk2(i,j)=FlxV(i,j-1)-2.*FlxV(i,j)+FlxV(i,j+1)

          enddo
        enddo


        do j=jmin+2,jmax-1
         do i=imin,imax


            cff=FlxV(i,j)+FlxV(i,j+1)-0.125*( wrk2(i,j )
     & +wrk2(i,j+1))

            VFe(i,j)=0.25*( cff*(v(i,j)+v(i,j+1))
     & -gamma*( max(cff,0.)*wrk1(i,j )
     & +min(cff,0.)*wrk1(i,j+1)
     & ))


        VFe(i,j)=VFe(i,j) - 0.25*( v(i,j)+v(i,j+1)
     & -0.125*(wrk1(i,j)+wrk1(i,j+1))
     & )*( FlxV(i,j)+FlxV(i,j+1)
     & -0.125*(wrk2(i,j)+wrk2(i,j+1)))


          enddo
        enddo
# 180 "R_tools_fort_routines_gula/get_kediss_2d.F"
       do j=jmin+1,jmax-1
         do i=imin,imax
            wrk1(i,j)=u(i,j-1)-2.*u(i,j)
     & +u(i,j+1)
          enddo
        enddo

        do j=jmin,jmax
         do i=imin+1,imax-1
           wrk2(i,j)=FlxV(i-1,j)-2.*FlxV(i,j)+FlxV(i+1,j)
          enddo
        enddo

        do j=jmin+1,jmax-1
         do i=imin+1,imax-1

            cff=FlxV(i,j)+FlxV(i-1,j)-0.125*( wrk2(i ,j)
     & +wrk2(i-1,j))
            UFe(i,j)=0.25*( cff*(u(i,j)+u(i,j-1))
     & -gamma*( max(cff,0.)*wrk1(i,j-1)
     & +min(cff,0.)*wrk1(i,j )
     & ))

            UFe(i,j)=UFe(i,j) - 0.25*( u(i,j)+u(i,j-1)
     & -0.125*(wrk1(i,j)+wrk1(i,j-1))
     & )*( FlxV(i,j)+FlxV(i-1,j)
     & -0.125*(wrk2(i,j)+wrk2(i-1,j)))

          enddo
        enddo







        do j=jmin,jmax
         do i=imin+1,imax-1

            wrk1(i,j)=v(i-1,j)-2.*v(i,j)
     & +v(i+1,j)
          enddo
        enddo

        do j=jmin+1,jmax-1
         do i=imin,imax

           wrk2(i,j)=FlxU(i,j-1)-2.*FlxU(i,j)+FlxU(i,j+1)
          enddo
        enddo

        do j=jmin+1,jmax-1
         do i=imin+1,imax-1


            cff=FlxU(i,j)+FlxU(i,j-1)-0.125*( wrk2(i,j )
     & +wrk2(i,j-1))
            VFx(i,j)=0.25*( cff*(v(i,j)+v(i-1,j))
     & -gamma*( max(cff,0.)*wrk1(i-1,j)
     & +min(cff,0.)*wrk1(i ,j)
     & ))

            VFx(i,j)=VFx(i,j) - 0.25*( v(i,j)+v(i-1,j)
     & -0.125*(wrk1(i,j)+wrk1(i-1,j))
     & )*( FlxU(i,j)+FlxU(i,j-1)
     & -0.125*(wrk2(i,j)+wrk2(i,j-1)))

          enddo
        enddo
# 262 "R_tools_fort_routines_gula/get_kediss_2d.F"
      do j=jmin+2,jmax-1
        do i=imin+2,imax-1

            advu(i,j) = (-UFx(i,j)+UFx(i-1,j)
     & -UFe(i,j+1)+UFe(i,j))



          enddo
        enddo

      do j=jmin+2,jmax-1
        do i=imin+2,imax-1


            advv(i,j) = (-VFx(i+1,j)+VFx(i,j)
     & -VFe(i,j)+VFe(i,j-1))


          enddo
        enddo





! Divide all diagnostic terms by (pm*pn).
! There after the unit of these terms are :
! s-2


      do j=jmin+2,jmax-1
        do i=imin+2,imax-1


            cff=0.25*(pm(i,j)+pm(i-1,j))
     & *(pn(i,j)+pn(i-1,j))


              advu(i,j)=advu(i,j)*cff
     & * u(i,j)



            cff=0.25*(pm(i,j)+pm(i,j-1))
     & *(pn(i,j)+pn(i,j-1))



              advv(i,j)=advv(i,j)*cff
     & * v(i,j)


          enddo
        enddo



!!!!!!!!!!!!!!!!!


      do j=jmin+2,jmax-1
        do i=imin+2,imax-1


              kediss(i,j) = 0.5*( advu(i-1,j) + advu(i,j)
     & + advv(i,j-1) + advv(i,j) )


          enddo
        enddo


!!!!!!!!!!!!!!!!!!

      return
      end
# 177 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/visc3d_GP.F" 1
# 9 "R_tools_fort_routines_gula/visc3d_GP.F"
      subroutine visc3d_GP (Lm,Mm,N,u,v,z_r,z_w
     & ,pm,pn,f,dt,rmask,umask, vmask
     & ,visc2, v_sponge,coord,coordmax
     & ,MHmix)

!
! Compute horizontal (along geopotential surfaces) viscous terms as
! divergence of symmetric stress tensor.
!
! Compute harmonic mixing of momentum, rotated along geopotentials,
! from the horizontal divergence of the stress tensor.
! A transverse isotropy is assumed so the stress tensor is splitted
! into vertical and horizontal subtensors.
!
! Reference:
!
! [1] Stelling, G. S., and J. A. Th. M. van Kester, 1994: On the
! approximation of horizontal gradients in sigma-coordinates
! for bathymetry with steep bottom slopes. Int. J. Num. Meth.
! in Fluids, v. 18, pp. 915-935.
!
! [2] Wajsowicz, R.C, 1993: A consistent formulation of the
! anisotropic stress tensor for use in models of the
! large-scale ocean circulation, JCP, 105, 333-338.
!
! [3] Sadourny, R. and K. Maynard, 1997: Formulations of lateral
! diffusion in geophysical fluid dynamics models, In
! "Numerical Methods of Atmospheric and Oceanic Modelling".
! Lin, Laprise, and Ritchie, Eds., NRC Research Press,
! 547-556.
!
! [4] Griffies, S.M. and R.W. Hallberg, 2000: Biharmonic friction
! with a Smagorinsky-like viscosity for use in large-scale
! eddy-permitting ocean models, Monthly Weather Rev.,v. 128,
! No. 8, pp. 2935-2946.
!
      implicit none



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      integer Lm,Mm,LLm,MMm,N, imin,imax,jmin,jmax, i,j,k,
     & dt,
     & itrc, k1,k2,isp,ibnd

        !INPUT
      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N)
     & ,z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N)
     & ,pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1)
     & ,rmask(0:Lm+1,0:Mm+1), pmask(1:Lm+1,1:Mm+1)
     & ,umask(1:Lm+1,0:Mm+1), vmask(0:Lm+1,1:Mm+1)
     & ,f(0:Lm+1,0:Mm+1)

      integer coord(4), coordmax(4)

      ! OUTPUTS
      real*8 MHmix(0:Lm+1,0:Mm+1,N,2)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      real visc2, v_sponge

      real UFe(0:Lm+1,0:Mm+1),
     & VFe(0:Lm+1,0:Mm+1), cff,
     & UFx(0:Lm+1,0:Mm+1), cff1,
     & VFx(0:Lm+1,0:Mm+1), cff2,
     & UFs(0:Lm+1,0:Mm+1,2), cff3,
     & VFs(0:Lm+1,0:Mm+1,2), cff4,
     & dmUde(0:Lm+1,0:Mm+1,2), cff5,
     & dmVde(0:Lm+1,0:Mm+1,2), cff6,
     & dnUdx(0:Lm+1,0:Mm+1,2), cff7,
     & dnVdx(0:Lm+1,0:Mm+1,2), cff8,
     & dUdz(0:Lm+1,0:Mm+1,2),
     & dVdz(0:Lm+1,0:Mm+1,2), dmUdz,
     & dZde_p(0:Lm+1,0:Mm+1,2), dnUdz,
     & dZde_r(0:Lm+1,0:Mm+1,2), dmVdz,
     & dZdx_p(0:Lm+1,0:Mm+1,2), dnVdz,
     & dZdx_r(0:Lm+1,0:Mm+1,2),
     & dm_p(0:Lm+1,0:Mm+1), dn_p(0:Lm+1,0:Mm+1),
     & dm_r(0:Lm+1,0:Mm+1), dn_r(0:Lm+1,0:Mm+1),
     & wrk(0:Lm+1,0:Mm+1)


      real*8 Hz(0:Lm+1,0:Mm+1,N),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & dm_u(0:Lm+1,0:Mm+1), dn_v(0:Lm+1,0:Mm+1),
     & visc2_r(0:Lm+1,0:Mm+1), visc2_p(0:Lm+1,0:Mm+1)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
# 102 "R_tools_fort_routines_gula/visc3d_GP.F" 2



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,u,v,z_r,z_w,pm,pn,f,dt,rmask,umask, vmask,visc2, v_sponge,coord,coordmax
Cf2py intent(out) MHmix


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! define visc coefficient (from set_nudgcof.F )
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      LLm = coordmax(4)-2
      MMm = coordmax(2)-2


      !isp=min((LLm+1)/12,(MMm+1)/12)
       isp=(LLm+1)/12 !old version of the code



       do j=max(-1,jstr-1),jend
         do i=max(-1,istr-1),iend
          ibnd=isp


          ibnd=min(ibnd,i+coord(3))


c ibnd=min(ibnd,LLm+1-i-coord(3))


          ibnd=min(ibnd,j+coord(1))


          ibnd=min(ibnd,MMm+1-j-coord(1))


          wrk(i,j)=float(isp-ibnd)/float(isp)
        enddo
       enddo


       do j=jstr-1,jend+1
        do i=istr-1,iend+1


          visc2_r(i,j)=visc2 + v_sponge*wrk(i,j)


        enddo
       enddo

       do j=jstr,jend+1
        do i=istr,iend+1


          visc2_p(i,j)=visc2 + 0.25*v_sponge*( wrk(i,j)
     & +wrk(i-1,j)+wrk(i,j-1)+wrk(i-1,j-1))


        enddo
       enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! define grid variables
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jstrR,jendR
        do i=istrR,iendR
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo
        enddo
      enddo


      do j=jstrR,jendR
        do i=istr,iendR
            dm_u(i,j) = 2./(pm(i,j)+pm(i-1,j))
            dn_u(i,j) = 2./(pn(i,j)+pn(i-1,j))
          enddo
        enddo

      do j=jstr,jendR
        do i=istrR,iendR
            dn_v(i,j) = 2./(pn(i,j)+pn(i,j-1))
            dm_v(i,j) = 2./(pm(i,j)+pm(i,j-1))
          enddo
        enddo


      do j=jstrR,jendR
        do i=istrR,iendR
              dm_r(i,j)=1./pm(i,j)
              dn_r(i,j)=1./pn(i,j)
          enddo
        enddo



! Compute n/m and m/n at horizontal PSI-points.
! Set mask according to slipperness parameter gamma.
!
      do j=jstrV-1,jend
        do i=istrU-1,iend


          dm_p(i,j)=4./(pm(i,j)+pm(i,j-1)+pm(i-1,j)+pm(i-1,j-1))
          dn_p(i,j)=4./(pn(i,j)+pn(i,j-1)+pn(i-1,j)+pn(i-1,j-1))


c pmask(i,j)=rmask(i,j)*rmask(i-1,j)*rmask(i,j-1)
c & *rmask(i-1,j-1)
C*** if (gamma2.lt.0.) pmask(i,j)=2.-pmask(i,j)
!
! Set no-slip boundary conditions on land-mask boundaries
! regardless of supplied value of gamma2.
!

          cff1=1. !<-- computation of off-diagonal nonlinear terms
          cff2=2.

          if (rmask(i-1,j ).gt.0.5 .and. rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and. rmask(i,j-1).gt.0.5) then
            pmask(i,j)=1.

          elseif(rmask(i-1,j ).lt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff1
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).lt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff1
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).lt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff1
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).lt.0.5) then
            pmask(i,j)=cff1


          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).lt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).lt.0.5) then
            pmask(i,j)=cff2
          elseif(rmask(i-1,j ).lt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).lt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff2
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).lt.0.5 .and.rmask(i,j-1).lt.0.5) then
            pmask(i,j)=cff2
          elseif(rmask(i-1,j ).lt.0.5 .and.rmask(i,j ).lt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff2

          else
            pmask(i,j)=0.
          endif

        enddo
      enddo





!
! Compute horizontal harmonic viscosity along geopotential surfaces.
!--------------------------------------------------------------------
!
! Compute horizontal and vertical gradients. Notice the recursive
! blocking sequence. The vertical placement of the gradients is:
!
! dZdx_r, dZde_r, dnUdx, dmVde(:,:,k1) k rho-points
! dZdx_r, dZde_r, dnUdx, dmVde(:,:,k2) k+1 rho-points
! dZdx_p, dZde_p, dnVdx, dmUde(:,:,k1) k psi-points
! dZdx_p, dZde_p, dnVdx, dmUde(:,:,k2) k+1 psi-points
! UFs, dUdz(:,:,k1) k-1/2 WU-points
! UFs, dUdz(:,:,k2) k+1/2 WU-points
! VFs, dVdz(:,:,k1) k-1/2 WV-points
! VFs, dVdz(:,:,k2) k+1/2 WV-points
!
! Compute sigma-slopes (nondimensional) at RHO- and PSI-points.
! Compute momentum horizontal (1/m/s) and vertical (1/s) gradients.
!
      k2=1
      do k=0,N,+1 !--> irreversible
        k1=k2
        k2=3-k1
        if (k.lt.N) then
          do j=jstr-1,jend+1
            do i=istrU-1,iend+1
              UFx(i,j)=0.5*(z_r(i,j,k+1)-z_r(i-1,j,k+1))
     & *(pm(i-1,j)+pm(i,j))

     & *umask(i,j)

            enddo
          enddo
          do j=jstrV-1,jend+1
            do i=istr-1,iend+1
              VFe(i,j)=0.5*(z_r(i,j,k+1)-z_r(i,j-1,k+1))
     & *(pn(i,j-1)+pn(i,j))

     & *vmask(i,j)

            enddo
          enddo
          do j=jstrV-1,jend
            do i=istrU-1,iend
              dnUdx(i,j,k2)=0.5*pm(i,j)*(
     & (pn(i ,j)+pn(i+1,j))*u(i+1,j,k+1)
     & -(pn(i-1,j)+pn(i ,j))*u(i ,j,k+1))

     & *rmask(i,j)

              dmVde(i,j,k2)=0.5*pn(i,j)*(
     & (pm(i,j )+pm(i,j+1))*v(i,j+1,k+1)
     & -(pm(i,j-1)+pm(i,j ))*v(i,j ,k+1))

     & *rmask(i,j)

              dZdx_r(i,j,k2)=0.5*(UFx(i,j)+UFx(i+1,j))
              dZde_r(i,j,k2)=0.5*(VFe(i,j)+VFe(i,j+1))
            enddo
          enddo
          do j=jstr,jend+1
            do i=istr,iend+1
              dmUde(i,j,k2)=0.125*(pn(i,j)+pn(i-1,j)+pn(i,j-1)
     & +pn(i-1,j-1))
     & *( (pm(i-1,j )+pm(i,j ))*u(i,j ,k+1)
     & -(pm(i-1,j-1)+pm(i,j-1))*u(i,j-1,k+1))

     & *pmask(i,j)

              dnVdx(i,j,k2)=0.125*(pm(i,j)+pm(i-1,j)+pm(i,j-1)
     & +pm(i-1,j-1))
     & *( (pn(i ,j-1)+pn(i ,j))*v(i ,j,k+1)
     & -(pn(i-1,j-1)+pn(i-1,j))*v(i-1,j,k+1))

     & *pmask(i,j)

              dZde_p(i,j,k2)=0.5*(VFe(i-1,j)+VFe(i,j))
              dZdx_p(i,j,k2)=0.5*(UFx(i,j-1)+UFx(i,j))
            enddo
          enddo !--> discard UFx,VFe, keep all others
        endif


c do j=jstrV-1,jend+1 ! This
c do i=istrU-1,iend+1
c dZdx_r(i,j,k2)=0.
c dZde_r(i,j,k2)=0.
c dZdx_p(i,j,k2)=0.
c dZde_p(i,j,k2)=0.
c enddo
c enddo



        if (k.eq.0 .or. k.eq.N) then
          do j=jstr-1,jend+1
            do i=istrU-1,iend+1
              dUdz(i,j,k2)=0.
              UFs(i,j,k2)=0.
            enddo
          enddo
          do j=jstrV-1,jend+1
            do i=istr-1,iend+1
              dVdz(i,j,k2)=0.
              VFs(i,j,k2)=0.
            enddo
          enddo
        else
          do j=jstr-1,jend+1
            do i=istrU-1,iend+1
              dUdz(i,j,k2)=2.*(u(i,j,k+1)-u(i,j,k))
     & /( z_r(i-1,j,k+1)-z_r(i-1,j,k)
     & +z_r(i ,j,k+1)-z_r(i ,j,k))
            enddo
          enddo
          do j=jstrV-1,jend+1
            do i=istr-1,iend+1
              dVdz(i,j,k2)=2.*(v(i,j,k+1)-v(i,j,k))
     & /( z_r(i,j-1,k+1)-z_r(i,j-1,k)
     & +z_r(i,j ,k+1)-z_r(i,j ,k))
            enddo
          enddo
        endif
!
! Compute components of the rotated viscous flux [m5/s2] along
! geopotential surfaces in the XI- and ETA-directions.
!
        if (k.gt.0) then
          do j=jstrV-1,jend
            do i=istrU-1,iend
              cff=visc2_r(i,j)*Hz(i,j,k)*(
     & dn_r(i,j)*( dnUdx(i,j,k1) - 0.5*pn(i,j)*(
     & min(dZdx_r(i,j,k1),0.)*(dUdz(i,j,k1)+dUdz(i+1,j,k2))
     & +max(dZdx_r(i,j,k1),0.)*(dUdz(i,j,k2)+dUdz(i+1,j,k1))
     & ))
     & -dm_r(i,j)*( dmVde(i,j,k1) - 0.5*pm(i,j)*(
     & min(dZde_r(i,j,k1),0.)*(dVdz(i,j,k1)+dVdz(i,j+1,k2))
     & +max(dZde_r(i,j,k1),0.)*(dVdz(i,j,k2)+dVdz(i,j+1,k1))
     & )))

     & *rmask(i,j)

              UFx(i,j)=dn_r(i,j)*dn_r(i,j)*cff
              VFe(i,j)=dm_r(i,j)*dm_r(i,j)*cff
            enddo
          enddo
          do j=jstr,jend+1
            do i=istr,iend+1
              cff=visc2_p(i,j)*0.25*( Hz(i,j,k) +Hz(i-1,j,k)
     & +Hz(i,j-1,k) +Hz(i-1,j-1,k))
     & *( dn_p(i,j)*( dnVdx(i,j,k1)-0.125*( pn(i,j)+pn(i-1,j)
     & +pn(i,j-1)+pn(i-1,j-1))
     & *( min(dZdx_p(i,j,k1),0.)*(dVdz(i-1,j,k1)+dVdz(i,j,k2))
     & +max(dZdx_p(i,j,k1),0.)*(dVdz(i-1,j,k2)+dVdz(i,j,k1))

     & )) + dm_p(i,j)*( dmUde(i,j,k1)-0.125*( pm(i,j)+pm(i-1,j)
     & +pm(i,j-1)+pm(i-1,j-1))
     & *( min(dZde_p(i,j,k1),0.)*(dUdz(i,j-1,k1)+dUdz(i,j,k2))
     & +max(dZde_p(i,j,k1),0.)*(dUdz(i,j-1,k2)+dUdz(i,j,k1))
     & )))

     & *pmask(i,j)

              UFe(i,j)=dm_p(i,j)*dm_p(i,j)*cff
              VFx(i,j)=dn_p(i,j)*dn_p(i,j)*cff
            enddo
          enddo
!



! Compute vertical flux [m^2/s^2] due to sloping terrain-following
! surfaces.
!
          if (k.lt.N) then
            do j=jstr,jend
              do i=istrU,iend
                cff1=0.5*(pn(i-1,j)+pn(i,j))
                cff2=0.5*(pm(i-1,j)+pm(i,j))
                cff=0.25*( dVdz(i,j,k2)+dVdz(i-1,j,k2)
     & +dVdz(i,j+1,k2)+dVdz(i-1,j+1,k2))
                dnUdz=cff1*dUdz(i,j,k2)
                dmUdz=cff2*dUdz(i,j,k2)
                dnVdz=cff1*cff
                dmVdz=cff2*cff

                cff1=min(dZdx_r(i-1,j,k1),0.)
                cff2=min(dZdx_r(i ,j,k2),0.)
                cff3=max(dZdx_r(i-1,j,k2),0.)
                cff4=max(dZdx_r(i ,j,k1),0.)
                cff5=min(dZde_r(i-1,j,k1),0.)
                cff6=min(dZde_r(i ,j,k2),0.)
                cff7=max(dZde_r(i-1,j,k2),0.)
                cff8=max(dZde_r(i ,j,k1),0.)

                cff=dn_u(i,j)*( cff1*(cff1*dnUdz-dnUdx(i-1,j,k1))
     & +cff2*(cff2*dnUdz-dnUdx(i ,j,k2))
     & +cff3*(cff3*dnUdz-dnUdx(i-1,j,k2))
     & +cff4*(cff4*dnUdz-dnUdx(i ,j,k1))
     & )
     & -dm_u(i,j)*( cff1*(cff5*dmVdz-dmVde(i-1,j,k1))
     & +cff2*(cff6*dmVdz-dmVde(i ,j,k2))
     & +cff3*(cff7*dmVdz-dmVde(i-1,j,k2))
     & +cff4*(cff8*dmVdz-dmVde(i ,j,k1))
     & )
                cff1=min(dZde_p(i,j ,k1),0.)
                cff2=min(dZde_p(i,j+1,k2),0.)
                cff3=max(dZde_p(i,j ,k2),0.)
                cff4=max(dZde_p(i,j+1,k1),0.)
                cff5=min(dZdx_p(i,j ,k1),0.)
                cff6=min(dZdx_p(i,j+1,k2),0.)
                cff7=max(dZdx_p(i,j ,k2),0.)
                cff8=max(dZdx_p(i,j+1,k1),0.)

                cff=cff + dm_u(i,j)*(
     & cff1*(cff1*dmUdz-dmUde(i,j ,k1))
     & +cff2*(cff2*dmUdz-dmUde(i,j+1,k2))
     & +cff3*(cff3*dmUdz-dmUde(i,j ,k2))
     & +cff4*(cff4*dmUdz-dmUde(i,j+1,k1))
     & )
     & +dn_u(i,j)*( cff1*(cff5*dnVdz-dnVdx(i,j ,k1))
     & +cff2*(cff6*dnVdz-dnVdx(i,j+1,k2))
     & +cff3*(cff7*dnVdz-dnVdx(i,j ,k2))
     & +cff4*(cff8*dnVdz-dnVdx(i,j+1,k1))
     & )

                UFs(i,j,k2)=0.25*(visc2_r(i-1,j)+visc2_r(i,j))*cff
              enddo
            enddo

            do j=jstrV,jend
              do i=istr,iend
                cff1=0.5*(pn(i,j-1)+pn(i,j))
                cff2=0.5*(pm(i,j-1)+pm(i,j))
                cff=0.25*( dUdz(i,j,k2)+dUdz(i+1,j,k2)
     & +dUdz(i,j-1,k2)+dUdz(i+1,j-1,k2))
                dnUdz=cff1*cff
                dmUdz=cff2*cff
                dnVdz=cff1*dVdz(i,j,k2)
                dmVdz=cff2*dVdz(i,j,k2)

                cff1=min(dZdx_p(i ,j,k1),0.)
                cff2=min(dZdx_p(i+1,j,k2),0.)
                cff3=max(dZdx_p(i ,j,k2),0.)
                cff4=max(dZdx_p(i+1,j,k1),0.)
                cff5=min(dZde_p(i ,j,k1),0.)
                cff6=min(dZde_p(i+1,j,k2),0.)
                cff7=max(dZde_p(i ,j,k2),0.)
                cff8=max(dZde_p(i+1,j,k1),0.)

                cff=dn_v(i,j)*( cff1*(cff1*dnVdz-dnVdx(i ,j,k1))
     & +cff2*(cff2*dnVdz-dnVdx(i+1,j,k2))
     & +cff3*(cff3*dnVdz-dnVdx(i ,j,k2))
     & +cff4*(cff4*dnVdz-dnVdx(i+1,j,k1))
     & )
     & +dm_v(i,j)*( cff1*(cff5*dmUdz-dmUde(i ,j,k1))
     & +cff2*(cff6*dmUdz-dmUde(i+1,j,k2))
     & +cff3*(cff7*dmUdz-dmUde(i ,j,k2))
     & +cff4*(cff8*dmUdz-dmUde(i+1,j,k1))
     & )
                cff1=min(dZde_r(i,j-1,k1),0.)
                cff2=min(dZde_r(i,j ,k2),0.)
                cff3=max(dZde_r(i,j-1,k2),0.)
                cff4=max(dZde_r(i,j ,k1),0.)
                cff5=min(dZdx_r(i,j-1,k1),0.)
                cff6=min(dZdx_r(i,j ,k2),0.)
                cff7=max(dZdx_r(i,j-1,k2),0.)
                cff8=max(dZdx_r(i,j ,k1),0.)

                cff=cff+dm_v(i,j)*(
     & cff1*(cff1*dmVdz-dmVde(i,j-1,k1))
     & +cff2*(cff2*dmVdz-dmVde(i,j ,k2))
     & +cff3*(cff3*dmVdz-dmVde(i,j-1,k2))
     & +cff4*(cff4*dmVdz-dmVde(i,j ,k1))
     & )
     & -dn_v(i,j)*( cff1*(cff5*dnUdz-dnUdx(i,j-1,k1))
     & +cff2*(cff6*dnUdz-dnUdx(i,j ,k2))
     & +cff3*(cff7*dnUdz-dnUdx(i,j-1,k2))
     & +cff4*(cff8*dnUdz-dnUdx(i,j ,k1))
     & )

                VFs(i,j,k2)=0.25*(visc2_r(i,j-1)+visc2_r(i,j))*cff
              enddo
            enddo
          endif
!
! Apply viscous terms. Note that at this stage arrays u,v(...,3-nstp)
! contain Hz*U and Hz*V with units of [m2/s]. Also compute vertical
! integral of viscous terms and add it into coupling terms for the
! barotropic mode
!

          do j=jstr,jend
            do i=istrU,iend
              cff=0.125*(pm(i-1,j)+pm(i,j))*(pn(i-1,j) +pn(i,j))
     & *( (pn(i-1,j)+pn(i,j))*(UFx(i,j)-UFx(i-1,j))
     & +(pm(i-1,j)+pm(i,j))*(UFe(i,j+1)-UFe(i,j))
     & )

              MHmix(i,j,k,1) = (cff+UFs(i,j,k2)-UFs(i,j,k1))
     & *dm_u(i,j)*dn_u(i,j)







            enddo
          enddo
          do j=jstrV,jend
            do i=istr,iend
              cff=0.125*(pm(i,j)+pm(i,j-1))*(pn(i,j) +pn(i,j-1))
     & *( (pn(i,j-1)+pn(i,j))*(VFx(i+1,j)-VFx(i,j))
     & -(pm(i,j-1)+pm(i,j))*(VFe(i,j)-VFe(i,j-1))
     & )


              MHmix(i,j,k,2) = (cff+VFs(i,j,k2)-VFs(i,j,k1))
     & *dm_v(i,j)*dn_v(i,j)




            enddo
          enddo
        endif
      enddo
      return
      end
# 179 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/visc3d_S.F" 1







      subroutine visc3d_S (Lm,Mm,N,u,v,z_r,z_w
     & ,pm,pn,f,dt,rmask,umask, vmask
     & ,visc2, v_sponge,coord,coordmax
     & ,MHmix)

!
! Compute horizontal (along geopotential surfaces) viscous terms as
! divergence of symmetric stress tensor.
!
! Compute harmonic mixing of momentum, rotated along geopotentials,
! from the horizontal divergence of the stress tensor.
! A transverse isotropy is assumed so the stress tensor is splitted
! into vertical and horizontal subtensors.
!
! Reference:
!
! [1] Stelling, G. S., and J. A. Th. M. van Kester, 1994: On the
! approximation of horizontal gradients in sigma-coordinates
! for bathymetry with steep bottom slopes. Int. J. Num. Meth.
! in Fluids, v. 18, pp. 915-935.
!
! [2] Wajsowicz, R.C, 1993: A consistent formulation of the
! anisotropic stress tensor for use in models of the
! large-scale ocean circulation, JCP, 105, 333-338.
!
! [3] Sadourny, R. and K. Maynard, 1997: Formulations of lateral
! diffusion in geophysical fluid dynamics models, In
! "Numerical Methods of Atmospheric and Oceanic Modelling".
! Lin, Laprise, and Ritchie, Eds., NRC Research Press,
! 547-556.
!
! [4] Griffies, S.M. and R.W. Hallberg, 2000: Biharmonic friction
! with a Smagorinsky-like viscosity for use in large-scale
! eddy-permitting ocean models, Monthly Weather Rev.,v. 128,
! No. 8, pp. 2935-2946.
!
      implicit none



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      integer Lm,Mm,LLm,MMm,N, imin,imax,jmin,jmax, i,j,k,
     & dt,
     & itrc, k1,k2,isp,ibnd

        !INPUT
      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N)
     & ,z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N)
     & ,pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1)
     & ,rmask(0:Lm+1,0:Mm+1), pmask(1:Lm+1,1:Mm+1)
     & ,umask(1:Lm+1,0:Mm+1), vmask(0:Lm+1,1:Mm+1)
     & ,f(0:Lm+1,0:Mm+1)

      integer coord(4), coordmax(4)

      ! OUTPUTS
      real*8 MHmix(0:Lm+1,0:Mm+1,N,2)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      real visc2, v_sponge

      real UFe(0:Lm+1,0:Mm+1),
     & VFe(0:Lm+1,0:Mm+1), cff,cff1,cff2,
     & UFx(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1),
     & dm_p(0:Lm+1,0:Mm+1), dn_p(0:Lm+1,0:Mm+1),
     & dm_r(0:Lm+1,0:Mm+1), dn_r(0:Lm+1,0:Mm+1),
     & wrk(0:Lm+1,0:Mm+1)


      real*8 Hz(0:Lm+1,0:Mm+1,N),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & dm_u(0:Lm+1,0:Mm+1), dn_v(0:Lm+1,0:Mm+1),
     & visc2_r(0:Lm+1,0:Mm+1), visc2_p(0:Lm+1,0:Mm+1)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
# 89 "R_tools_fort_routines_gula/visc3d_S.F" 2



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,u,v,z_r,z_w,pm,pn,f,dt,rmask,umask, vmask,visc2, v_sponge,coord,coordmax
Cf2py intent(out) MHmix


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! define visc coefficient (from set_nudgcof.F )
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      LLm = coordmax(4)-2
      MMm = coordmax(2)-2


       isp=min((LLm+1)/12,(MMm+1)/12)
      ! isp=(LLm+1)/12 !old version of the code


       do j=max(-1,jstr-1),jend
         do i=max(-1,istr-1),iend
          ibnd=isp


          ibnd=min(ibnd,i+coord(3))


          ibnd=min(ibnd,LLm+1-i-coord(3))


          ibnd=min(ibnd,j+coord(1))


          ibnd=min(ibnd,MMm+1-j-coord(1))


          wrk(i,j)=float(isp-ibnd)/float(isp)
        enddo
       enddo


       do j=jstr-1,jend+1
        do i=istr-1,iend+1


          visc2_r(i,j)=visc2 + v_sponge*wrk(i,j)


        enddo
       enddo

       do j=jstr,jend+1
        do i=istr,iend+1


          visc2_p(i,j)=visc2 + 0.25*v_sponge*( wrk(i,j)
     & +wrk(i-1,j)+wrk(i,j-1)+wrk(i-1,j-1))


        enddo
       enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! define grid variables
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jstrR,jendR
        do i=istrR,iendR
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo
              dm_r(i,j)=1./pm(i,j)
              dn_r(i,j)=1./pn(i,j)
        enddo
      enddo

      do j=jstrR,jendR
        do i=istr,iendR
            dn_u(i,j) = 2./(pn(i,j)+pn(i-1,j))
            dm_u(i,j) = 2./(pm(i,j)+pm(i-1,j))
          enddo
      enddo



      do j=jstr,jendR
        do i=istrR,iendR
            dm_v(i,j) = 2./(pm(i,j)+pm(i,j-1))
            dn_v(i,j) = 2./(pn(i,j)+pn(i,j-1))
          enddo
      enddo



! Compute n/m and m/n at horizontal PSI-points.
! Set mask according to slipperness parameter gamma.
!
      do j=jstrV-1,jend
        do i=istrU-1,iend


          dm_p(i,j)=4./(pm(i,j)+pm(i,j-1)+pm(i-1,j)+pm(i-1,j-1))
          dn_p(i,j)=4./(pn(i,j)+pn(i,j-1)+pn(i-1,j)+pn(i-1,j-1))


c pmask(i,j)=rmask(i,j)*rmask(i-1,j)*rmask(i,j-1)
c & *rmask(i-1,j-1)
C*** if (gamma2.lt.0.) pmask(i,j)=2.-pmask(i,j)
!
! Set no-slip boundary conditions on land-mask boundaries
! regardless of supplied value of gamma2.
!

          cff1=1. !<-- computation of off-diagonal nonlinear terms
          cff2=2.

          if (rmask(i-1,j ).gt.0.5 .and. rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and. rmask(i,j-1).gt.0.5) then
            pmask(i,j)=1.

          elseif(rmask(i-1,j ).lt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff1
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).lt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff1
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).lt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff1
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).lt.0.5) then
            pmask(i,j)=cff1


          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).lt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).lt.0.5) then
            pmask(i,j)=cff2
          elseif(rmask(i-1,j ).lt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).lt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff2
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).lt.0.5 .and.rmask(i,j-1).lt.0.5) then
            pmask(i,j)=cff2
          elseif(rmask(i-1,j ).lt.0.5 .and.rmask(i,j ).lt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff2

          else
            pmask(i,j)=0.
          endif


        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


! Compute horizontal harmonic viscosity along constant S-surfaces.
!
! Compute flux-components of the horizontal divergence of the stress
! tensor (m5/s2) in XI- and ETA-directions.
!
      do k=1,N


        do j=jstrV-1,jend
          do i=istrU-1,iend
            cff=0.5*Hz(i,j,k)*visc2_r(i,j)*(

     & dn_r(i,j)*pm(i,j)*( (pn(i ,j)+pn(i+1,j))*u(i+1,j,k)
     & -(pn(i-1,j)+pn(i ,j))*u(i ,j,k)
     & )

     & -dm_r(i,j)*pn(i,j)*( (pm(i,j )+pm(i,j+1))*v(i,j+1,k)
     & -(pm(i,j-1)+pm(i,j ))*v(i,j ,k)
     & ))
            UFx(i,j)= cff*dn_r(i,j)*dn_r(i,j)
            VFe(i,j)= -cff*dm_r(i,j)*dm_r(i,j)

          enddo
        enddo



        do j=jstr,jend+1
          do i=istr,iend+1
            cff=0.125*(Hz(i-1,j,k)+Hz(i,j,k)+Hz(i-1,j-1,k)
     & +Hz(i,j-1,k))*visc2_p(i,j)*(

     & 0.25*(pm(i-1,j)+pm(i,j)+pm(i-1,j-1)+pm(i,j-1))*dn_p(i,j)
     & *( (pn(i ,j-1)+pn(i ,j))*v(i ,j,k)
     & -(pn(i-1,j-1)+pn(i-1,j))*v(i-1,j,k)
     & )

     & +0.25*(pn(i-1,j)+pn(i,j)+pn(i-1,j-1)+pn(i,j-1))*dm_p(i,j)
     & *( (pm(i-1,j )+pm(i,j ))*u(i,j ,k)
     & -(pm(i-1,j-1)+pm(i,j-1))*u(i,j-1,k)
     & ))

     & *pmask(i,j)

            UFe(i,j)= cff*dm_p(i,j)*dm_p(i,j)
            VFx(i,j)= cff*dn_p(i,j)*dn_p(i,j)
          enddo
        enddo




!
!
! if (k.eq.1) then
! write(*,*) 'u(1:5,82,0,nstp)',u(1:5,82,1)
! write(*,*) 'v(1:5,82,0,nstp)',v(1:5,82,1)
! write(*,*) 'u(1:5,81,0,nstp)',u(1:5,81,1)
! write(*,*) 'v(1:5,81,0,nstp)',v(1:5,81,1)
! write(*,*) 'UFx(i,j)',UFx(1:5,82)
! write(*,*) 'VFe(i,j)',VFe(1:5,82)
! write(*,*) 'UFe(i,j)',UFe(1:5,82)
! write(*,*) 'VFx(i,j)',VFx(1:5,82)
! write(*,*) 'dm_p(i,j)*dm_p(i,j)',dm_p(5,82),dn_p(5,82)
! write(*,*) 'pmask(i,j)',pmask(5,82)
! write(*,*) 'pm',pm(4:5,81:82)
! write(*,*) 'pn',pn(4:5,81:82)
! write(*,*) 'visc2_p',visc2_p(5,82)
! write(*,*) 'Hz',Hz(4,82,1)+Hz(5,82,1)+Hz(4,81,1)+Hz(5,81,1)
! endif
!
!



!
! Apply viscous terms. Note that at this stage arrays u,v(...,3-nstp)
! contain Hz*U and Hz*V with units of [m2/s]. Also compute vertical
! integral of viscous terms and add it into coupling terms for the
! barotropic mode
!

          do j=jstr,jend
            do i=istrU,iend
              cff=0.125*(pm(i-1,j)+pm(i,j))*(pn(i-1,j) +pn(i,j))
     & *( (pn(i-1,j)+pn(i,j))*(UFx(i,j)-UFx(i-1,j))
     & +(pm(i-1,j)+pm(i,j))*(UFe(i,j+1)-UFe(i,j))
     & )

              MHmix(i,j,k,1) = cff *dm_u(i,j)*dn_u(i,j)

            enddo
          enddo


        do j=jstrV,jend
          do i=istr,iend
            cff=0.125*(pm(i,j)+pm(i,j-1))*(pn(i,j) +pn(i,j-1))
     & *( (pn(i,j-1)+pn(i,j))*(VFx(i+1,j)-VFx(i,j))
     & +(pm(i,j-1)+pm(i,j))*(VFe(i,j)-VFe(i,j-1))
     & )


              MHmix(i,j,k,2) = cff*dm_v(i,j)*dn_v(i,j)



            enddo
          enddo


      enddo
      return
      end
# 181 "R_tools_fort_gula.F" 2
# 1 "R_tools_fort_routines_gula/visc3d_S_baham.F" 1







      subroutine visc3d_S_baham (Lm,Mm,N,u,v,z_r,z_w
     & ,pm,pn,f,dt,rmask,umask, vmask
     & ,visc2, v_sponge,coord,coordmax
     & ,MHmix)

!
! Compute horizontal (along geopotential surfaces) viscous terms as
! divergence of symmetric stress tensor.
!
! Compute harmonic mixing of momentum, rotated along geopotentials,
! from the horizontal divergence of the stress tensor.
! A transverse isotropy is assumed so the stress tensor is splitted
! into vertical and horizontal subtensors.
!
! Reference:
!
! [1] Stelling, G. S., and J. A. Th. M. van Kester, 1994: On the
! approximation of horizontal gradients in sigma-coordinates
! for bathymetry with steep bottom slopes. Int. J. Num. Meth.
! in Fluids, v. 18, pp. 915-935.
!
! [2] Wajsowicz, R.C, 1993: A consistent formulation of the
! anisotropic stress tensor for use in models of the
! large-scale ocean circulation, JCP, 105, 333-338.
!
! [3] Sadourny, R. and K. Maynard, 1997: Formulations of lateral
! diffusion in geophysical fluid dynamics models, In
! "Numerical Methods of Atmospheric and Oceanic Modelling".
! Lin, Laprise, and Ritchie, Eds., NRC Research Press,
! 547-556.
!
! [4] Griffies, S.M. and R.W. Hallberg, 2000: Biharmonic friction
! with a Smagorinsky-like viscosity for use in large-scale
! eddy-permitting ocean models, Monthly Weather Rev.,v. 128,
! No. 8, pp. 2935-2946.
!
      implicit none



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      integer Lm,Mm,LLm,MMm,N, imin,imax,jmin,jmax, i,j,k,
     & dt,
     & itrc, k1,k2,isp,ibnd

        !INPUT
      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N)
     & ,z_r(0:Lm+1,0:Mm+1,N), z_w(0:Lm+1,0:Mm+1,0:N)
     & ,pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1)
     & ,rmask(0:Lm+1,0:Mm+1), pmask(1:Lm+1,1:Mm+1)
     & ,umask(1:Lm+1,0:Mm+1), vmask(0:Lm+1,1:Mm+1)
     & ,f(0:Lm+1,0:Mm+1)

      integer coord(4), coordmax(4)

      ! OUTPUTS
      real*8 MHmix(0:Lm+1,0:Mm+1,N,2)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      real visc2, v_sponge

      real UFe(0:Lm+1,0:Mm+1),
     & VFe(0:Lm+1,0:Mm+1), cff,cff1,cff2,
     & UFx(0:Lm+1,0:Mm+1),
     & VFx(0:Lm+1,0:Mm+1),
     & dm_p(0:Lm+1,0:Mm+1), dn_p(0:Lm+1,0:Mm+1),
     & dm_r(0:Lm+1,0:Mm+1), dn_r(0:Lm+1,0:Mm+1),
     & wrk(0:Lm+1,0:Mm+1)


      real*8 Hz(0:Lm+1,0:Mm+1,N),
     & dn_u(0:Lm+1,0:Mm+1), dm_v(0:Lm+1,0:Mm+1),
     & dm_u(0:Lm+1,0:Mm+1), dn_v(0:Lm+1,0:Mm+1),
     & visc2_r(0:Lm+1,0:Mm+1), visc2_p(0:Lm+1,0:Mm+1)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
# 89 "R_tools_fort_routines_gula/visc3d_S_baham.F" 2



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,u,v,z_r,z_w,pm,pn,f,dt,rmask,umask, vmask,visc2, v_sponge,coord,coordmax
Cf2py intent(out) MHmix


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! define visc coefficient (from set_nudgcof.F )
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      LLm = coordmax(4)-2
      MMm = coordmax(2)-2


      !isp=min((LLm+1)/12,(MMm+1)/12)
      ! isp=(LLm+1)/12 !old version of the code

      ! for BAHAN. BAHAM
      isp=min((LLm+1)/20,(MMm+1)/20)


       do j=max(-1,jstr-1),jend
         do i=max(-1,istr-1),iend
          ibnd=isp


          ibnd=min(ibnd,i+coord(3))


          ibnd=min(ibnd,LLm+1-i-coord(3))


          ibnd=min(ibnd,j+coord(1))


          ibnd=min(ibnd,MMm+1-j-coord(1))


          wrk(i,j)=float(isp-ibnd)/float(isp)
        enddo
       enddo


       do j=jstr-1,jend+1
        do i=istr-1,iend+1


          visc2_r(i,j)=visc2 + v_sponge*wrk(i,j)


        enddo
       enddo

       do j=jstr,jend+1
        do i=istr,iend+1


          visc2_p(i,j)=visc2 + 0.25*v_sponge*( wrk(i,j)
     & +wrk(i-1,j)+wrk(i,j-1)+wrk(i-1,j-1))


        enddo
       enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! define grid variables
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do j=jstrR,jendR
        do i=istrR,iendR
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo
              dm_r(i,j)=1./pm(i,j)
              dn_r(i,j)=1./pn(i,j)
        enddo
      enddo

      do j=jstrR,jendR
        do i=istr,iendR
            dn_u(i,j) = 2./(pn(i,j)+pn(i-1,j))
            dm_u(i,j) = 2./(pm(i,j)+pm(i-1,j))
          enddo
      enddo



      do j=jstr,jendR
        do i=istrR,iendR
            dm_v(i,j) = 2./(pm(i,j)+pm(i,j-1))
            dn_v(i,j) = 2./(pn(i,j)+pn(i,j-1))
          enddo
      enddo



! Compute n/m and m/n at horizontal PSI-points.
! Set mask according to slipperness parameter gamma.
!
      do j=jstrV-1,jend
        do i=istrU-1,iend


          dm_p(i,j)=4./(pm(i,j)+pm(i,j-1)+pm(i-1,j)+pm(i-1,j-1))
          dn_p(i,j)=4./(pn(i,j)+pn(i,j-1)+pn(i-1,j)+pn(i-1,j-1))


c pmask(i,j)=rmask(i,j)*rmask(i-1,j)*rmask(i,j-1)
c & *rmask(i-1,j-1)
C*** if (gamma2.lt.0.) pmask(i,j)=2.-pmask(i,j)
!
! Set no-slip boundary conditions on land-mask boundaries
! regardless of supplied value of gamma2.
!

          cff1=1. !<-- computation of off-diagonal nonlinear terms
          cff2=2.

          if (rmask(i-1,j ).gt.0.5 .and. rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and. rmask(i,j-1).gt.0.5) then
            pmask(i,j)=1.

          elseif(rmask(i-1,j ).lt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff1
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).lt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff1
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).lt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff1
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).lt.0.5) then
            pmask(i,j)=cff1


          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).lt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).lt.0.5) then
            pmask(i,j)=cff2
          elseif(rmask(i-1,j ).lt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).lt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff2
          elseif(rmask(i-1,j ).gt.0.5 .and.rmask(i,j ).gt.0.5 .and.
     & rmask(i-1,j-1).lt.0.5 .and.rmask(i,j-1).lt.0.5) then
            pmask(i,j)=cff2
          elseif(rmask(i-1,j ).lt.0.5 .and.rmask(i,j ).lt.0.5 .and.
     & rmask(i-1,j-1).gt.0.5 .and.rmask(i,j-1).gt.0.5) then
            pmask(i,j)=cff2

          else
            pmask(i,j)=0.
          endif


        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


! Compute horizontal harmonic viscosity along constant S-surfaces.
!
! Compute flux-components of the horizontal divergence of the stress
! tensor (m5/s2) in XI- and ETA-directions.
!
      do k=1,N


        do j=jstrV-1,jend
          do i=istrU-1,iend
            cff=0.5*Hz(i,j,k)*visc2_r(i,j)*(

     & dn_r(i,j)*pm(i,j)*( (pn(i ,j)+pn(i+1,j))*u(i+1,j,k)
     & -(pn(i-1,j)+pn(i ,j))*u(i ,j,k)
     & )

     & -dm_r(i,j)*pn(i,j)*( (pm(i,j )+pm(i,j+1))*v(i,j+1,k)
     & -(pm(i,j-1)+pm(i,j ))*v(i,j ,k)
     & ))
            UFx(i,j)= cff*dn_r(i,j)*dn_r(i,j)
            VFe(i,j)= -cff*dm_r(i,j)*dm_r(i,j)

          enddo
        enddo



        do j=jstr,jend+1
          do i=istr,iend+1
            cff=0.125*(Hz(i-1,j,k)+Hz(i,j,k)+Hz(i-1,j-1,k)
     & +Hz(i,j-1,k))*visc2_p(i,j)*(

     & 0.25*(pm(i-1,j)+pm(i,j)+pm(i-1,j-1)+pm(i,j-1))*dn_p(i,j)
     & *( (pn(i ,j-1)+pn(i ,j))*v(i ,j,k)
     & -(pn(i-1,j-1)+pn(i-1,j))*v(i-1,j,k)
     & )

     & +0.25*(pn(i-1,j)+pn(i,j)+pn(i-1,j-1)+pn(i,j-1))*dm_p(i,j)
     & *( (pm(i-1,j )+pm(i,j ))*u(i,j ,k)
     & -(pm(i-1,j-1)+pm(i,j-1))*u(i,j-1,k)
     & ))

     & *pmask(i,j)

            UFe(i,j)= cff*dm_p(i,j)*dm_p(i,j)
            VFx(i,j)= cff*dn_p(i,j)*dn_p(i,j)
          enddo
        enddo




!
!
! if (k.eq.1) then
! write(*,*) 'u(1:5,82,0,nstp)',u(1:5,82,1)
! write(*,*) 'v(1:5,82,0,nstp)',v(1:5,82,1)
! write(*,*) 'u(1:5,81,0,nstp)',u(1:5,81,1)
! write(*,*) 'v(1:5,81,0,nstp)',v(1:5,81,1)
! write(*,*) 'UFx(i,j)',UFx(1:5,82)
! write(*,*) 'VFe(i,j)',VFe(1:5,82)
! write(*,*) 'UFe(i,j)',UFe(1:5,82)
! write(*,*) 'VFx(i,j)',VFx(1:5,82)
! write(*,*) 'dm_p(i,j)*dm_p(i,j)',dm_p(5,82),dn_p(5,82)
! write(*,*) 'pmask(i,j)',pmask(5,82)
! write(*,*) 'pm',pm(4:5,81:82)
! write(*,*) 'pn',pn(4:5,81:82)
! write(*,*) 'visc2_p',visc2_p(5,82)
! write(*,*) 'Hz',Hz(4,82,1)+Hz(5,82,1)+Hz(4,81,1)+Hz(5,81,1)
! endif
!
!



!
! Apply viscous terms. Note that at this stage arrays u,v(...,3-nstp)
! contain Hz*U and Hz*V with units of [m2/s]. Also compute vertical
! integral of viscous terms and add it into coupling terms for the
! barotropic mode
!

          do j=jstr,jend
            do i=istrU,iend
              cff=0.125*(pm(i-1,j)+pm(i,j))*(pn(i-1,j) +pn(i,j))
     & *( (pn(i-1,j)+pn(i,j))*(UFx(i,j)-UFx(i-1,j))
     & +(pm(i-1,j)+pm(i,j))*(UFe(i,j+1)-UFe(i,j))
     & )

              MHmix(i,j,k,1) = cff *dm_u(i,j)*dn_u(i,j)

            enddo
          enddo


        do j=jstrV,jend
          do i=istr,iend
            cff=0.125*(pm(i,j)+pm(i,j-1))*(pn(i,j) +pn(i,j-1))
     & *( (pn(i,j-1)+pn(i,j))*(VFx(i+1,j)-VFx(i,j))
     & +(pm(i,j-1)+pm(i,j))*(VFe(i,j)-VFe(i,j-1))
     & )


              MHmix(i,j,k,2) = cff*dm_v(i,j)*dn_v(i,j)



            enddo
          enddo


      enddo
      return
      end
# 182 "R_tools_fort_gula.F" 2

# 1 "R_tools_fort_routines_gula/get_akv.F" 1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subpart of the lmd_kpp.F routine (myroms version)
! used to compute the Kv
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





c----#define WND_AT_RHO_POINTS

      subroutine get_akv (Lm,Mm,N,alpha,beta, z_r,z_w
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
# 30 "R_tools_fort_routines_gula/get_akv.F" 2

      parameter (NT=2)
      parameter (itemp=1,isalt=2)
      parameter (epsil=1.E-16)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      real ustar3,
     & Bfsfc, zscale,
     & zetahat, ws,wm,

     & zscaleb


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
     & nu0c=0.1,Cv=1.8,
     & betaT=-0.2,epssfc=0.1,C_MO=1.,C_Ek=258.,
     & Cstar=10.,eps=1.E-20,zeta_m=-0.2,a_m=1.257,
     & c_m=8.360,zeta_s=-1.0,a_s=-28.86,c_s=98.96,
     & r2=0.5, r3=1./3., r4=0.25)




      real ustar2, Kv0, Kt0, Ks0, my_Akv_bak,
     & my_Akt_bak, my_Aks_bak
      real*8 hbbl(0:Lm+1,0:Mm+1)


      real*8 hbl(0:Lm+1,0:Mm+1)



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

      real*8 Kt(0:Lm+1,0:Mm+1,0:N)

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
# 135 "R_tools_fort_routines_gula/get_akv.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Cf2py intent(in) Lm,Mm,N,alpha,beta ,z_r,z_w,stflx,srflx, swr_frac, sustr, svstr ,Ricr,hbls, f, u, v, bvf, rmask, r_D
Cf2py intent(out) Kv

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
# 196 "R_tools_fort_routines_gula/get_akv.F"
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





!================================
! Search for mixed layer depth
!================================
!





      do k=1,N-1
        do i=imin,imax
          cff=1./(Hz(i,j,k)+Hz(i,j,k+1))
          wrk1(i,k)=cff*( u(i,j,k+1)+u(i+1,j,k+1)
     & -u(i,j,k )-u(i+1,j,k ))
          wrk2(i,k)=cff*( v(i,j,k+1)+v(i,j+1,k+1)
     & -v(i,j,k )-v(i,j+1,k ))
        enddo
      enddo

      do i=imin,imax
        wrk1(i,N)=wrk1(i,N-1)
        wrk2(i,N)=wrk2(i,N-1)
        wrk1(i,0)=wrk1(i, 1)
        wrk2(i,0)=wrk2(i, 1)
      enddo


      do k=N,1,-1
        do i=imin,imax
          zscale=z_w(i,j,N)-z_w(i,j,k-1)
! zscaleb=z_w(i,j,k)-z_w(i,j,0)
          Kern=zscale/(zscale+epssfc*hbl(i,j))
! # ifdef
! & *zscaleb/(zscaleb+epssfc*hbbls(i,j))
! # endif
          Bfsfc=Bo(i,j) +Bosol(i,j)*(1.-swr_frac(i,j,k-1))

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
# 298 "R_tools_fort_routines_gula/get_akv.F" 2

          cff=bvf(i,j,k)*bvf(i,j,k-1)
          if (cff.gt.0.D0) then
            cff=cff/(bvf(i,j,k)+bvf(i,j,k-1))
          else
            cff=0.D0
          endif



          FC(i,k-1)=FC(i,k) + Kern*Hz(i,j,k)*(
     & 0.375*( wrk1(i,k)**2+wrk1(i,k-1)**2
     & +wrk2(i,k)**2 +wrk2(i,k-1)**2 )
     & +0.25 *(wrk1(i,k-1)*wrk1(i,k)
     & +wrk2(i,k-1)*wrk2(i,k))
     & -Ri_inv*( cff + 0.25*(bvf(i,j,k)+bvf(i,j,k-1)))
     & -C_Ek*f(i,j)*f(i,j) )



          Vtsq=Vtc*ws*sqrt(max(0., bvf(i,j,k-1)))

          Cr(i,k-1)=FC(i,k-1) +Vtsq
          if (kbl(i).eq.0 .and. Cr(i,k-1).lt.0.) kbl(i)=k






        enddo
      enddo





      do i=imin,imax
c?? if (kbl(i).eq.N) then
c?? hbl(i,j)=z_w(i,j,N)-z_w(i,j,N-1)

        if (kbl(i).gt.0) then
          k=kbl(i)
          hbl(i,j)=z_w(i,j,N)-( z_w(i,j,k-1)*Cr(i,k)
     & -z_w(i,j,k)*Cr(i,k-1)
     & )/(Cr(i,k)-Cr(i,k-1))

c** if (Cr(i,k)*Cr(i,k-1).gt.0.D0 ) write(*,*)
c** & '### ERROR', k, Cr(i,k), Cr(i,k-1), hbl(i,j)

        else
          hbl(i,j)=z_w(i,j,N)-z_w(i,j,0)+eps
        endif
# 367 "R_tools_fort_routines_gula/get_akv.F"
        hbl(i,j)=hbl(i,j)*rmask(i,j)

      enddo




!
!======================================
! Search for bottom mixed layer depth
!======================================
!
        do i=imin,imax
          kbl(i) = 0 ! reset Cr at bottom and kbl for BKPP
          Cr(i,0) = 0.
        enddo
        do k=1,N,+1
          do i=imin,imax
            Cr(i,k)=FC(i,k)-FC(i,0)
            if (kbl(i).eq.0 .and. Cr(i,k).gt.0.) kbl(i)=k
          enddo
        enddo
        do i=imin,imax
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

          hbbl(i,j)=hbbl(i,j)*rmask(i,j)

        enddo !--> discard FC, Cr and kbl


!======================================



!================================
! Smoothing hbl/hbbl
!================================
!

      enddo !<-- j terminate j-loop only if smothing takes place.


# 1 "R_tools_fort_routines_gula/kpp_smooth.h" 1
!
! Apply horizontal smoothing operator to hbl, while avoiding land-
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
          FX(i,j)=(hbl(i,j)-hbl(i-1,j))
        enddo
      enddo
      do j=jstr,jend
        do i=istr,iend
          FE(i,j)=(hbl(i,j)-hbl(i,j-1))
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
          hbl(i,j)=hbl(i,j)+cff1*( FX(i+1,j)-FX(i,j)
     & +FE1(i,j+1)-FE1(i,j))
        enddo !--> discard FX,FE,FE1
      enddo
# 419 "R_tools_fort_routines_gula/get_akv.F" 2




# 1 "R_tools_fort_routines_gula/kpp_smooth.h" 1
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
# 424 "R_tools_fort_routines_gula/get_akv.F" 2


      do j=jstr,jend !--> restart j-loop
# 441 "R_tools_fort_routines_gula/get_akv.F"
!




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
# 481 "R_tools_fort_routines_gula/get_akv.F" 2


          f1=5.0 * max(0., Bfsfc) * vonKar/(ustar(i,j)**4+eps)



          cff=1./(z_w(i,j,k)-z_w(i,j,k-1))
          cff_up=cff*(z_bl -z_w(i,j,k-1))
          cff_dn=cff*(z_w(i,j,k) -z_bl)

          Av_bl=cff_up*Kv(i,j,k)+cff_dn*Kv(i,j,k-1)
          dAv_bl=cff * (Kv(i,j,k) - Kv(i,j,k-1))
          Gm1(i)=Av_bl/(hbl(i,j)*wm+eps)
          dGm1dS(i)=min(0., Av_bl*f1-dAv_bl/(wm+eps))

! At_bl=cff_up*Kt(i,j,k)+cff_dn*Kt(i,j,k-1)
! dAt_bl=cff * (Kt(i,j,k) - Kt(i,j,k-1))
! Gt1(i)=At_bl/(hbl(i,j)*ws+eps)
! dGt1dS(i)=min(0., At_bl*f1-dAt_bl/(ws+eps))
!
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
# 521 "R_tools_fort_routines_gula/get_akv.F" 2
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
          do k=kbl(i)-1,1,-1

            ghat(i,j,k)=0.
# 566 "R_tools_fort_routines_gula/get_akv.F"
          enddo
        enddo
!
!================================
! Bottom KPP
!================================
!

        do i=istr,iend
          kbl(i)=N !<-- initialize search
        enddo
        do k=N-1,1,-1 ! find new boundary layer index "kbl".
          do i=istr,iend
            if (z_r(i,j,k)-z_w(i,j,0).gt.hbbl(i,j)) kbl(i)=k
          enddo
        enddo
!



!================================

        Zob=0.01

! Compute nondimensional shape function coefficients Gx( ) by
! matching values and vertical derivatives of interior mixing
! coefficients at hbbl (sigma=1).
!
        do i=istr,iend
          ustar2 = r_D(i,j)*sqrt(
     & ( (0.5*(u(i,j,1)+u(i+1,j,1)))**2
     & +(0.5*(v(i,j,1)+v(i,j+1,1)))**2 ) )
          wm=vonKar*sqrt(ustar2) ! turbulent velocity scales (wm,ws)
          ws=wm ! with buoyancy effects neglected.
          k=kbl(i)
          z_bl=z_w(i,j,0)+hbbl(i,j)
          if (z_bl.lt.z_w(i,j,k-1)) k=k-1

          cff=1./(z_w(i,j,k)-z_w(i,j,k-1))
          cff_up=cff*(z_bl -z_w(i,j,k-1))
          cff_dn=cff*(z_w(i,j,k) -z_bl)

          Av_bl=cff_up*Kv(i,j,k)+cff_dn*Kv(i,j,k-1)
          dAv_bl=cff * (Kv(i,j,k) - Kv(i,j,k-1))
          Gm1(i)=Av_bl/(hbbl(i,j)*wm+eps)
          dGm1dS(i)=min(0., -dAv_bl/(wm+eps))
!
! At_bl=cff_up*Kt(i,j,k)+cff_dn*Kt(i,j,k-1)
! dAt_bl=cff * (Kt(i,j,k) - Kt(i,j,k-1))
! Gt1(i)=At_bl/(hbbl(i,j)*ws+eps)
! dGt1dS(i)=min(0., -dAt_bl/(ws+eps))
!
! # ifdef
! As_bl=cff_up*Ks(i,j,k)+cff_dn*Ks(i,j,k-1)
! dAs_bl=cff * (Ks(i,j,k) - Ks(i,j,k-1))
! Gs1(i)=As_bl/(hbbl(i,j)*ws+eps)
! dGs1dS(i)=min(0., -dAs_bl/(ws+eps))
! # endif
!
! Compute boundary layer mixing coefficients.
!--------- -------- ----- ------ -------------
! Compute turbulent velocity scales at vertical W-points.

          do k=1,N-1
            if (k.lt.kbl(i)) then
              sigma=min((z_w(i,j,k)-z_w(i,j,0)+Zob)/(hbbl(i,j)+Zob),1.)
              a1=sigma-2.
              a2=3.-2.*sigma
              a3=sigma-1.


              if (sigma.lt.0.07D0) then
                cff=0.5*(sigma-0.07D0)**2/0.07D0
              else
                cff=0.D0
              endif


              Kv0 =wm*hbbl(i,j)*(cff + sigma*( 1.+sigma*(
     & a1+a2*Gm1(i)+a3*dGm1dS(i) )))
! Kt0 =ws*hbbl(i,j)*(cff + sigma*( 1.+sigma*(
! & a1+a2*Gt1(i)+a3*dGt1dS(i) )))
! # ifdef
! Ks0 =ws*hbbl(i,j)*(cff + sigma*( 1.+sigma*(
! & a1+a2*Gs1(i)+a3*dGs1dS(i) )))
! # endif
!
! If BBL reaches into SBL, take the max of surface and bottom values.
!
              z_bl=z_w(i,j,N)-hbl(i,j)
              if (z_w(i,j,k).gt.z_bl) then
                Kv0=max(Kv(i,j,k),Kv0)
! Kt0=max(Kt(i,j,k),Kt0)
! # ifdef
! Ks0=max(Ks(i,j,k),Ks0)
! # endif
              endif
              Kv(i,j,k)=Kv0
! Kt(i,j,k)=Kt0
! # ifdef
! Ks(i,j,k)=Ks0
! # endif


            else !<-- k > kbl(i)
              if (bvf(i,j,k).lt.0.) then

                z_bl=z_w(i,j,N)-hbl(i,j)
                if (z_w(i,j,k).lt.z_bl) then

                  Kv(i,j,k)=Kv(i,j,k) + nu0c ! Add convective
! Kt(i,j,k)=Kt(i,j,k) + nu0c ! adjustment outside
! # ifdef
! Ks(i,j,k)=Ks(i,j,k) + nu0c ! of mixed layers.
! # endif

                endif

              endif

            endif !<-- k < kbl(i)
          enddo !<-- k
        enddo !<-- i

!
!================================
! Finalize
!================================
!


      enddo !<-- j



!======================================


      return
      end
# 184 "R_tools_fort_gula.F" 2

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
# 186 "R_tools_fort_gula.F" 2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 1 "R_tools_fort_routines_gula/solve_ttw.F" 1

      subroutine solve_ttw(Lm,Mm,N,bx,by,Av,sustr,svstr,f,pm,pn,
     & z_r,z_w,u,v,ug,vg)


      implicit none

      integer Lm,Mm,N, imin,imax,jmin,jmax, i,j,k,
     & istr,iend,jstr,jend,istrU,jstrV


      real*8 u(1:Lm+1,0:Mm+1,N), v(0:Lm+1,1:Mm+1,N),
     & ug(1:Lm+1,0:Mm+1,N), vg(0:Lm+1,1:Mm+1,N),
     & z_r(0:Lm+1,0:Mm+1,N),z_w(0:Lm+1,0:Mm+1,0:N),
     & Av(0:Lm+1,0:Mm+1,0:N),
     & Hz(0:Lm+1,0:Mm+1,N),f(0:Lm+1,0:Mm+1),
     & pm(0:Lm+1,0:Mm+1), pn(0:Lm+1,0:Mm+1),
     & sustr(0:Lm+1,0:Mm+1), svstr(0:Lm+1,0:Mm+1),
     & bx(0:Lm+1,0:Mm+1,N), by(0:Lm+1,0:Mm+1,N)
! & AA(0:Lm+1,0:Mm+1,0:N),rho(0:Lm+1,0:Mm+1,N)

      real b11,b12,b21,b22, det, TauX,TauY
      real, dimension(0:N) :: FC, c11,c12,c21,c22
      real, dimension(N) :: d1,d2

      real cff, cff0,cff1,cff2,cff3, xx

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
# 29 "R_tools_fort_routines_gula/solve_ttw.F" 2


Cf2py intent(in) Lm,Mm,N,bx,by,Av,sustr,svstr,f,pm,pn,z_r,z_w
Cf2py intent(out) u,v,ug,vg

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      imin=0
      imax=Lm+1
      jmin=0
      jmax=Mm+1

      istr=1
      iend=Lm
      jstr=1
      jend=Mm

      istrU=istr+1
      jstrV=jstr+1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





      do j=jmin,jmax
        do i=imin,imax
          do k=1,N,+1
           Hz(i,j,k) = z_w(i,j,k) - z_w(i,j,k-1)
          enddo
        enddo
      enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! do k=1,N,+1 ! Put rho on Psi-grid
! do j=jstr,jend+1
! do i=istr,iend+1
! AA(i,j,k)=0.25*( rho(i,j,k) +rho(i-1,j,k)
! & +rho(i,j-1,k) +rho(i-1,j-1,k)
! & )
! enddo
! enddo
! enddo !<-- k


! do k=1,N,+1
! if (k == 1) then ! The following segment converts
! do j=jstrV-1,jend+1 ! "AA" from density anomaly to
! do i=istrU-1,iend+1 ! hydrostatic pressure anomaly.
! AA(i,j,0)=0. ! <-- initialize vertical
! enddo ! integration
! enddo
! else
! cff=0.125
! ! cff=0.5
! do j=jstrV-1,jend+1
! do i=istrU-1,iend+1
! xx=AA(i,j,0) !<-- save (k-1)th value
!
! AA(i,j,0)=AA(i,j,0) +cff*(AA(i,j,k)+AA(i,j,k-1))*(
! & z_r(i,j,k)+z_r(i-1,j,k)+z_r(i,j-1,k)+z_r(i-1,j-1,k)
! & -z_r(i,j,k-1)-z_r(i-1,j,k-1)
! & -z_r(i,j-1,k-1)-z_r(i-1,j-1,k-1)
! & )
!
! ! AA(i,j,0)=AA(i,j,0) +cff*(AA(i,j,k)+AA(i,j,k-1))*(
! ! & z_r(i,j,k)
! ! & -z_r(i,j,k-1)
! ! & )
!
! AA(i,j,k-1)=xx
!
! enddo
! enddo
! endif
! enddo !<-- k
!
! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! do j=jstrV-1,jend+1 ! After the above steps "AA(:,:,k)"
! do i=istrU-1,iend+1 ! becomes baroclinic pressure anomaly
! AA(i,j,N)=AA(i,j,0) ! normalized as P/(g*rho0) and defined
! do k=1,N
! AA(i,j,k) = AA(i,j,k) - AA(i,j,N) + z_w(i,j,N)
! enddo
! enddo ! at horizontal PSI- and vertical RHO
! enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do j=jstr,jend ! Compute horizontal pressure gradient
        do i=istr,iend ! components and place them into d1,d2


! cff=0.5*g
! do k=1,N
! d1(k)=cff*pm(i,j)*Hz(i,j,k)*( AA(i,j ,k)-AA(i+1,j ,k)
! & +AA(i,j+1,k)-AA(i+1,j+1,k))
!
! d2(k)=cff*pn(i,j)*Hz(i,j,k)*( AA(i,j ,k)+AA(i+1,j ,k)
! & -AA(i,j+1,k)-AA(i+1,j+1,k))
! enddo

         do k=1,N
           d1(k)= -Hz(i,j,k)*bx(i,j ,k)
           d2(k)= -Hz(i,j,k)*by(i,j ,k)
         enddo

          TauX=sustr(i,j)
          TauY=svstr(i,j)

          do k=1,N-1
            FC(k)=2.*Av(i,j,k)/(Hz(i,j,k)+Hz(i,j,k+1))
          enddo
          FC(0)=0.
          FC(N)=0.

! Block-tri-diagonal problem for u,v-velocity components which are in
! simultaneous geostrophic balance with the density field initialized
! above and in viscous-Coriolis (Ekman) balance with each other.

          b11=FC(1) ! Free-slip bottom boundary
          b22=FC(1) ! condition: everything is the
          b12=-f(i,j)*Hz(i,j,1) ! same as it would be in k-loop
          b21= f(i,j)*Hz(i,j,1) ! for k=1 with all (k-1)-terms
                                        ! vanishing,
          det=1./(b11*b22-b12*b21) ! FC(k-1) --> FC(0) --> 0

          c11(1)= det*b22*FC(1) ; c12(1)=-det*b12*FC(1)
          c21(1)=-det*b21*FC(1) ; c22(1)= det*b11*FC(1)

! d1(1)= det*( b22*d1(1) -b12*d2(1))
! d2(1)= det*(-b21*d1(1) +b11*d2(1))

          cff1 = det*( b22*d1(1) -b12*d2(1))
          d2(1)= det*(-b21*d1(1) +b11*d2(1))
          d1(1) = cff1

          do k=2,N-1,+1 !--> forward sweep
            b11=FC(k) +FC(k-1)*(1.-c11(k-1)) ! of block-Gaussian
            b22=FC(k) +FC(k-1)*(1.-c22(k-1)) ! elimination
            b12=-f(i,j)*Hz(i,j,k) -FC(k-1)*c12(k-1)
            b21= f(i,j)*Hz(i,j,k) -FC(k-1)*c21(k-1)

            det=1./(b11*b22-b12*b21)

            c11(k)= det*b22*FC(k) ; c12(k)=-det*b12*FC(k)
            c21(k)=-det*b21*FC(k) ; c22(k)= det*b11*FC(k)

            cff1=d1(k)+FC(k-1)*d1(k-1)
            cff2=d2(k)+FC(k-1)*d2(k-1)
            d1(k)= det*( b22*cff1 -b12*cff2)
            d2(k)= det*(-b21*cff1 +b11*cff2)
          enddo

          b11= FC(N-1)*(1.-c11(N-1)) ! surface boundary
          b22= FC(N-1)*(1.-c22(N-1)) ! condition: apply
          b12=-f(i,j)*Hz(i,j,N) -FC(N-1)*c12(N-1) ! wind stress, i.e.,
          b21= f(i,j)*Hz(i,j,N) -FC(N-1)*c21(N-1) ! replace what would
                                                 ! be upper interface
          det=1./(b11*b22-b12*b21) ! implicit viscous
                                                 ! flux
          cff1=d1(N)+TauX +FC(N-1)*d1(N-1) ! FC(k)*(u(k+1)-u(k))
          cff2=d2(N)+TauY +FC(N-1)*d2(N-1) ! at k=N with TauX,y.
          d1(N)= det*( b22*cff1 -b12*cff2)
          d2(N)= det*(-b21*cff1 +b11*cff2)

          do k=N-1,1,-1 !--> backsubstitution
            d1(k)=d1(k) +c11(k)*d1(k+1)+c12(k)*d2(k+1)
            d2(k)=d2(k) +c21(k)*d1(k+1)+c22(k)*d2(k+1)
          enddo

! After this moment d1,d2 contain horizontal velocity components u,v
! computed at common location on C-grid (at RHO-points). Interpolate
! them to their native placements.

          if (i < iend .and. j >= jstr) then ! Note: this code
            do k=1,N ! segment relies on
              u(i+1,j,k)=0.5*d1(k) ! the fact that
            enddo ! i,j-indices are
          endif ! increasing in their
          if (i >= istrU .and. j >= jstr) then ! respective loops
            do k=1,N ! enclosing this part,
              u(i,j,k)=u(i,j,k)+0.5*d1(k) ! hence for each
            enddo ! individual value of
          endif ! "i" in computation
                                                 ! of "u" the upper
          if (i >= istr .and. j < jend) then ! "if" statement is
            do k=1,N ! executed first.
              v(i,j+1,k)=0.5*d2(k) ! Same applies to "j"
            enddo ! and "v".
          endif
          if (i >= istr .and. j >= jstrV) then
            do k=1,N
              v(i,j,k)=v(i,j,k)+0.5*d2(k)
            enddo
          endif
        enddo !<-- i
      enddo !<-- j

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      do k=1,N ! Alternative to VTWB: compute velocity
                              ! components from geostrophic balance at
      do j=jstr,jend ! their native C-grid locations.
        do i=istrU,iend
          ug(i,j,k)=-by(i,j,k)/f(i,j)
        enddo
      enddo
      do j=jstrV,jend
        do i=istr,iend
          vg(i,j,k)=bx(i,j,k)/f(i,j)
        enddo
      enddo
      enddo


! do k=1,N ! Alternative to VTWB: compute velocity
! cff=0.5*g ! components from geostrophic balance at
! do j=jstr,jend ! their native C-grid locations.
! do i=istrU,iend
! ug(i,j,k)=-cff*(pn(i,j)+pn(i-1,j))*(AA(i,j+1,k)-AA(i,j,k))/f(i,j)
! enddo
! enddo
! do j=jstrV,jend
! do i=istr,iend
! vg(i,j,k)=+cff*(pm(i,j)+pm(i,j-1))*(AA(i+1,j,k)-AA(i,j,k))/f(i,j)
! enddo
! enddo
! enddo


      return
      end
# 191 "R_tools_fort_gula.F" 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Test if a (1-D) array if fortran formatted or not
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine test_arg_1d (Lm, T)

      implicit none
      integer Lm
      real*4 T(0:Lm+1)
Cf2py intent(in) Lm
Cf2py intent(inout) T

      write(*,*) T(2)

      end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Test if a (1-D) array if fortran formatted or not
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine test_arg_1d_double (Lm, T)

      implicit none
      integer Lm
      real*8 T(0:Lm+1)
Cf2py intent(in) Lm
Cf2py intent(inout) T

      write(*,*) T(2)

      end


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Test if a (2-D) array if fortran formatted or not
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine test_arg_2d (Lm,Mm, T)

      implicit none
      integer Lm,Mm
      real*4 T(0:Lm+1,0:Mm+1)
Cf2py intent(in) Lm,Mm
Cf2py intent(inout) T

      write(*,*) T(2,2)

      end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Test if a (2-D) array if fortran formatted or not
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine test_arg_2d_double (Lm,Mm, T)

      implicit none
      integer Lm,Mm
      real*8 T(0:Lm+1,0:Mm+1)
Cf2py intent(in) Lm,Mm
Cf2py intent(inout) T

      write(*,*) T(2,2)

      end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Test if a 3-D array if fortran formatted or not
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine test_arg_3d (Lm,Mm,N, T)


      implicit none
      integer Lm,Mm,N
      real*4 T(0:Lm+1,0:Mm+1,N)
Cf2py intent(in) Lm,Mm,N
Cf2py intent(inout) T

      write(*,*) T(2,2,2)

      end



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Test if a 3-D array if fortran formatted or not
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine test_arg_3d_double (Lm,Mm,N, T)


      implicit none
      integer Lm,Mm,N
      real*8 T(0:Lm+1,0:Mm+1,N)
Cf2py intent(in) Lm,Mm,N
Cf2py intent(inout) T

      write(*,*) T(2,2,2)

      end
