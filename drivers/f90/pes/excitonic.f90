! exciton effective potentials in 1D and 2D
! M. Rossi 2022

      SUBROUTINE screencoulomb_3D(nat,q,pot,force, screen)
        ! Notes: nat is always 2 here. screen is the dielectric constant which is assumed to be isotropic in 3D
        ! Expected units: atomic units
        ! V(q1, q2) = e^2/4 pi eps_0 1/screen 1/|q1-q2| -> 1/screen 1/|q1-q2| in a.u.
        IMPLICIT NONE
        integer      :: nat
        real(kind=8) :: q(nat,3),pot,force(nat,3), screen
        real(kind=8) :: distvec(3), dist, distfh
!        real(kind=8) :: k


!        k  = 1836*(3800.0d0/219323d0)**2
        distvec(:)     = q(1,:)-q(2,:)
        dist           = sqrt(dot_product(distvec, distvec))
        pot            = 1.d0/(screen*dist)
        distfh         = dist**(5.d0/2.d0)
        force(1,1)     = distvec(1)/(screen*distfh)
        force(1,2)     = distvec(2)/(screen*distfh)
        force(1,3)     = distvec(3)/(screen*distfh)
        force(2,1)     = -force(1,1)
        force(2,2)     = -force(1,2)
        force(2,3)     = -force(1,3)

        return
      end

    SUBROUTINE screen_dipole3D(nat,q,dip)
        IMPLICIT NONE
        integer                   :: nat
        real(kind=8),intent(inout):: q(nat,3)
        real(kind=8),intent(out)  :: dip(3)

        dip = 0.0d0
        dip(1) = q(1,1)-q(2,1)
        dip(2) = q(1,2)-q(2,2)
        dip(3) = q(1,3)-q(2,3)
!check !dip(:) = q(1,:)-q(2,:)
        return
    END

      SUBROUTINE screencoulomb_2D(nat,q,pot,force, alpha2D)
        ! Notes: nat is always 2 here. Implementing only Eq. 10 of PRB 84, 085406 (2011)
        ! Expected units: atomic units
        ! V(q1, q2) = -1/r0 (ln (rho/(rho+r0)) + (gamma + ln2) e^(-rho/r0))
        ! r0 = 2 pi alpha2D
        ! rho = distance in the plane
        ! gamma = 0.5772156649 (Euler constant) 
        ! r0 seems to need to have units of bohr^2 which should be correct if it is prop to alpha2D
        IMPLICIT NONE
        integer      :: nat
        real(kind=8) :: q(nat,3),pot,force(nat,3), alpha2D
        real(kind=8) :: distvec(2), rho 
        real(kind=8) :: gam, k, r0, pi, x1, x2, y1, y2

        pi=4.D0*DATAN(1.D0)
        k  = 1836*(3800.0d0/219323d0)**2 ! This will be a constrain in z, for an effective 2D potential. How stiff this should be, not yet sure.
        distvec(:)     = q(1,1:2)-q(2,1:2)
        rho            = sqrt(dot_product(distvec, distvec))
        r0             = 2*pi*alpha2D
        pot            = -1.d0/r0 * (log(dist/(dist+r0)) + (gam + log(2.d0) * exp(-dist/r0))) 
        pot            = pot + 0.5 * k * (q(1,3)-q(2,3))**2  ! adding spring contraint term
        ! next is from Mathematica. Probably should get better expressions
        x1 = q(1,1)
        x2 = q(2,1)
        y1 = q(1,2)
        y2 = 1(2,2)
        force(2,1) = (((x1 - x2)/(r0 + rho)**2 - (x1 - x2)/((r0 + rho)*Sqrt((x1 - x2)**2 + (y1 - y2)**2)) + ((x1 - x2)*(EulerGamma - Log(2)))/ (E**(rho/r0)*r0*Sqrt((x1 - x2)**2 + (y1 - y2)**2)))/ (r0*(rho/(r0 + rho) + (EulerGamma - Log(2))/E**(rho/r0))))
        force(1,2)     = distvec(2)/(screen*distfh)
        force(1,1)     = -force(1,1)
        force(2,2)     = -force(1,2)
        
        force(1,3)     = - k * (q(1,3)-q(2,3))
        force(2,3)     = - force(1,3)

        return
      end
