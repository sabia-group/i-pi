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
