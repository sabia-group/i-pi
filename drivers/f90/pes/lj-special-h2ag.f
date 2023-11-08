! Parametrized for a layer of h2 molecules on fcc sites of Ag(111) surface with PBE+vdWTS
      SUBROUTINE getljh2ag(q,pot,force,nat)
         IMPLICIT NONE
         integer ::  nat
         real*8  :: q(nat,3), pot, force(nat,3)
         real*8  :: a, b, c, fcom, zsurf, zcom, com(3)
         integer :: i
         a = 2.41808209e-03 !1.02524232e-03
         b = 7.62206977e-01 !7.60164990e-01
         c = 5.46997694 !5.87553087e+00
         force(:,:)=0.d0
         ! Very non-general. Assumes that pairs of H atoms are connected and accepts only H atoms
         zsurf=13.088243 ! this has to become a parameter that is passed in. Correspnds to 6.926 angstrom
         do i=1, nat-1, 2
            com(:) = (q(i,:)+q(i+1,:))/2.d0
            zcom = com(3)-zsurf
            pot = a*(1-dexp(-b*(zcom-c)))**2-a
            fcom = -2.d0*a*b*(dexp(-b*(zcom-c))-dexp(-2.d0*b*(zcom-c)))
            force(i,3) = 0.5*fcom
            force(i+1,3) = force(i,3)
         enddo
      END SUBROUTINE
