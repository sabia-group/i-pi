! Parametrized for a layer of h2 molecules on fcc sites of Ag(111) surface with PBE+vdWTS
      SUBROUTINE getljh2ag(q,pot,force,nat)
         IMPLICIT NONE
         integer ::  nat
         real*8  :: q(:,3), pot, force(:,3)
         real*8  :: nq, dr, diff(3), a, b, c, zdist, fcom
         a = 1.02524232e-03
         b = 7.60164990e-01
         c = 5.87553087e+00
         ! Very non-general. Assumes that pairs of H atoms are connected and accepts only H atoms
         zsurf=13.095154 ! this has to become a parameter that is passed in. Correspnds to 6.929656887500001 angstrom
         do i=1, nat-1, 2
            com(:) = (q(i,:)+q(i+1,:))/2.d0
            zcom = diff(3)-zsurf
            pot = a*(1-dexp(-b*(zcom-c)))**2-a
            fcom = -2.d0*a*b*(dexp(-b*(zcom-x))-dexp(-2.d0*b*(zcom-x)))
            force(i,3)=force(i+1,3)= 0.5*fcom
            force(i,2)=force(i+1,2)=0.d0
            force(i,1)=force(i+1,1)=0.d0
         enddo
      END SUBROUTINE
