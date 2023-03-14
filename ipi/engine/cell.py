"""Classes which deal with the system box.

Used for implementing the minimum image convention.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np
from numpy.linalg import norm, inv

from ipi.utils.depend import *
from ipi.utils.mathtools import *
from ipi.utils.messages import warning,verbosity


__all__ = ["Cell"]

def norm_cols(M):
    """normalize the columns of a matrix"""
    for i in range(M.shape[1]):
        M[:,i] = M[:,i] / norm(M[:,i])
    return M


class Cell(dobject):

    """Base class to represent the simulation cell in a periodic system.

    This class has the base attributes required for either flexible or
    isotropic cell dynamics. Uses an upper triangular lattice vector matrix to
    represent the cell.

    Depend objects:
       h: An array giving the lattice vector matrix.
       ih: An array giving the inverse of the lattice vector matrix.
       V: The volume of the cell.
    """

    def __init__(self, h=None):
        """Initialises base cell class.

        Args:
           h: Optional array giving the initial lattice vector matrix. The
              reference cell matrix is set equal to this. Must be an upper
              triangular 3*3 matrix. Defaults to a 3*3 zeroes matrix.
        """

        if h is None:
            h = np.zeros((3, 3), float)

        dself = dd(self)  # gets a direct-access view to self

        dself.h = depend_array(name="h", value=h)
        dself.ih = depend_array(
            name="ih",
            value=np.zeros((3, 3), float),
            func=self.get_ih,
            dependencies=[dself.h],
        )
        dself.V = depend_value(name="V", func=self.get_volume, dependencies=[dself.h])

    def copy(self):
        return Cell(dstrip(self.h).copy())

    def get_ih(self):
        """Inverts the lattice vector matrix."""

        return invert_ut3x3(self.h)

    def get_volume(self):
        """Calculates the volume of the system box."""

        return det_ut3x3(self.h)

    def apply_pbc(self, atom):
        """Uses the minimum image convention to return a particle to the
           unit cell.

        Args:
           atom: An Atom object.

        Returns:
           An array giving the position of the image that is inside the
           system box.
        """

        s = np.dot(self.ih, atom.q)

        for i in range(3):
            s[i] = s[i] - round(s[i])

        return np.dot(self.h, s)

    def array_pbc(self, pos):
        """Uses the minimum image convention to return a list of particles to the
           unit cell.

        Args:
           atom: An Atom object.

        Returns:
           An array giving the position of the image that is inside the
           system box.
        """

        s = dstrip(pos).copy()
        s.shape = (len(pos) // 3, 3)

        s = np.dot(dstrip(self.ih), s.T)
        s = s - np.round(s)

        s = np.dot(dstrip(self.h), s).T

        pos[:] = s.reshape((len(s) * 3))

    def minimum_distance(self, atom1, atom2):
        """Takes two atoms and tries to find the smallest vector between two
        images.

        This is only rigorously accurate in the case of a cubic cell,
        but gives the correct results as long as the cut-off radius is defined
        as smaller than the smallest width between parallel faces even for
        triclinic cells.

        Args:
           atom1: An Atom object.
           atom2: An Atom object.

        Returns:
           An array giving the minimum distance between the positions of atoms
           atom1 and atom2 in the minimum image convention.
        """

        s = np.dot(self.ih, atom1.q - atom2.q)
        for i in range(3):
            s[i] -= round(s[i])
        return np.dot(self.h, s)
    
    ###
    def _rlv2cart_M(self):
        """Return the cartesian component of a vector given its components w.r.t. the (normalized) reciprocal lattice vectors"""
        # self.h contains the lattice vectors (each column is a vector)
        # compute the reciprocal lattice vectors (each column is a vector)
        h = np.asarray(self.h.copy())
        B = inv(h.T)
        # normalize per column
        return norm_cols(B)
    
    def _lv2cart_M(self):
        """Return the cartesian component of a vector given its components w.r.t. the lattice vectors"""
        h = np.asarray(self.h.copy())
        return norm_cols(h)
    
    def _cart2rlv_M(self):
        return inv(self._rlv2cart_M())
    
    def _cart2lv_M(self):
        return norm_cols(self._lv2cart_M())
    
    # def rlv2cart(self,v):
    #     """Return the cartesian component of a vector given its components w.r.t. the (normalized) reciprocal lattice vectors"""
    #     # self.h contains the lattice vectors (each column is a vector)
    #     # compute the reciprocal lattice vectors (each column is a vector)
    #     h = np.asarray(self.h.copy())
    #     B = inv(h.T)
    #     # normalize per column
    #     B = norm_cols(B)
    #     # get the cartesian components
    #     return B @ v
    
    # def lv2cart(self,v):
    #     """Return the cartesian component of a vector given its components w.r.t. the lattice vectors"""
    #     h = np.asarray(self.h.copy())
    #     A = norm_cols(h)
    #     return A @ v
    
    def change_basis(self,orig,dest,v=None,M=None,output="v",verbose=False):

        """
        v != None: 
        orig: original basis set
        dest: final basis set

        M !+ None 
        orig: orig[0] is the original basis set for the rows, orig[1] for the columns 
        dest: dest[0] is the final basis set for the rows, dest[1] for the columns 
        """

        # ( v is None and M is None ) or
        if (v is not None and M is not None ) :
            raise ValueError("You have to specify a vector 'v' or a matrix 'M', but not both!")
        
        def check_value(value,valids):
            if value not in valids:
                raise ValueError("'{:s}' is not a valid basis choice.".format(value))
        
        # vector
        if v is not None or output == "R":
            valids = ["lv","rlv","cart"]
            check_value(orig,valids)
            check_value(dest,valids)
            if orig == dest :
                if verbose : 
                    print("!'orig' and 'dest' basis are the same: no transformation will be performed")
                if output == "v":
                    return v
                elif output == "R" :
                    return np.eye(3)
                elif output == "both" :
                    return v, np.eye(3)
                else :
                    raise ValueError("wrong output flag")
            
            M = [None,None]
            for n,name in enumerate([orig,dest]):
                if name == "lv":
                    M[n] = self._lv2cart_M()
                elif name == "rlv":
                    M[n] = self._rlv2cart_M()
                elif name == "cart" :
                    M[n] = np.eye(3)
                else :
                    raise ValueError("Something wrong here")
            
            # I use the cartesian coordinates as an intermediate step
            R = inv(M[1]) @ M[0]
            if output == "v":
                return R @ v
            elif output == "R" :
                return R
            elif output == "both" :
                return R @ v, R
            else :
                raise ValueError("wrong output flag")

        # matrix
        elif M is not None:
            valids = ["lv","rlv","cart"]
            valids = [ [i,j] for j in valids for i in valids ]
            check_value(orig,valids)
            check_value(dest,valids)

            A = self.change_basis(orig=orig[0],dest=dest[0],output="R")
            B = self.change_basis(orig=dest[1],dest=orig[1],output="R")

            #if output == "v":
            return A @ M @ B
            # elif output == "R" :
            #     return A, B
            # elif output == "both" :
            #     return A @ M @ B, A, B
            # else :
            #     raise ValueError("wrong output flag")

        #else :
        #    raise ValueError("Something wrong here")
