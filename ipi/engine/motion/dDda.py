
__all__ = ["dDdaTensorCalculator"]

import numpy as np
from numpy.linalg import norm


from ipi.engine.motion import Motion
from ipi.utils.depend import *
from ipi.utils.softexit import softexit
from ipi import ipi_global_settings


class dDdaTensorCalculator(Motion):

    """dP/da tensors calculator."""

    def __init__(
        self,
        mode="fd",
        pos_shift=0.01,
        matrix=np.zeros((6,3), float),
        prefix="",
        # asr="none",
    ):
        """Initialises dDdaTensorCalculator.
        Args:
        fixcom  : An optional boolean which decides whether the centre of mass
                  motion will be constrained or not. Defaults to False.
        matrix : A 3Nx3 array that stores the dynamic matrix.
        refdynmatrix : A 3Nx3N array that stores the refined dynamic matrix.
        """

        if mode != "fd" :
            raise ValueError(mode, "mode is not implemented in dDda calculator (the only allowed one is 'fd')")
        
        super(dDdaTensorCalculator, self).__init__(fixcom=False, fixatoms=None)

        self.mode = mode
        self.phcalc = FDdDdaTensorCalculator()

        self.deltax = pos_shift
        self.matrix = matrix.copy()
        #self.Ecorrection = np.zeros(0, float)
        #self.Icorrection = np.zeros(0, float)
        #self.Tcorrection = np.zeros(0, float)

        self.prefix = prefix
        # self.asr = asr

        # if self.prefix == "":
        #     self.prefix = "dDda"

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):

        super(dDdaTensorCalculator, self).bind(ens, beads, nm, cell, bforce, prng, omaker)

        # Raises error for nbeads not equal to 1.
        if self.beads.nbeads > 1:
            raise ValueError(
                "Calculation not possible for number of beads greater than one."
            )

        self.phcalc.bind(self)

    def step(self, step=None):
        """Executes one step of dDda computation."""
        if step < 6:
            self.phcalc.step(step)
        else:
            # self.phcalc.transform()
            # self.apply_asr()
            self.printall()
            softexit.trigger(
                status="success",
                message="dDda tensors have been calculated. Exiting simulation",
            )

    def printall(self):
        """Prints matrices to file"""

        M = self.matrix
        fmt = ipi_global_settings["floatformat"] 
        file = "{:s}.dDda.txt".format(self.prefix) if len(self.prefix) > 0 else "dDda.txt".format(self.prefix) 
        np.savetxt(file,M,delimiter=" ",fmt=fmt)

        return 

    # def apply_asr(self):
    #     """
    #     Removes the translations and/or rotations depending on the asr mode.
    #     """

    #     #self.Ecorrection = self.Ematrix.sum(axis=0)/self.beads.natoms
    #     #self.Icorrection = self.Imatrix.sum(axis=0)/self.beads.natoms
    #     #self.Tcorrection = self.matrix.sum(axis=0)/self.beads.natoms

    #     # if self.asr == "lin" :            
    #     #     self.Ematrix -= self.Ecorrection            
    #     #     self.Imatrix -= self.Icorrection            
    #     #     self.matrix -= self.Tcorrection

    #     if self.asr == "none" :
    #         return 
    #     else :
    #         raise ValueError("'{:s}' sum rule not implemented yet")

    #     # We should add the Rotational Sum Rule(s)

class FDdDdaTensorCalculator(dobject):

    """dP/da tensors calculator using finite differences."""

    #
    # author: Elia Stocco
    # e-mail: stocco@fhi-berlin.mpg.de
    #

    def __init__(self):
        pass

    def bind(self, dm):
        """Reference all the variables for simpler access."""
        
        #super(FDdDdaTensorCalculator, self).bind(dm)
        self.dm = dm
        self.h_original = dstrip(self.dm.cell.h).copy().flatten()
        self.ih_original = dstrip(self.dm.cell.ih).copy()
        self.pos_original  = np.asarray(dstrip(self.dm.beads.q[0]).copy()) 
        self.frac_original = (self.ih_original @ self.pos_original.reshape((-1,3)).T).T

        #print(type(self.dm.ensemble.ElecPol))
        #dd(self.dm.ensemble).ElecPol.add_dependency(dd(self.dm.dbeads).q)

        def check_dimension(M):
            if M.size != ( 3 * 6 ):
                if M.size == 0:
                    M = np.full((6, 3),np.nan,dtype=float)
                else:
                    raise ValueError("dipole matrix constant matrix size does not match system size")
            else:
                M = M.reshape(((6, 3 )))
            return M

        self.dm.matrix = check_dimension(self.dm.matrix)

        return
    
    def new_geo(self,dev):

        # modify the cell parameters
        h = (self.h_original + dev).reshape((3,3))
        self.dm.cell.h.set(h)

        # fix fractional coordinates, modify the cartesian ones
        q = ( self.dm.cell.h @ self.frac_original.T ).T.flatten()
        self.dm.beads.q.set( q )

        pass

    def step(self, step=None):
        """Computes one row of the dDda tensors"""

        # initializes the finite deviation
        dev = np.zeros(9, float)
        
        # displacement of lattice vectors
        #
        #   step        indices 
        # | 0 1 2 |    | 0 1 2 |
        # | / 3 4 |    | 3 4 5 |
        # | / / 5 |    | 6 7 8 |
        #
        #
        get_i = { "0":0,\
                  "1":1,\
                  "2":2,\
                  "3":4,\
                  "4":5,\
                  "5":8 }
        dev[get_i[str(step)]] = self.dm.deltax

        # displaces kth d.o.f by delta
        self.new_geo(dev)

        # ES: FIX HERE
        #Eplus = np.asarray(dstrip(self.dm.ensemble.ElecPol).copy())
        #Iplus = np.asarray(dstrip(self.dm.ensemble.IonsPol).copy())
        Dplus = np.asarray(dstrip(self.dm.ensemble.eda.dipole).copy())

        # displaces kth d.o.f by -delta.
        self.new_geo(-dev)

        # ES: FIX HERE
        #Eminus = np.asarray(dstrip(self.dm.ensemble.ElecPol).copy())
        #Iminus = np.asarray(dstrip(self.dm.ensemble.IonsPol).copy())
        Dminus = np.asarray(dstrip(self.dm.ensemble.eda.dipole).copy())

        Delta_a = 2 * self.dm.deltax
        factor = 1 # self.dm.ensemble.cell.V / Constants.e
        #self.dm.Ematrix[step] = factor * ( Eplus - Eminus ) / Delta_a
        #self.dm.Imatrix[step] = factor * ( Iplus - Iminus ) / Delta_a
        self.dm.matrix[step] = factor * ( Dplus - Dminus ) / Delta_a

        return

    # def transform(self):

    #     # reshape
    #     #self.dm.Ematrix = self.dm.Ematrix.reshape((self.dm.beads.natoms,3,3))
    #     #self.dm.Imatrix = self.dm.Imatrix.reshape((self.dm.beads.natoms,3,3))
    #     #self.dm.matrix = self.dm.matrix.reshape((self.dm.beads.natoms,3,3))

    #     return
        