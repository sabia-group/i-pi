
__all__ = ["BECTensorsCalculator"]

import numpy as np
from numpy.linalg import norm


from ipi.engine.motion import Motion
from ipi.utils.depend import *
from ipi.utils.softexit import softexit
from ipi.utils.messages import verbosity, info, warning
from ipi.utils.units import Constants


class BECTensorsCalculator(Motion):

    """BEC tensors calculator using finite difference."""

    def __init__(
        self,
        atoms=["all"],
        # mode="fd",
        pos_shift=0.001,
        bec=np.zeros(0, float),
        prefix="",
        asr="none",
    ):
        """Initialises BECTensorsCalculator.
        Args:
        fixcom  : An optional boolean which decides whether the centre of mass
                  motion will be constrained or not. Defaults to False.
        bec : A 3Nx3 array that stores the dynamic matrix.
        refdynmatrix : A 3Nx3N array that stores the refined dynamic matrix.
        """

        # if fixcom is False :
        #     raise ValueError("fixcom=False is not implemented in BEC calculator")        
        # if len(fixatoms) > 0 :
        #     raise ValueError("fixatoms is not implemented in BEC calculator")
        # if mode != "fd" :
        #     raise ValueError(mode, "mode is not implemented in BEC calculator (the only allowed one is 'fd')")
        
        super(BECTensorsCalculator, self).__init__(fixcom=False, fixatoms=None)

        # self.mode = mode
        self.phcalc = FDBECTensorsCalculator()

        self.deltax = pos_shift
        self.bec = bec.copy()
        self.correction = np.zeros(0, float)

        self.prefix = prefix
        self.asr = asr
        self.atoms = atoms

        if self.prefix == "":
            self.prefix = "BEC"

        #self.fixdof = np.full(self.beads.natoms,None)
        # if self.atoms in ["all",]:
        #     fixdof = np.concatenate((3 * fixatoms, 3 * fixatoms + 1, 3 * fixatoms + 2))
        #     self.fixdof = np.sort(fixdof)
        # else:
        #     self.fixdof = np.array([])

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):

        super(BECTensorsCalculator, self).bind(ens, beads, nm, cell, bforce, prng, omaker)

        # Raises error for nbeads not equal to 1.
        if self.beads.nbeads > 1:
            raise ValueError(
                "Calculation not possible for number of beads greater than one."
            )

        #self.ism = 1 / np.sqrt(dstrip(self.beads.m3[-1]))
        #self.m = dstrip(self.beads.m)
        self.tomove = np.full(3*self.beads[0].natoms,False)
        if len(self.atoms) == 0 :
            self.tomove = np.full(3*self.beads[0].natoms,False)
        elif len(self.atoms) == 1 and self.atoms[0].lower() ==  "all":
            self.tomove = np.full(3*self.beads[0].natoms,True)
        else:
            for i in self.atoms:
                if i.isdigit():
                    i = int(i)
                    self.tomove[i*3:(i+1)*3] = True
                else:
                    if i not in list(self.beads[0].names):
                        raise ValueError("wrong input")
                    else:
                        index = list(self.beads[0].names).index(i)
                    if not hasattr(index,'__len__'):
                        index = [index]
                    for j in index:
                        j = int(j)
                        self.tomove[j*3:(j+1)*3] = True
                

        self.phcalc.bind(self)

        #self.dbeads = self.beads.copy()
        #self.dcell = self.cell.copy()
        #dd(self.ensemble).ElecPol.add_dependency(dd(self.dbeads).q)
        #pass

    def step(self, step=None):
        """Executes one step of BEC computation."""
        if step < 3 * self.beads.natoms:
            self.phcalc.step(step)
        else:
            self.phcalc.transform()
            self.apply_asr()
            self.printall()
            softexit.trigger(
                status="success",
                message="BEC tensors have been calculated. Exiting simulation",
            )

    def printall(self):
        """Prints matrices to file"""

        file = "{:s}.txt".format(self.prefix,)
        np.savetxt(file,self.bec.reshape((-1,3)),delimiter=" ",fmt="%15.10f")

        if self.correction.shape != (0,) :
            file = "{:s}.correction.txt".format(self.prefix)
            np.savetxt(file,self.correction.reshape((-1,3)),delimiter=" ",fmt="%15.10f")

        return 

    def apply_asr(self):
        """
        Removes the translations and/or rotations depending on the asr mode.
        """

        #
        # Translational Sum Rule :
        # \sum_I Z^I_ij = 0 
        #
        # This means that the translation of all the ions does not lead to any change in the polarization.
        #
        # So we compute this sum, which should be zero, but it is not due to "numerical noise",
        # and then we subtract this amount (divided by the number of atoms) to each BEC.
        #
        # Pay attention that in this case self.bec has already the shape (natoms,3,3)
        # and self.correction has the shape (3,3) 
        #

        if self.asr == "lin" :                       
            
            if np.all(self.tomove):
                warning("Sum Ruls can not be applied because some dofs were kept fixed")

            self.correction = self.bec.sum(axis=0)/self.beads.natoms
            self.bec -= self.correction

        elif self.asr == "none" :
            return 

        # We should add the Rotational Sum Rule(s)

class FDBECTensorsCalculator(dobject):

    """Finite difference BEC tensors evaluator."""

    #
    # author: Elia Stocco
    # e-mail: stocco@fhi-berlin.mpg.de
    #

    def __init__(self):
        pass

    def bind(self, dm):
        """Reference all the variables for simpler access."""
        
        self.dm = dm
        self.original = np.asarray(dstrip(self.dm.beads.q[0]).copy()) 
        self.atoms = dm.atoms


        def check_dimension(M):
            if M.size != ( 3 * self.dm.beads.q.size ):
                if M.size == 0:
                    M = np.full((self.dm.beads.q.size, 3 ),np.nan,dtype=float)
                else:
                    raise ValueError("matrix size does not match system size")
            else:
                M = M.reshape(((self.dm.beads.q.size, 3 )))
            return M

        # Initialises a 3*number of atoms X 3*number of atoms dynamic matrix.
        self.dm.bec = check_dimension(self.dm.bec)

        return

    def step(self, step=None):
        """Computes one row of the BEC tensors"""

        if self.dm.tomove[step]:

            # initializes the finite deviation
            dev = np.zeros(3 * self.dm.beads.natoms, float)
            
            # displacement in cartesian components
            dev[step] = self.dm.deltax

            # displaces kth d.o.f by delta.
            self.dm.beads.q.set(self.original + dev)
            Tplus = np.asarray(dstrip(self.dm.ensemble.eda.polarization).copy())

            # displaces kth d.o.f by -delta.
            self.dm.beads.q.set(self.original - dev)
            Tminus = np.asarray(dstrip(self.dm.ensemble.eda.polarization).copy())

            #
            # the following line computes a component of a BEC tensor
            # Z^I_ij = Omega/e (delta P_i / delta R_j_I)
            # I : atom index
            # i : polarization index, i.e. P_i is the component along 
            #     the i-th reciprocal lattice vectors
            # j : displacement index, i.e. R_j is the component of the 
            #     displacement (of the atom I-th) along the j-th lattice vectors
            # Omega : primitive unit cell volum
            # 
            # Pay attention: i-PI asks the driver to give the polarization expressed 
            # w.r.t. the (normalized) reciprocal lattice vectors.
            # This choide stems from the fact that in DFT codes the polarization 
            # is usually computed along these directions.
            #
            # Then i-PI doesn't ask the cartesian components to the driver, 
            # in order to avoid to implement some (bug prone) code in each driver.
            #
            # Moreover, the displacements are performed along the lattice vectors, and not along the cartesian axis!
            # In this way, the output quantities (BEC) are indepenedent on the lattice orientation.
            # If you need to change the lattice orientation for any reason, the BEC tensors expressed in this way do not change.
            # so you can jusy copy and paste
            #
            # Have a nice day :)
            #

            self.dm.bec[step] = ( self.dm.ensemble.cell.V / Constants.e ) * ( Tplus - Tminus ) / ( 2 * self.dm.deltax )

        else:
            info(" We have skipped the dof # {}.".format(step), verbosity.low)

        return

    def transform(self):

        # reshape
        self.dm.bec = self.dm.bec.reshape((self.dm.beads.natoms,3,3))

        # transpose
        # for i in range(self.dm.beads.natoms):
        #     self.dm.bec[i] = self.dm.bec[i].T

        return
        
