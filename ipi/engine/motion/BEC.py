
__all__ = ["BECTensorsCalculator"]

import numpy as np
from numpy.linalg import norm


from ipi.engine.motion import Motion
from ipi.utils.depend import *
from ipi.utils.softexit import softexit
from ipi.utils.messages import verbosity, info
from ipi.utils.units import Constants


class BECTensorsCalculator(Motion):

    """BEC tensors calculator using finite difference."""

    def __init__(
        self,
        #fixcom=True,
        #fixatoms=None,
        mode="fd",
        #energy_shift=0.0,
        pos_shift=0.001,
        #output_shift=0.000,
        Epolmatrix=np.zeros(0, float),
        Ipolmatrix=np.zeros(0, float),
        Tpolmatrix=np.zeros(0, float),
        #refdynmat=np.zeros(0, float),
        prefix="",
        asr="none",
    ):
        """Initialises BECTensorsCalculator.
        Args:
        fixcom  : An optional boolean which decides whether the centre of mass
                  motion will be constrained or not. Defaults to False.
        polmatrix : A 3Nx3 array that stores the dynamic matrix.
        refdynmatrix : A 3Nx3N array that stores the refined dynamic matrix.
        """

        # if fixcom is False :
        #     raise ValueError("fixcom=False is not implemented in BEC calculator")        
        # if len(fixatoms) > 0 :
        #     raise ValueError("fixatoms is not implemented in BEC calculator")
        if mode != "fd" :
            raise ValueError(mode, "mode is not implemented in BEC calculator (the only allowed one is 'fd')")
        
        super(BECTensorsCalculator, self).__init__(fixcom=False, fixatoms=None)

        # Finite difference option.
        self.mode = mode
        #if self.mode == "fd":
        self.phcalc = FDBECTensorsCalculator()
        # elif self.mode == "nmfd":
        #     self.phcalc = NMFDPhononCalculator()
        # elif self.mode == "enmfd":
        #     self.phcalc = ENMFDPhononCalculator()

        #self.deltaw = output_shift
        self.deltax = pos_shift
        #self.deltae = energy_shift
        self.Epolmatrix = Epolmatrix.copy()
        self.Ipolmatrix = Ipolmatrix.copy()
        self.Tpolmatrix = Tpolmatrix.copy()
        self.Ecorrection = np.zeros(0, float)
        self.Icorrection = np.zeros(0, float)
        self.Tcorrection = np.zeros(0, float)

        # self.frefine = False
        # self.U = None
        # self.V = None
        self.prefix = prefix
        self.asr = asr

        if self.prefix == "":
            self.prefix = "BEC"

        # if len(fixatoms) > 0:
        #     fixdof = np.concatenate((3 * fixatoms, 3 * fixatoms + 1, 3 * fixatoms + 2))
        #     self.fixdof = np.sort(fixdof)
        #     # if self.mode == "enmfd" or self.mode == "nmfd":
        #     #     raise ValueError("Fixatoms is not implemented for the selected mode.")
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

        for contr,M in zip(["electrons","ions","total"],[self.Epolmatrix,self.Ipolmatrix,self.Tpolmatrix]):
            file = "{:s}.BEC.{:s}.txt".format(self.prefix,contr)
            header = "BEC for {:s} polarization.\n".format(contr)+\
                "The polarization (row indices of the BEC tensors) are expressed w.r.t. the (normalized) lattice vectors.\n"+\
                "The displacements (column indices of the BEC tensors) are expressed (and have been performed) along the (normalized) lattice vectors.\n"+\
                "The BEC tensors of each ions are printed consecutively."
            np.savetxt(file,M.reshape((-1,3)),delimiter=" ",fmt="%15.10f",header=header)

        for contr,M in zip(["electrons","ions","total"],[self.Ecorrection,self.Icorrection,self.Tcorrection]):
            file = "{:s}.BEC.correction.{:s}.txt".format(self.prefix,contr)
            header = "correction to the BEC tensors for {:s} computed imposing the Translational Sum Rule".format(contr)
            np.savetxt(file,M.reshape((-1,3)),delimiter=" ",fmt="%15.10f",header=header)

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
        # Pay attention that in this case self.polmatrix has already the shape (natoms,3,3)
        # and self.correction has the shape (3,3) 
        #

        self.Ecorrection = self.Epolmatrix.sum(axis=0)/self.beads.natoms
        self.Icorrection = self.Ipolmatrix.sum(axis=0)/self.beads.natoms
        self.Tcorrection = self.Tpolmatrix.sum(axis=0)/self.beads.natoms

        if self.asr == "lin" :            
            self.Epolmatrix -= self.Ecorrection            
            self.Ipolmatrix -= self.Icorrection            
            self.Tpolmatrix -= self.Tcorrection

        elif self.asr == "nonr" :
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
        
        #super(FDBECTensorsCalculator, self).bind(dm)
        self.dm = dm
        self.original = np.asarray(dstrip(self.dm.beads.q[0]).copy()) 

        #print(type(self.dm.ensemble.ElecPol))
        #dd(self.dm.ensemble).ElecPol.add_dependency(dd(self.dm.dbeads).q)

        def check_dimension(M,name):
            if M.size != ( 3 * self.dm.beads.q.size ):
                if M.size == 0:
                    M = np.zeros((self.dm.beads.q.size, 3 ),float)
                else:
                    raise ValueError("{:s} polarization matrix constant matrix size does not match system size".format(name))
            else:
                M = M.reshape(((self.dm.beads.q.size, 3 )))
            return M

        # Initialises a 3*number of atoms X 3*number of atoms dynamic matrix.
        self.dm.Epolmatrix = check_dimension(self.dm.Epolmatrix,"Electronic")
        self.dm.Ipolmatrix = check_dimension(self.dm.Ipolmatrix,"Ionic")
        self.dm.Tpolmatrix = check_dimension(self.dm.Tpolmatrix,"Total")

        return

    def step(self, step=None):
        """Computes one row of the BEC tensors"""

        # initializes the finite deviation
        dev = np.zeros(3 * self.dm.beads.natoms, float)
        
        # displacement along the lattice vectors
        dev[step] = self.dm.deltax

        # #print(step - step%3,":",step - step%3 + 3)
        # displ = dev[step - step%3 : step - step%3 + 3]

        # # displacement in cartesian coordinates
        # cart = self.dm.ensemble.cell.change_basis(v=displ,orig="lv",dest="cart")
    
        # #print("norm:",norm(cart))
        # dev[step - step%3 : step - step%3 + 3] = cart

        # displaces kth d.o.f by delta.
        self.dm.beads.q.set(self.original + dev)
        Eplus = np.asarray(dstrip(self.dm.ensemble.ElecPol).copy())
        Iplus = np.asarray(dstrip(self.dm.ensemble.IonsPol).copy())
        Tplus = np.asarray(dstrip(self.dm.ensemble.TotalPol).copy())

        # displaces kth d.o.f by -delta.
        self.dm.beads.q.set(self.original - dev)
        Eminus = np.asarray(dstrip(self.dm.ensemble.ElecPol).copy())
        Iminus = np.asarray(dstrip(self.dm.ensemble.IonsPol).copy())
        Tminus = np.asarray(dstrip(self.dm.ensemble.TotalPol).copy())

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

        self.dm.Epolmatrix[step] = ( self.dm.ensemble.cell.V / Constants.e ) * ( Eplus - Eminus ) / ( 2 * self.dm.deltax )
        self.dm.Ipolmatrix[step] = ( self.dm.ensemble.cell.V / Constants.e ) * ( Iplus - Iminus ) / ( 2 * self.dm.deltax )
        self.dm.Tpolmatrix[step] = ( self.dm.ensemble.cell.V / Constants.e ) * ( Tplus - Tminus ) / ( 2 * self.dm.deltax )

        return

    def transform(self):

        # reshape
        self.dm.Epolmatrix = self.dm.Epolmatrix.reshape((self.dm.beads.natoms,3,3))
        self.dm.Ipolmatrix = self.dm.Ipolmatrix.reshape((self.dm.beads.natoms,3,3))
        self.dm.Tpolmatrix = self.dm.Tpolmatrix.reshape((self.dm.beads.natoms,3,3))

        # transpose
        for i in range(self.dm.beads.natoms):
            self.dm.Epolmatrix[i] = self.dm.Epolmatrix[i].T
            self.dm.Ipolmatrix[i] = self.dm.Ipolmatrix[i].T
            self.dm.Tpolmatrix[i] = self.dm.Tpolmatrix[i].T

        # change of basis from (rlv,lv) to (lv,lv)
        for i in range(self.dm.beads.natoms):
            self.dm.Epolmatrix[i] = self.dm.cell.change_basis(M=self.dm.Epolmatrix[i],orig=("rlv","lv"),dest=("lv","lv"),verbose=False)
            self.dm.Ipolmatrix[i] = self.dm.cell.change_basis(M=self.dm.Ipolmatrix[i],orig=("rlv","lv"),dest=("lv","lv"),verbose=False)
            self.dm.Tpolmatrix[i] = self.dm.cell.change_basis(M=self.dm.Tpolmatrix[i],orig=("rlv","lv"),dest=("lv","lv"),verbose=False)

        return
        

