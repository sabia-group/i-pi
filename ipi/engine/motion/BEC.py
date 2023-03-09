
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
        fixcom=True,
        fixatoms=None,
        mode="fd",
        #energy_shift=0.0,
        pos_shift=0.001,
        #output_shift=0.000,
        polmat=np.zeros(0, float),
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

        if fixcom is False :
            raise ValueError("fixcom=False is not implemented in BEC calculator")        
        if len(fixatoms) > 0 :
            raise ValueError("fixatoms is not implemented in BEC calculator")
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
        self.polmatrix = polmat
        self.correction = np.zeros(0, float)

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
            #self.phcalc.transform()
            self.polmatrix = self.polmatrix.reshape((self.beads.natoms,3,3))
            self.apply_asr()
            self.printall()
            softexit.trigger(
                status="success",
                message="BEC tensors have been calculated. Exiting simulation",
            )

    def printall(self):
        """Prints out diagnostics for a given dynamical matrix."""

        np.savetxt("correction.txt",self.correction)
        np.savetxt("BEC.txt",self.polmatrix)

        return 

        # dmatx = dmatx + np.eye(len(dmatx)) * deltaw
        # if deltaw != 0.0:
        #     wstr = " !! Shifted by %e !!" % (deltaw)
        # else:
        #     wstr = ""

        # # get active arrays:
        # activedof = 3 * self.beads.natoms - fixdof.size
        # if fixdof.size > 0:
        #     mask = np.delete(np.arange(3 * self.beads.natoms), fixdof)
        # else:
        #     mask = np.arange(3 * self.beads.natoms)

        # dmatx_full = dmatx.copy()
        # ism_full = self.ism.copy()
        # dmatx = dmatx[mask][:, mask]
        # ism = self.ism[mask]

        # # prints out the dynamical matrix
        # outfile = self.output_maker.get_output(self.prefix + ".dynmat", "w")
        # outfile.write("# Dynamical matrix (atomic units)" + wstr + "\n")
        # for i in range(activedof):
        #     outfile.write(" ".join(map(str, dmatx[i])) + "\n")
        # outfile.close_stream()

        # # prints out the Hessian for the activedof
        # outfile = self.output_maker.get_output(self.prefix + ".hess", "w")
        # outfile.write("# Hessian matrix (atomic units)" + wstr + "\n")
        # for i in range(activedof):
        #     outfile.write(" ".join(map(str, dmatx[i] / (ism[i] * ism))) + "\n")
        # outfile.close_stream()

        # # prints out the full Hessian (with all zeros)
        # outfile = self.output_maker.get_output(self.prefix + "_full.hess", "w")
        # outfile.write("# Hessian matrix (atomic units)" + wstr + "\n")
        # for i in range(3 * self.beads.natoms):
        #     outfile.write(
        #         " ".join(map(str, dmatx_full[i] / (ism_full[i] * ism_full))) + "\n"
        #     )
        # outfile.close_stream()

        # eigsys = np.linalg.eigh(dmatx)

        # # prints eigenvalues
        # outfile = self.output_maker.get_output(self.prefix + ".eigval", "w")
        # outfile.write("# Eigenvalues (atomic units)" + wstr + "\n")
        # outfile.write("\n".join(map(str, eigsys[0])))
        # outfile.close_stream()

        # # prints eigenvectors
        # outfile = self.output_maker.get_output(self.prefix + ".eigvec", "w")
        # outfile.write("# Eigenvector  matrix (normalized)" + "\n")
        # for i in range(activedof):
        #     outfile.write(" ".join(map(str, eigsys[1][i])) + "\n")
        # outfile.close_stream()

        # # prints eigenmodes
        # eigmode = 1.0 * eigsys[1]
        # for i in range(activedof):
        #     eigmode[i] *= ism[i]
        # for i in range(activedof):
        #     eigmode[:, i] /= np.sqrt(np.dot(eigmode[:, i], eigmode[:, i]))
        # outfile = self.output_maker.get_output(self.prefix + ".mode", "w")

        # outfile.write("# Phonon modes (mass-scaled)" + "\n")
        # for i in range(activedof):
        #     outfile.write(" ".join(map(str, eigmode[i])) + "\n")
        # outfile.close_stream()

    def apply_asr(self):
        """
        Removes the translations and/or rotations depending on the asr mode.
        """
        self.correction = self.polmatrix.sum(axis=0)/self.beads.natoms
        self.polmatrix -= self.correction
        # if self.asr == "none":
        #     return dm

        # if self.asr == "crystal":
        #     # Computes the centre of mass.
        #     com = (
        #         np.dot(
        #             np.transpose(self.beads.q.reshape((self.beads.natoms, 3))), self.m
        #         )
        #         / self.m.sum()
        #     )
        #     qminuscom = self.beads.q.reshape((self.beads.natoms, 3)) - com
        #     # Computes the moment of inertia tensor.
        #     moi = np.zeros((3, 3), float)
        #     for k in range(self.beads.natoms):
        #         moi -= (
        #             np.dot(
        #                 np.cross(qminuscom[k], np.identity(3)),
        #                 np.cross(qminuscom[k], np.identity(3)),
        #             )
        #             * self.m[k]
        #         )

        #     U = (np.linalg.eig(moi))[1]
        #     R = np.dot(qminuscom, U)
        #     D = np.zeros((3, 3 * self.beads.natoms), float)

        #     # Computes the vectors along rotations.
        #     D[0] = np.tile([1, 0, 0], self.beads.natoms) / self.ism
        #     D[1] = np.tile([0, 1, 0], self.beads.natoms) / self.ism
        #     D[2] = np.tile([0, 0, 1], self.beads.natoms) / self.ism

        #     # Computes unit vecs.
        #     for k in range(3):
        #         D[k] = D[k] / np.linalg.norm(D[k])

        #     # Computes the transformation matrix.
        #     transfmatrix = np.eye(3 * self.beads.natoms) - np.dot(D.T, D)
        #     r = np.dot(transfmatrix.T, np.dot(dm, transfmatrix))
        #     return r

        # elif self.asr == "poly":
        #     # Computes the centre of mass.
        #     com = (
        #         np.dot(
        #             np.transpose(self.beads.q.reshape((self.beads.natoms, 3))), self.m
        #         )
        #         / self.m.sum()
        #     )
        #     qminuscom = self.beads.q.reshape((self.beads.natoms, 3)) - com
        #     # Computes the moment of inertia tensor.
        #     moi = np.zeros((3, 3), float)
        #     for k in range(self.beads.natoms):
        #         moi -= (
        #             np.dot(
        #                 np.cross(qminuscom[k], np.identity(3)),
        #                 np.cross(qminuscom[k], np.identity(3)),
        #             )
        #             * self.m[k]
        #         )

        #     U = (np.linalg.eig(moi))[1]
        #     R = np.dot(qminuscom, U)
        #     D = np.zeros((6, 3 * self.beads.natoms), float)

        #     # Computes the vectors along translations and rotations.
        #     D[0] = np.tile([1, 0, 0], self.beads.natoms) / self.ism
        #     D[1] = np.tile([0, 1, 0], self.beads.natoms) / self.ism
        #     D[2] = np.tile([0, 0, 1], self.beads.natoms) / self.ism
        #     for i in range(3 * self.beads.natoms):
        #         iatom = i // 3
        #         idof = np.mod(i, 3)
        #         D[3, i] = (
        #             R[iatom, 1] * U[idof, 2] - R[iatom, 2] * U[idof, 1]
        #         ) / self.ism[i]
        #         D[4, i] = (
        #             R[iatom, 2] * U[idof, 0] - R[iatom, 0] * U[idof, 2]
        #         ) / self.ism[i]
        #         D[5, i] = (
        #             R[iatom, 0] * U[idof, 1] - R[iatom, 1] * U[idof, 0]
        #         ) / self.ism[i]

        #     # Computes unit vecs.
        #     for k in range(6):
        #         D[k] = D[k] / np.linalg.norm(D[k])

        #     # Computes the transformation matrix.
        #     transfmatrix = np.eye(3 * self.beads.natoms) - np.dot(D.T, D)
        #     r = np.dot(transfmatrix.T, np.dot(dm, transfmatrix))
        #     return r


class DummyBECTensorsCalculator(dobject):

    """No-op PhononCalculator"""

    def __init__(self):
        pass

    def bind(self, dm):
        """Reference all the variables for simpler access."""
        self.dm = dm
        self.original = np.asarray(dstrip(self.dm.beads.q[0]).copy())

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""
        pass

    def transform(self):
        """Dummy transformation step which does nothing."""
        pass


class FDBECTensorsCalculator(DummyBECTensorsCalculator):

    """Finite difference BEC tensors evaluator."""

    #
    # author: Elia Stocco
    # email: stocco@fhi-berlin.mpg.de
    #

    def bind(self, dm):
        """Reference all the variables for simpler access."""
        super(FDBECTensorsCalculator, self).bind(dm)
        #print(type(self.dm.ensemble.ElecPol))
        #dd(self.dm.ensemble).ElecPol.add_dependency(dd(self.dm.dbeads).q)

        # Initialises a 3*number of atoms X 3*number of atoms dynamic matrix.
        if self.dm.polmatrix.size != ( 3 * self.dm.beads.q.size ):
            if self.dm.polmatrix.size == 0:
                self.dm.polmatrix = np.zeros(
                    (self.dm.beads.q.size, 3 ), float
                )
            else:
                raise ValueError(
                    "Polarization matrix constant matrix size does not match system size"
                )
        else:
            self.dm.polmatrix = self.dm.polmatrix.reshape(
                ((self.dm.beads.q.size, 3 ))
            )

    def step(self, step=None):
        """Computes one row of the BEC tensors"""

        # original coordinates
        #original = np.asarray(dstrip(self.dm.beads.q[0]).copy())

        # initializes the finite deviation
        dev = np.zeros(3 * self.dm.beads.natoms, float)
        
        # displacement along the lattice vectors
        dev[step] = self.dm.deltax
        #print(step - step%3,":",step - step%3 + 3)
        displ = dev[step - step%3 : step - step%3 + 3]

        # displacement in cartesian coordinates
        cart = self.dm.ensemble.cell.lv2cart(displ)
        #print("norm:",norm(cart))
        dev[step - step%3 : step - step%3 + 3] = cart

        # displaces kth d.o.f by delta.
        self.dm.beads.q.set(self.original + dev)
        plus = np.asarray(self.dm.ensemble.ElecPol)#.copy()

        # displaces kth d.o.f by -delta.
        self.dm.beads.q.set(self.original - dev)
        minus = np.asarray(self.dm.ensemble.ElecPol)#.copy()

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

        self.dm.polmatrix[step] = ( self.dm.ensemble.cell.V / Constants.e ) * ( plus - minus) / ( 2 * self.dm.deltax )

        # else:
        #     info(" We have skipped the dof # {}.".format(step), verbosity.low)

    # def transform(self):
    #     dm = self.dm.polmatrix.copy()
    #     rdm = self.dm.polmatrix.copy()
    #     self.dm.polmatrix = 0.50 * (dm + dm.T)
    #     # self.dm.refdynmatrix = 0.50 * (rdm + rdm.T)

