
__all__ = ["BECTensorsCalculator"]

import numpy as np


from ipi.engine.motion import Motion
from ipi.utils.depend import *
from ipi.utils.softexit import softexit
from ipi.utils.messages import verbosity, info


class BECTensorsCalculator(Motion):

    """BEC tensors calculator using finite difference."""

    def __init__(
        self,
        fixcom=False,
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

        super(BECTensorsCalculator, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

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
        #self.refdynmatrix = refdynmat
        self.frefine = False
        self.U = None
        self.V = None
        self.prefix = prefix
        self.asr = asr

        if self.prefix == "":
            self.prefix = "BEC"

        if len(fixatoms) > 0:
            fixdof = np.concatenate((3 * fixatoms, 3 * fixatoms + 1, 3 * fixatoms + 2))
            self.fixdof = np.sort(fixdof)
            # if self.mode == "enmfd" or self.mode == "nmfd":
            #     raise ValueError("Fixatoms is not implemented for the selected mode.")
        else:
            self.fixdof = np.array([])

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):

        super(BECTensorsCalculator, self).bind(ens, beads, nm, cell, bforce, prng, omaker)

        # Raises error for nbeads not equal to 1.
        if self.beads.nbeads > 1:
            raise ValueError(
                "Calculation not possible for number of beads greater than one."
            )

        self.ism = 1 / np.sqrt(dstrip(self.beads.m3[-1]))
        self.m = dstrip(self.beads.m)
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
            # self.refdynmatrix = self.apply_asr(self.refdynmatrix.copy())
            self.printall(self.prefix, self.polmatrix.copy(), fixdof=self.fixdof)
            softexit.trigger(
                status="success",
                message="Dynamic matrix is calculated. Exiting simulation",
            )

    def printall(self, prefix, dmatx, deltaw=0.0, fixdof=np.array([])):
        """Prints out diagnostics for a given dynamical matrix."""

        dmatx = dmatx + np.eye(len(dmatx)) * deltaw
        if deltaw != 0.0:
            wstr = " !! Shifted by %e !!" % (deltaw)
        else:
            wstr = ""

        # get active arrays:
        activedof = 3 * self.beads.natoms - fixdof.size
        if fixdof.size > 0:
            mask = np.delete(np.arange(3 * self.beads.natoms), fixdof)
        else:
            mask = np.arange(3 * self.beads.natoms)

        dmatx_full = dmatx.copy()
        ism_full = self.ism.copy()
        dmatx = dmatx[mask][:, mask]
        ism = self.ism[mask]

        # prints out the dynamical matrix
        outfile = self.output_maker.get_output(self.prefix + ".dynmat", "w")
        outfile.write("# Dynamical matrix (atomic units)" + wstr + "\n")
        for i in range(activedof):
            outfile.write(" ".join(map(str, dmatx[i])) + "\n")
        outfile.close_stream()

        # prints out the Hessian for the activedof
        outfile = self.output_maker.get_output(self.prefix + ".hess", "w")
        outfile.write("# Hessian matrix (atomic units)" + wstr + "\n")
        for i in range(activedof):
            outfile.write(" ".join(map(str, dmatx[i] / (ism[i] * ism))) + "\n")
        outfile.close_stream()

        # prints out the full Hessian (with all zeros)
        outfile = self.output_maker.get_output(self.prefix + "_full.hess", "w")
        outfile.write("# Hessian matrix (atomic units)" + wstr + "\n")
        for i in range(3 * self.beads.natoms):
            outfile.write(
                " ".join(map(str, dmatx_full[i] / (ism_full[i] * ism_full))) + "\n"
            )
        outfile.close_stream()

        eigsys = np.linalg.eigh(dmatx)

        # prints eigenvalues
        outfile = self.output_maker.get_output(self.prefix + ".eigval", "w")
        outfile.write("# Eigenvalues (atomic units)" + wstr + "\n")
        outfile.write("\n".join(map(str, eigsys[0])))
        outfile.close_stream()

        # prints eigenvectors
        outfile = self.output_maker.get_output(self.prefix + ".eigvec", "w")
        outfile.write("# Eigenvector  matrix (normalized)" + "\n")
        for i in range(activedof):
            outfile.write(" ".join(map(str, eigsys[1][i])) + "\n")
        outfile.close_stream()

        # prints eigenmodes
        eigmode = 1.0 * eigsys[1]
        for i in range(activedof):
            eigmode[i] *= ism[i]
        for i in range(activedof):
            eigmode[:, i] /= np.sqrt(np.dot(eigmode[:, i], eigmode[:, i]))
        outfile = self.output_maker.get_output(self.prefix + ".mode", "w")

        outfile.write("# Phonon modes (mass-scaled)" + "\n")
        for i in range(activedof):
            outfile.write(" ".join(map(str, eigmode[i])) + "\n")
        outfile.close_stream()

    def apply_asr(self, dm):
        """
        Removes the translations and/or rotations depending on the asr mode.
        """
        if self.asr == "none":
            return dm

        if self.asr == "crystal":
            # Computes the centre of mass.
            com = (
                np.dot(
                    np.transpose(self.beads.q.reshape((self.beads.natoms, 3))), self.m
                )
                / self.m.sum()
            )
            qminuscom = self.beads.q.reshape((self.beads.natoms, 3)) - com
            # Computes the moment of inertia tensor.
            moi = np.zeros((3, 3), float)
            for k in range(self.beads.natoms):
                moi -= (
                    np.dot(
                        np.cross(qminuscom[k], np.identity(3)),
                        np.cross(qminuscom[k], np.identity(3)),
                    )
                    * self.m[k]
                )

            U = (np.linalg.eig(moi))[1]
            R = np.dot(qminuscom, U)
            D = np.zeros((3, 3 * self.beads.natoms), float)

            # Computes the vectors along rotations.
            D[0] = np.tile([1, 0, 0], self.beads.natoms) / self.ism
            D[1] = np.tile([0, 1, 0], self.beads.natoms) / self.ism
            D[2] = np.tile([0, 0, 1], self.beads.natoms) / self.ism

            # Computes unit vecs.
            for k in range(3):
                D[k] = D[k] / np.linalg.norm(D[k])

            # Computes the transformation matrix.
            transfmatrix = np.eye(3 * self.beads.natoms) - np.dot(D.T, D)
            r = np.dot(transfmatrix.T, np.dot(dm, transfmatrix))
            return r

        elif self.asr == "poly":
            # Computes the centre of mass.
            com = (
                np.dot(
                    np.transpose(self.beads.q.reshape((self.beads.natoms, 3))), self.m
                )
                / self.m.sum()
            )
            qminuscom = self.beads.q.reshape((self.beads.natoms, 3)) - com
            # Computes the moment of inertia tensor.
            moi = np.zeros((3, 3), float)
            for k in range(self.beads.natoms):
                moi -= (
                    np.dot(
                        np.cross(qminuscom[k], np.identity(3)),
                        np.cross(qminuscom[k], np.identity(3)),
                    )
                    * self.m[k]
                )

            U = (np.linalg.eig(moi))[1]
            R = np.dot(qminuscom, U)
            D = np.zeros((6, 3 * self.beads.natoms), float)

            # Computes the vectors along translations and rotations.
            D[0] = np.tile([1, 0, 0], self.beads.natoms) / self.ism
            D[1] = np.tile([0, 1, 0], self.beads.natoms) / self.ism
            D[2] = np.tile([0, 0, 1], self.beads.natoms) / self.ism
            for i in range(3 * self.beads.natoms):
                iatom = i // 3
                idof = np.mod(i, 3)
                D[3, i] = (
                    R[iatom, 1] * U[idof, 2] - R[iatom, 2] * U[idof, 1]
                ) / self.ism[i]
                D[4, i] = (
                    R[iatom, 2] * U[idof, 0] - R[iatom, 0] * U[idof, 2]
                ) / self.ism[i]
                D[5, i] = (
                    R[iatom, 0] * U[idof, 1] - R[iatom, 1] * U[idof, 0]
                ) / self.ism[i]

            # Computes unit vecs.
            for k in range(6):
                D[k] = D[k] / np.linalg.norm(D[k])

            # Computes the transformation matrix.
            transfmatrix = np.eye(3 * self.beads.natoms) - np.dot(D.T, D)
            r = np.dot(transfmatrix.T, np.dot(dm, transfmatrix))
            return r


class DummyBECTensorsCalculator(dobject):

    """No-op PhononCalculator"""

    def __init__(self):
        pass

    def bind(self, dm):
        """Reference all the variables for simpler access."""
        self.dm = dm

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""
        pass

    def transform(self):
        """Dummy transformation step which does nothing."""
        pass


class FDBECTensorsCalculator(DummyBECTensorsCalculator):

    """Finite difference BEC tensors evaluator."""

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

        # # Initialises a 3*number of atoms X 3*number of atoms refined dynamic matrix.
        # if self.dm.refdynmatrix.size != (self.dm.beads.q.size * self.dm.beads.q.size):
        #     if self.dm.refdynmatrix.size == 0:
        #         self.dm.refdynmatrix = np.zeros(
        #             (self.dm.beads.q.size, self.dm.beads.q.size), float
        #         )
        #     else:
        #         raise ValueError(
        #             "Force constant matrix size does not match system size"
        #         )
        # else:
        #     self.dm.refdynmatrix = self.dm.refdynmatrix.reshape(
        #         ((self.dm.beads.q.size, self.dm.beads.q.size))
        #     )

    def step(self, step=None):
        """Computes one row of the dynamic matrix."""

        original = self.dm.beads.q

        if step not in self.dm.fixdof:
            # initializes the finite deviation
            dev = np.zeros(3 * self.dm.beads.natoms, float)
            dev[step] = self.dm.deltax
            #dd(self.dm.ensemble).ElecPol.add_dependency(dd(self.dm.dbeads).q)
            # displaces kth d.o.f by delta.
            # self.dm.beads.q._tainted[0] = True
            self.dm.beads.q = original + dev
            #self.dm.ensemble.ElecPol = np.asarray([1,2,3])
            #print(self.dm.ensemble._get_pol(what="elec"))
            #print(self.dm.ensemble.ElecPol)
            plus = -dstrip(self.dm.ensemble.ElecPol).copy()
            # displaces kth d.o.f by -delta.
            #self.dm.dbeads.q = self.dm.beads.q - dev
            self.dm.beads.q = original - dev
            # self.dm.beads.q._tainted[0] = True
            minus = -dstrip(self.dm.ensemble.ElecPol).copy()
            # computes a row of force-constant matrix
            dmrow = plus - minus
            #(
            #     (plus - minus) / (2 * self.dm.deltax) * self.dm.ism[step] * self.dm.ism
            # )
            self.dm.polmatrix[step] = dmrow
            # self.dm.refdynmatrix[step] = dmrow
        else:
            info(" We have skipped the dof # {}.".format(step), verbosity.low)

    def transform(self):
        dm = self.dm.polmatrix.copy()
        rdm = self.dm.polmatrix.copy()
        self.dm.polmatrix = 0.50 * (dm + dm.T)
        # self.dm.refdynmatrix = 0.50 * (rdm + rdm.T)

