"""Classes that deal with optimizer algorithms"""

import numpy as np
import time

from ipi.engine.motion import Motion
from ipi.utils.depend import dstrip
from ipi.utils.softexit import softexit
from ipi.utils.mintools import min_brent, BFGS, BFGSTRM, L_BFGS, Damped_BFGS
from ipi.utils.messages import verbosity, info

class Optimizer:
    """class for all optimizer classes"""

    def __init__(self):
        """initialises object for LineMapper (1-d function)
        and for GradientMapper (multi-dimensional function)
        """

        self.lm = LineMapper()
        self.gm = GradientMapper()
        self.converged = False

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""
        pass

    def bind(self, geop):
        """
        bind optimization options and call bind function
        of LineMapper and GradientMapper (get beads, cell,forces)
        check whether force size, direction size and inverse Hessian size
        from previous step match system size
        """
        self.beads = geop.beads
        self.cell = geop.cell
        self.forces = geop.forces
        self.fixcom = geop.fixcom
        self.fixatoms = geop.fixatoms

        self.mode = geop.mode

        # Check for very tight tolerances
        if self.tolerances["position"] < 1e-7:
            raise ValueError(
                "The position tolerance is too small for any typical calculation. "
                "We stop here. Comment this line and continue only if you know what you are doing"
            )
        if self.tolerances["force"] < 1e-7:
            raise ValueError(
                "The force tolerance is too small for any typical calculation. "
                "We stop here. Comment this line and continue only if you know what you are doing"
            )
        if self.tolerances["energy"] < 1e-10:
            raise ValueError(
                "The energy tolerance is too small for any typical calculation. "
                "We stop here. Comment this line and continue only if you know what you are doing"
            )

        # The resize action must be done before the bind (though not every optimizers needs old_x, old_f, old_u or d)
        if hasattr(self, 'old_x'):
            if self.old_x.size != self.beads.q.size:
                if self.old_x.size == 0:
                    self.old_x = np.zeros((self.beads.nbeads, 3 * self.beads.natoms), float)
                else:
                    raise ValueError(
                        "Conjugate gradient position size does not match system size"
                    )
        if hasattr(self, 'old_f'):         
            if self.old_f.size != self.beads.q.size:
                if self.old_f.size == 0:
                    self.old_f = np.zeros((self.beads.nbeads, 3 * self.beads.natoms), float)
                else:
                    raise ValueError(
                        "Conjugate gradient force size does not match system size"
                    )
        if hasattr(self, 'old_u'):   
            if self.old_u.size != 1:
                if self.old_u.size == 0:
                    self.old_u = np.zeros(1, float)
                else:
                    raise ValueError("Conjugate gradient has weird potential (size != 1)"
                    )
        if hasattr(self, 'd'):
            if self.d.size != self.beads.q.size:
                if self.d.size == 0:
                    self.d = np.zeros((self.beads.nbeads, 3 * self.beads.natoms), float)
                else:
                    raise ValueError(
                        "Conjugate gradient direction size does not match system size"
                    )


    def exitstep(self, fx, u0, x):
        """Exits the simulation step. Computes time, checks for convergence."""

        info(" @GEOP: Updating bead positions", verbosity.debug)
        self.qtime += time.time()

        if len(self.fixatoms) > 0:
            ftmp = self.forces.f.copy()
            for dqb in ftmp:
                dqb[self.fixatoms * 3] = 0.0
                dqb[self.fixatoms * 3 + 1] = 0.0
                dqb[self.fixatoms * 3 + 2] = 0.0
            fmax = np.amax(np.absolute(ftmp))
        else:
            fmax = np.amax(np.absolute(self.forces.f))

        e = np.absolute((fx - u0) / self.beads.natoms)
        info("@GEOP", verbosity.medium)
        self.tolerances["position"]
        info("   Current energy             %e" % (fx))
        info(
            "   Position displacement      %e  Tolerance %e"
            % (x, self.tolerances["position"]),
            verbosity.medium,
        )
        info(
            "   Max force component        %e  Tolerance %e"
            % (fmax, self.tolerances["force"]),
            verbosity.medium,
        )
        info(
            "   Energy difference per atom %e  Tolerance %e"
            % (e, self.tolerances["energy"]),
            verbosity.medium,
        )

        if np.linalg.norm(self.forces.f.flatten() - self.old_f.flatten()) <= 1e-20:
            info(
                "Something went wrong, the forces are not changing anymore."
                " This could be due to an overly small tolerance threshold "
                "that makes no physical sense. Please check if you are able "
                "to reach such accuracy with your force evaluation"
                " code (client)."
            )

        if (
            (np.absolute((fx - u0) / self.beads.natoms) <= self.tolerances["energy"])
            and (fmax <= self.tolerances["force"])
            and (x <= self.tolerances["position"])
        ):
            self.converged = True

class LineMapper(object):
    """Creation of the one-dimensional function that will be minimized.
    Used in steepest descent and conjugate gradient minimizers.

    Attributes:
        x0: initial position
        d: move direction
    """

    def __init__(self):
        self.x0 = self.d = None
        self.fcount = 0

    def bind(self, dumop):
        self.dbeads = dumop.beads.copy()
        self.dcell = dumop.cell.copy()
        self.dforces = dumop.forces.copy(self.dbeads, self.dcell)

        self.fixatoms_mask = np.ones(
            3 * dumop.beads.natoms, dtype=bool
        )  # Mask to exclude fixed atoms from 3N-arrays
        if len(dumop.fixatoms) > 0:
            self.fixatoms_mask[3 * dumop.fixatoms] = 0
            self.fixatoms_mask[3 * dumop.fixatoms + 1] = 0
            self.fixatoms_mask[3 * dumop.fixatoms + 2] = 0

    def set_dir(self, x0, mdir):
        self.x0 = x0.copy()

        # exclude fixed degrees of freedom and renormalize direction vector to unit length:
        tmp3 = mdir.copy()[:, self.fixatoms_mask]
        self.d = tmp3 / np.sqrt(np.dot(tmp3.flatten(), tmp3.flatten()))
        del tmp3
        if self.x0[:, self.fixatoms_mask].shape != self.d.shape:
            raise ValueError(
                "Incompatible shape of initial value and displacement direction"
            )

    def __call__(self, x):
        """computes energy and gradient for optimization step
        determines new position (x0+d*x)"""

        self.fcount += 1
        self.dbeads.q[:, self.fixatoms_mask] = (
            self.x0[:, self.fixatoms_mask] + self.d * x
        )
        e = self.dforces.pot  # Energy
        g = -np.dot(
            dstrip(self.dforces.f[:, self.fixatoms_mask]).flatten(), self.d.flatten()
        )  # Gradient
        return e, g


class GradientMapper(object):
    """Creation of the multi-dimensional function that will be minimized.
    Used in the BFGS and L-BFGS minimizers.

    Attributes:
        dbeads:   copy of the bead object
        dcell:   copy of the cell object
        dforces: copy of the forces object
    """

    def __init__(self):
        self.fcount = 0
        pass

    def bind(self, dumop):
        self.dbeads = dumop.beads.copy()
        self.dcell = dumop.cell.copy()
        self.dforces = dumop.forces.copy(self.dbeads, self.dcell)

        self.fixatoms_mask = np.ones(
            3 * dumop.beads.natoms, dtype=bool
        )  # Mask to exclude fixed atoms from 3N-arrays
        if len(dumop.fixatoms) > 0:
            self.fixatoms_mask[3 * dumop.fixatoms] = 0
            self.fixatoms_mask[3 * dumop.fixatoms + 1] = 0
            self.fixatoms_mask[3 * dumop.fixatoms + 2] = 0

    def __call__(self, x):
        """computes energy and gradient for optimization step"""

        self.fcount += 1
        self.dbeads.q[:, self.fixatoms_mask] = x
        e = self.dforces.pot  # Energy
        g = -self.dforces.f[:, self.fixatoms_mask]  # Gradient
        return e, g
    

class SDOptimizer(Optimizer):
    """
    Steepest descent minimization
    dq1 = direction of steepest descent
    dq1_unit = unit vector of dq1
    """

    def __init__(self, ls_options, tolerances, old_force, exit_on_convergence):
        super(SDOptimizer, self).__init__()
        self.ls_options = ls_options
        self.tolerances = tolerances
        self.old_f = old_force
        self.conv_exit = exit_on_convergence

    def bind(self, geop):
        # call bind function from DummyOptimizer
        super(SDOptimizer, self).bind(geop)
        self.lm.bind(self)
        #self.ls_options = geop.ls_options

    def step(self, step=None):
        """Does one simulation time step
        Attributes:
        ttime: The time taken in applying the thermostat steps.
        """

        self.qtime = -time.time()
        info("\nMD STEP %d" % step, verbosity.debug)

        # Store previous forces for warning exit condition
        self.old_f[:] = self.forces.f

        # Check for fixatoms
        if len(self.fixatoms) > 0:
            for dqb in self.old_f:
                dqb[self.fixatoms * 3] = 0.0
                dqb[self.fixatoms * 3 + 1] = 0.0
                dqb[self.fixatoms * 3 + 2] = 0.0

        dq1 = dstrip(self.old_f)

        # Move direction for steepest descent
        dq1_unit = dq1 / np.sqrt(np.dot(dq1.flatten(), dq1.flatten()))
        info(" @GEOP: Determined SD direction", verbosity.debug)

        # Set position and direction inside the mapper
        self.lm.set_dir(dstrip(self.beads.q), dq1_unit)

        # Reuse initial value since we have energy and forces already
        u0, du0 = (
            self.forces.pot.copy(),
            np.dot(dstrip(self.forces.f.flatten()), dq1_unit.flatten()),
        )

        # Do one SD iteration; return positions and energy
        # (x, fx,dfx) = min_brent(self.lm, fdf0=(u0, du0), x0=0.0,  #DELETE
        min_brent(
            self.lm,
            fdf0=(u0, du0),
            x0=0.0,
            tol=self.ls_options["tolerance"] * self.tolerances["energy"],
            itmax=self.ls_options["iter"],
            init_step=self.ls_options["step"],
        )
        info("   Number of force calls: %d" % (self.lm.fcount))
        self.lm.fcount = 0

        # Update positions and forces
        self.beads.q = self.lm.dbeads.q
        self.forces.transfer_forces(
            self.lm.dforces
        )  # This forces the update of the forces

        d_x = np.absolute(np.subtract(self.beads.q, self.lm.x0))
        x = np.linalg.norm(d_x)
        # Automatically adapt the search step for the next iteration.
        # Relaxes better with very small step --> multiply by factor of 0.1 or 0.01

        self.ls_options["step"] = (
            0.1 * x * self.ls_options["adaptive"]
            + (1 - self.ls_options["adaptive"]) * self.ls_options["step"]
        )

        # Exit simulation step
        d_x_max = np.amax(np.absolute(d_x))
        self.exitstep(self.forces.pot, u0, d_x_max)

class BFGSOptimizer(Optimizer):
    """BFGS Minimization"""

    def __init__(self, ls_options, old_pos, old_force, old_pot, old_direction,
                        invhessian_bfgs, biggest_step, 
                        tolerances, exit_on_convergence):
        
        super(BFGSOptimizer, self).__init__()
        self.ls_options = ls_options
        self.old_x = old_pos
        self.old_f = old_force
        self.old_u = old_pot
        self.d = old_direction
        self.invhessian = invhessian_bfgs
        self.big_step = biggest_step
        self.tolerances = tolerances
        self.conv_exit = exit_on_convergence

    def bind(self, geop):
        # call bind function from DummyOptimizer
        super(BFGSOptimizer, self).bind(geop)

        if self.invhessian.size != (self.beads.q.size * self.beads.q.size):
            if self.invhessian.size == 0:
                self.invhessian = np.eye(self.beads.q.size, self.beads.q.size, 0, float)
            else:
                raise ValueError("Inverse Hessian size does not match system size")

        self.gm.bind(self)

    def step(self, step=None):
        """Does one simulation time step.
        Attributes:
        qtime: The time taken in updating the positions.
        """

        self.qtime = -time.time()
        info("\nMD STEP %d" % step, verbosity.debug)

        if step == 0:
            info(" @GEOP: Initializing BFGS", verbosity.debug)
            self.d += dstrip(self.forces.f) / np.sqrt(
                np.dot(self.forces.f.flatten(), self.forces.f.flatten())
            )

            if len(self.fixatoms) > 0:
                for dqb in self.d:
                    dqb[self.fixatoms * 3] = 0.0
                    dqb[self.fixatoms * 3 + 1] = 0.0
                    dqb[self.fixatoms * 3 + 2] = 0.0

        self.old_x[:] = self.beads.q
        self.old_u[:] = self.forces.pot
        self.old_f[:] = self.forces.f

        if len(self.fixatoms) > 0:
            for dqb in self.old_f:
                dqb[self.fixatoms * 3] = 0.0
                dqb[self.fixatoms * 3 + 1] = 0.0
                dqb[self.fixatoms * 3 + 2] = 0.0

            fdf0 = (self.old_u, -self.old_f[:, self.gm.fixatoms_mask])

            # Reduce dimensionality
            masked_old_x = self.old_x[:, self.gm.fixatoms_mask]
            masked_d = self.d[:, self.gm.fixatoms_mask]
            masked_invhessian = self.invhessian[
                np.ix_(self.gm.fixatoms_mask, self.gm.fixatoms_mask)
            ]

            # Do one iteration of BFGS
            # The invhessian and the directions are updated inside.
            # Everything passed inside BFGS() in masked form, including the invhessian
            BFGS(
                masked_old_x,
                masked_d,
                self.gm,
                fdf0,
                masked_invhessian,
                self.big_step,
                self.ls_options["tolerance"] * self.tolerances["energy"],
                self.ls_options["iter"],
            )

            # Restore dimensionality of d and invhessian
            self.d[:, self.gm.fixatoms_mask] = masked_d
            self.invhessian[
                np.ix_(self.gm.fixatoms_mask, self.gm.fixatoms_mask)
            ] = masked_invhessian

        else:
            fdf0 = (self.old_u, -self.old_f)

            # Do one iteration of BFGS
            # The invhessian and the directions are updated inside.
            BFGS(
                self.old_x,
                self.d,
                self.gm,
                fdf0,
                self.invhessian,
                self.big_step,
                self.ls_options["tolerance"] * self.tolerances["energy"],
                self.ls_options["iter"],
            )

        info("   Number of force calls: %d" % (self.gm.fcount))
        self.gm.fcount = 0
        # Update positions and forces
        self.beads.q = self.gm.dbeads.q
        self.forces.transfer_forces(
            self.gm.dforces
        )  # This forces the update of the forces

        # Exit simulation step
        d_x_max = np.amax(np.absolute(np.subtract(self.beads.q, self.old_x)))
        self.exitstep(self.forces.pot, self.old_u, d_x_max)
