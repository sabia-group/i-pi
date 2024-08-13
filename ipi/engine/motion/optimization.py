"""Contains the classes that deal with the different optimizations.

"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np

from ipi.engine.motion import Motion
from ipi.utils.depend import *
from ipi.engine.optimizers import Optimizer
from ipi.utils.softexit import softexit
from ipi.utils.messages import warning, verbosity, info


class Optimization(Motion):
    """self optimization class.

    Gives the standard methods and attributes needed in all the
    optimization classes.

    Attributes:
        beads: A beads object giving the atoms positions.
        cell: A cell object giving the system box.
        forces: A forces object giving the virial and the forces acting on
            each bead.
        prng: A random number generator object.
        nm: An object which does the normal modes transformation.

    """

    def __init__(
        self,
        mode="minimize",
        optimizer=None,
        fixcom=False,
        fixatoms=None,
    ):
        """Initialises a "optimization" motion object.
        """

        super(Optimization, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        if optimizer is None:
            self.optimizer = Optimizer()
        else:
            self.optimizer = optimizer

        self.mode = mode
        if self.mode == "minimize":
            self.opt_type = MinGeopt()
        elif self.mode == "neb":
            softexit.trigger(
                status="bad",
                message="!! Sorry, neb optimization not yet refactored. !!",
            )
        elif self.mode == "instanton":
            softexit.trigger(
                status="bad",
                message="!! Sorry, instanton optimization not yet refactored. !!",
            )
        elif self.mode == "string":
            softexit.trigger(
                status="bad",
                message="!! Sorry, string optimization not yet refactored. !!",
            )
        else:
            self.opt_type = DummyOptimization()

        # constraints
        self.fixcom = fixcom
        if fixatoms is None:
            self.fixatoms = np.zeros(0, int)
            #self.fixatoms3 = np.zeros(0, int)  # should I change that?
        else:
            self.fixatoms = fixatoms
            #self.fixatoms3 = np.array(
               # [[3 * i, 3 * i + 1, 3 * i + 2] for i in self.fixatoms]
            #).flatten()


    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        """Binds beads, cell, bforce, and prng to the optimization.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the optimization.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network. Note that the conserved
        quantity is defined in the init, but as each ensemble has a different
        conserved quantity the dependencies are defined in bind.

        Args:
            beads: The beads object from which the bead positions are taken.
            nm: A normal modes object used to do the normal modes transformation.
            cell: The cell object from which the system box is taken.
            bforce: The forcefield object from which the force and virial are
                taken.
            prng: The random number generator object which controls random number
                generation.
        """

        super(Optimization, self).bind(ens, beads, nm, cell, bforce, prng, omaker)

        self.opt_type.bind(self)

        if len(self.fixatoms) == len(self.beads[0]):
            softexit.trigger(
                status="bad",
                message="WARNING: all atoms are fixed, geometry won't change. Exiting simulation",
            )


    def step(self, step=None):
        """Advances the optimization by one time step"""

        self.opt_type.step(step)


#dproperties(Optimization, ["dt", "nmts", "splitting", "ntemp"])

class DummyOptimization:  # add something here later
    """Dummy class for all optimization types"""

    def __init__(self):
        pass

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""
        pass


class MinGeopt:
    """ Class for simple geometry optimization.
    
    Attributes:
        optimizer: An optimizer object 

    """

    def bind(self, opt):
        
        self.beads = opt.beads
        self.cell = opt.cell
        self.forces = opt.forces
        self.fixcom = opt.fixcom
        self.fixatoms = opt.fixatoms
        self.mode = opt.mode
        self.optimizer = opt.optimizer

        self.optimizer.bind(self)

    def step(self, step=None):
        if self.optimizer.converged:
            # if required, exit upon convergence. otherwise just return without action
            if self.optimizer.conv_exit:
                softexit.trigger(
                    status="success",
                    message="Geometry optimization converged. Exiting simulation",
                )
            else:
                info(
                    "Convergence threshold met. Will carry on but do nothing.",
                    verbosity.high,
                )
        else:
            self.optimizer.step(step)
