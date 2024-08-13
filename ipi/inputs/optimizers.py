"""Creates objects that deal with optimization simulations.

Chooses between the different possible optimization options (sd, bfgs, ....) and creates the
appropriate optimization object, with suitable parameters.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


from copy import copy

import numpy as np

import ipi.engine.optimizers as geoptimizers
from ipi.utils.depend import *
from ipi.utils.inputvalue import *


__all__ = ["InputOptimizers"]


class InputOptimizer(Input):

    """Optimizer input class.

    Handles generating the appropriate optimizer class from the xml input file,
    and generating the xml checkpoiunt tags and data from an instance of the
    object.

    Attributes:
       mode: A string giving the type of the optimizer used.

    Fields:
       ls_options: options for sd and cg optimizers

    """

    attribs = {
        "mode": (
            InputAttribute,
            {
                "dtype": str,
                "options": [
                    "",
                    "sd",
                    "bfgs",
                    "bfgstrm",
                    "lbfgs",
                    "cg",
                    "damped_bfgs",
                ],
                #"default": "sd",
                "help": "The type of optimizer",
            },
        )
    }
    fields = {
        "ls_options": (
            InputDictionary,
            {
                "dtype": [float, int, float, float],
                "help": """"Options for line search methods. Includes:
                              tolerance: stopping tolerance for the search (as a fraction of the overall energy tolerance),
                              iter: the maximum number of iterations,
                              step: initial step for bracketing,
                              adaptive: whether to update initial step.
                              """,
                "options": ["tolerance", "iter", "step", "adaptive"],
                "default": [1e-4, 100, 1e-3, 1.0],
                "dimension": ["energy", "undefined", "length", "undefined"],
            },
        ),
        "tolerances": (
            InputDictionary,
            {
                "dtype": float,
                "options": ["energy", "force", "position"],
                "default": [1e-7, 1e-4, 1e-3],
                "help": "Convergence criteria for optimization. Default values are extremely conservative. Set them to appropriate values for production runs.",
                "dimension": ["energy", "force", "length"],
            },
        ), 
        "exit_on_convergence": (
            InputValue,
            {
                "dtype": bool,
                "default": True,
                "help": "Terminates the simulation when the convergence criteria are met.",
            },
        ),
        "invhessian_bfgs": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.eye, args=(0,)),
                "help": "Approximate inverse Hessian for BFGS, if known.",
            },
        ),  
        "biggest_step": (
            InputValue,
            {
                "dtype": float,
                "default": 100.0,
                "help": "The maximum step size for (L)-BFGS line minimizations.",
            },
        ), 
        # re-start parameters, estimate hessian, etc.
        "old_pos": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "The previous positions in an optimization step.",
                "dimension": "length",
            },
        ),
        "old_pot": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "The previous potential energy in an optimization step.",
                "dimension": "energy",
            },
        ),
        "old_force": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "The previous force in an optimization step.",
                "dimension": "force",
            },
        ),
        "old_direction": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "The previous direction in a CG or SD optimization.",
            },
        ),          
    }


    default_help = "Simulates geometry optimiztion."
    default_label = "Optimizers"

    def store(self, optimizer):
        """Takes a optimizer instance and stores a minimal representation of it.

        Args:
           optimizer: A optimizer object.

        Raises:
           TypeError: Raised if the optimizer is not a recognized type.
        """

        super(InputOptimizer, self).store(optimizer)
        if type(optimizer) is geoptimizers.SDOptimizer:
            self.mode.store("sd")
            self.ls_options.store(optimizer.ls_options)
            self.tolerances.store(optimizer.tolerances)
            # do I need to store everything? also old_force etc.?
        elif type(optimizer) is geoptimizers.BFGSOptimizer:
            self.ls_options.store(optimizer.ls_options)
            self.biggest_step.store(optimizer.big_step)
            self.tolerances.store(optimizer.tolerances)
        elif type(optimizer) is geoptimizers.Optimizer:
            self.mode.store("")
        else:
            raise TypeError("Unknown optimizer mode " + type(optimizer).__name__)

    def fetch(self):
        """Creates a optimizer object.

        Returns:
           A optimizer object of the appropriate type and with the appropriate
           parameters given the attributes of the InputOptimizer object.

        Raises:
           TypeError: Raised if the optimizer type is not a recognized option.
        """

        super(InputOptimizer, self).fetch()
        if self.mode.fetch() == "sd":
            optimizer = geoptimizers.SDOptimizer(ls_options=self.ls_options.fetch(), tolerances=self.tolerances.fetch(),
                                                 old_force=self.old_force.fetch(), exit_on_convergence=self.exit_on_convergence.fetch())
        elif self.mode.fetch() == "":
            optimizer = geoptimizers.Optimizer()
        elif self.mode.fetch() == "bfgs":
            optimizer = geoptimizers.BFGSOptimizer(ls_options=self.ls_options.fetch(), old_pos=self.old_pos.fetch(),
                        old_force=self.old_force.fetch(), old_pot=self.old_pot.fetch(), old_direction=self.old_direction.fetch(),
                        invhessian_bfgs=self.invhessian_bfgs.fetch(), biggest_step=self.biggest_step.fetch(), 
                        tolerances=self.tolerances.fetch(), exit_on_convergence=self.exit_on_convergence.fetch())
        elif self.mode.fetch() == "bfgstrm":
            raise TypeError("Optimizer mode " + self.mode.fetch() + " has not been refactored yet")
        elif self.mode.fetch() == "lbfgs":
            raise TypeError("Optimizer mode " + self.mode.fetch() + " has not been refactored yet")
        elif self.mode.fetch() == "cg":
            raise TypeError("Optimizer mode " + self.mode.fetch() + " has not been refactored yet")
        elif self.mode.fetch() == "damped_bfgs":
            raise TypeError("Optimizer mode " + self.mode.fetch() + " has not been refactored yet")
        else:
            raise TypeError("Invalid optimizer mode " + self.mode.fetch())

        return optimizer

    def check(self):
        """Checks that the parameter arrays represents a valid optimizer."""

        super(InputOptimizer, self).check()
        mode = self.mode.fetch()

        if mode in ["sd", ""]:
            pass  # maybe implement some checks later

