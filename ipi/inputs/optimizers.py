"""Creates objects that deal with optimization simulations.

Chooses between the different possible optimization options (sd, bfgs, ....) and creates the
appropriate optimization object, with suitable parameters.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


from copy import copy

import numpy as np

import ipi.engine.motion.geop as geoptimizers
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
        elif type(optimizer) is geoptimizers.GeopMotion:
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
            optimizer = geoptimizers.SDOptimizer(ls_options=self.ls_options.fetch())
        elif self.mode.fetch() == "":
            optimizer = geoptimizers.GeopMotion()
        else:
            raise TypeError("Invalid optimizer mode " + self.mode.fetch())

        return optimizer

    def check(self):
        """Checks that the parameter arrays represents a valid optimizer."""

        super(InputOptimizer, self).check()
        mode = self.mode.fetch()

        if mode in ["sd", ""]:
            pass  # maybe implement some checks later

'''
class InputOptimizer(InputOptimizerBase):


    attribs = copy(InputOptimizerBase.attribs)


    dynamic = {
        "optimizer": (
            InputOptimizerBase,
            {
                "default": input_default(factory=geoptimizers.GeopMotion),
                "help": "The optimizer for the atoms.",
            },
        )
    }

    def store(self, optimizer):
        super(InputOptimizer, self).store(optimizer)

    def fetch(self):
        optimizer = super(InputOptimizer, self).fetch()
        return optimizer'''
