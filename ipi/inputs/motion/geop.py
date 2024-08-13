"""Deals with creating the ensembles class.

Copyright (C) 2013, Joshua More and Michele Ceriotti

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http.//www.gnu.org/licenses/>.

Inputs created by Michele Ceriotti and Benjamin Helfrecht, 2015

Classes:
   InputGeop: Deals with creating the Geop object from a file, and
      writing the checkpoints.
"""

import numpy as np
from ipi.engine.motion import *
import ipi.engine.motion.geop
from ipi.utils.inputvalue import *
from ipi.inputs.thermostats import *
from ipi.inputs.initializer import *
from ipi.utils.units import *
from ipi.utils.inputvalue import (
    InputDictionary,
    InputAttribute,
    InputValue,
    InputArray,
    input_default,
)

# create InputOptimizer file in ipi.inputs
from ipi.inputs.optimizers import InputOptimizer
import ipi.engine.motion.geop as geoptimizers

__all__ = ["InputGeop"]


class InputGeop(InputDictionary):
    """Geometry optimization options.

    Contains options related with geometry optimization, such as method,
    thresholds, linear search strategy, etc.

    """

    attribs = {
        "mode": (
            InputAttribute,
            {
                "dtype": str,
                "default": "",
                "help": "The geometry optimization algorithm to be used",
                "options": ["", "sd"],
            },
        )
    }

    # options of the method (mostly tolerances)
    fields = {
        "optimizer":(
            InputOptimizer,
            {
                "default": input_default(factory=ipi.engine.motion.geop.GeopMotion),
                "help": "The geometry optimization algorithm used",
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
    }

    dynamic = {}

    default_help = (
        "A Geometry Optimization class implementing most of the standard methods"
    )
    default_label = "GEOP"

    def store(self, geop):
        if geop == {}:
            return

        self.mode.store(geop.mode)
        self.tolerances.store(geop.tolerances)
        self.exit_on_convergence.store(geop.conv_exit)
        self.optimizer.store(geop.optimizer)

        """if geop.mode == "bfgs":
            self.old_direction.store(geop.d)
            self.invhessian_bfgs.store(geop.invhessian)
            self.biggest_step.store(geop.big_step)
        elif geop.mode == "bfgstrm":
            self.hessian_trm.store(geop.hessian)
            self.tr_trm.store(geop.tr)
            self.biggest_step.store(geop.big_step)
        elif geop.mode == "lbfgs":
            self.old_direction.store(geop.d)
            self.qlist_lbfgs.store(geop.qlist)
            self.glist_lbfgs.store(geop.glist)
            self.corrections_lbfgs.store(geop.corrections)
            self.scale_lbfgs.store(geop.scale)
            self.biggest_step.store(geop.big_step)
        elif geop.mode == "sd":
            self.ls_options.store(geop.ls_options)
        elif geop.mode == "cg":
            self.old_direction.store(geop.d)
            self.ls_options.store(geop.ls_options)
            self.old_force.store(geop.old_f)
        if geop.mode == "damped_bfgs":
            self.old_direction.store(geop.d)
            self.invhessian_bfgs.store(geop.invhessian)
            self.biggest_step.store(geop.big_step)"""

    def fetch(self):
        rv = super(InputGeop, self).fetch()
        rv["mode"] = self.mode.fetch()
        return rv
