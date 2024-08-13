"""Creates objects that deal with the different types of optimization."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np
import ipi.engine.optimizers
from ipi.utils.inputvalue import (
    InputDictionary,
    InputAttribute,
    InputValue,
    InputArray,
    input_default,
)
from ipi.inputs.optimizers import InputOptimizer


__all__ = ["InputOptimization"]


class InputOptimization(InputDictionary):
    """Optimization input class.

    Handles generating the appropriate type of optimization class from the xml input file,
    and generating the xml checkpoint tags and data from an instance of the
    object.

    Attributes:
        mode: An optional string giving the type of optimization to be simulated.
            Defaults to 'minimize'.

    Fields:
        optimizer: The optimizer used during the optimization
    """

    attribs = {
        "mode": (
            InputAttribute,
            {
                "dtype": str,
                "default": "minimize",
                "help": """The type of optimization that will be carried out.
                minimize: simple geometry optimization;
                neb: Option for NEB optimization;
                string: Option for string minimal-energy path optimization
                instanton: Option for instanton optimization
                 """,
                "options": [
                    "minimize",
                    "neb",
                    "instanton",
                    "string",
                ],
            },
        ),
    }

    fields = {
        "optimizer": (
            InputOptimizer,
            {
                "default": input_default(factory=ipi.engine.optimizers.Optimizer),
                "help": "The optimizer for the geometry",
            },
        ),
    }

    dynamic = {}

    default_help = "Holds all the information for the optimization, such as type of optimization and optimizer."
    default_label = "OPTIMIZATION"

    def store(self, opt):
        """Takes an optimization instance and stores a minimal representation of it.

        Args:
            opt: An integrator object.
        """

        if opt == {}:
            return

        self.mode.store(opt.mode)
        self.optimizer.store(opt.optimizer)

    def fetch(self):
        """Creates an optimization object.

        Returns:
            An optimization object of the appropriate mode and with the appropriate
            objects given the attributes of the InputOptimization object.
        """

        rv = super(InputOptimization, self).fetch()
        rv["mode"] = self.mode.fetch()
        print(self)
        return rv