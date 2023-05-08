import numpy as np
from ipi.engine.motion import *
from ipi.utils.inputvalue import *
from ipi.inputs.thermostats import *
from ipi.inputs.initializer import *
from ipi.utils.units import *
from ipi.inputs.initializer import InputInitPositions

__all__ = ["InputSequence"]


class InputSequence(InputDictionary):

    """
    """

    attribs = { }
    dynamic = { }
    fields = {
        "input": (
            InputValue,
            {
                "dtype": str,
                "default": "",
                "help": "The input file (in xyz format) with the positions to be considered",
            },
        ),
    }

    default_help = "Fill in."
    default_label = "seq"

    def store(self, ii):
        if ii == {}:
            return
        self.input.store(ii.input)
        pass

    def fetch(self):
        rv = super(InputSequence, self).fetch()
        rv["input"] = self.input.fetch()
        return rv
