import numpy as np
from ipi.engine.motion import *
from ipi.utils.inputvalue import *
from ipi.inputs.thermostats import *
from ipi.inputs.initializer import *
from ipi.utils.units import *

__all__ = ["InputdDdaTensorCalculator"]


class InputdDdaTensorCalculator(InputDictionary):

    """
    Contains options related with finite difference computation of dD/da tensor
    """

    attribs = {
        "mode": (
            InputAttribute,
            {
                "dtype": str,
                "default": "fd",
                "help": "The algorithm to be used: only finite differences (fd) is currently implemented.",
                "options": ["fd"],
            },
        )
    }
    fields = {
        "pos_shift": (
            InputValue,
            {
                "dtype": float,
                "default": 0.01,
                "help": "The finite displacement in position used to compute derivative of the dipole.",
            },
        ),
        "prefix": (
            InputValue,
            {"dtype": str, "default": "dDda", "help": "Prefix of the output files."},
        ),
        "matrix": (
            InputArray,
            {
                "dtype": float,
                "default": np.zeros((6,3), float),
                "help": "Portion of the total matrix known up to now.",
            },
        ),
    }

    dynamic = {}

    default_help = "Fill in."
    default_label = "dDda"

    def store(self, phonons):
        if phonons == {}:
            return
        self.mode.store(phonons.mode)
        self.pos_shift.store(phonons.deltax)
        self.prefix.store(phonons.prefix)
        # self.asr.store(phonons.asr)
        self.matrix.store(phonons.matrix)

    def fetch(self):
        rv = super(InputdDdaTensorCalculator, self).fetch()
        rv["mode"] = self.mode.fetch()
        return rv
