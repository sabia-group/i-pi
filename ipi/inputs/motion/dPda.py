import numpy as np
from ipi.engine.motion import *
from ipi.utils.inputvalue import *
from ipi.inputs.thermostats import *
from ipi.inputs.initializer import *
from ipi.utils.units import *

__all__ = ["InputdPdaTensorCalculator"]


class InputdPdaTensorCalculator(InputDictionary):

    """
    Contains options related with finite difference computation of dP/da tensor
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
                "help": "The finite displacement in position used to compute derivative of the polarization.",
            },
        ),
        "prefix": (
            InputValue,
            {"dtype": str, "default": "dPda", "help": "Prefix of the output files."},
        ),
        # "asr": (
        #     InputValue,
        #     {
        #         "dtype": str,
        #         "default": "none",
        #         "options": ["none"],#, "poly", "lin", "crystal"],
        #         "help": "Removes the zero frequency vibrational modes depending on the symmerty of the system.",
        #     },
        # ),
        # "Epolmatrix": (
        #     InputArray,
        #     {
        #         "dtype": float,
        #         "default": np.zeros(0, float),
        #         "help": "Portion of the electronic polarization matrix known up to now.",
        #     },
        # ),
        # "Ipolmatrix": (
        #     InputArray,
        #     {
        #         "dtype": float,
        #         "default": np.zeros(0, float),
        #         "help": "Portion of the ionic polarization matrix known up to now.",
        #     },
        # ),
        "polmatrix": (
            InputArray,
            {
                "dtype": float,
                "default": np.zeros((6,3), float),
                "help": "Portion of the total polarization matrix known up to now.",
            },
        ),
    }

    dynamic = {}

    default_help = "Fill in."
    default_label = "dPda"

    def store(self, phonons):
        if phonons == {}:
            return
        self.mode.store(phonons.mode)
        self.pos_shift.store(phonons.deltax)
        self.prefix.store(phonons.prefix)
        # self.asr.store(phonons.asr)
        self.polmatrix.store(phonons.polmatrix)

    def fetch(self):
        rv = super(InputdPdaTensorCalculator, self).fetch()
        rv["mode"] = self.mode.fetch()
        return rv
