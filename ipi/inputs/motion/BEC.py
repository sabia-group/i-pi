import numpy as np
from ipi.engine.motion import *
from ipi.utils.inputvalue import *
from ipi.inputs.thermostats import *
from ipi.inputs.initializer import *
from ipi.utils.units import *

__all__ = ["InputBECTensorsCalculator"]


class InputBECTensorsCalculator(InputDictionary):

    """
    Contains options related with finite difference computation of Born Effective Charges.
    """

    attribs = {
        # "mode": (
        #     InputAttribute,
        #     {
        #         "dtype": str,
        #         "default": "fd",
        #         "help": "The algorithm to be used: only finite differences (fd) is currently implemented.",
        #         "options": ["fd"],
        #     },
        # )
    }
    fields = {
        "pos_shift": (
            InputValue,
            {
                "dtype": float,
                "default": 0.01,
                "help": "The finite displacement in position used to compute derivative of (electronic) polarization.",
            },
        ),
        "prefix": (
            InputValue,
            {"dtype": str, "default": "BEC", "help": "Prefix of the output files."},
        ),
        "asr": (
            InputValue,
            {
                "dtype": str,
                "default": "none",
                "options": ["none","lin"],#, "poly", "lin", "crystal"],
                "help": "Removes the zero frequency vibrational modes depending on the symmerty of the system.",
            },
        ),
        "polmatrix": (
            InputArray,
            {
                "dtype": float,
                "default": np.zeros(0, float),
                "help": "Portion of the total polarization matrix known up to now.",
            },
        ),
        "atoms": (
            InputArray,
            {
                "dtype": str,
                "default": np.asarray(["all"]),
                "help": "Atoms whose BEC tensor has to be computed. It can be 'all', a chemical species ('Li', 'Mg') or an atom index. List of the previous cases are accepted.",
            },
        ),
    }

    dynamic = {}

    default_help = "Fill in."
    default_label = "BEC"

    def store(self, phonons):
        if phonons == {}:
            return
        # self.mode.store(phonons.mode)
        self.pos_shift.store(phonons.deltax)
        self.prefix.store(phonons.prefix)
        self.asr.store(phonons.asr)
        self.polmatrix.store(phonons.polmatrix)
        self.atoms.store(phonons.atoms)

    def fetch(self):
        rv = super(InputBECTensorsCalculator, self).fetch()
        # rv["mode"] = self.mode.fetch()
        return rv
