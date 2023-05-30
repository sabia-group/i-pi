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
        "Tpolmatrix": (
            InputArray,
            {
                "dtype": float,
                "default": np.zeros(0, float),
                "help": "Portion of the total polarization matrix known up to now.",
            },
        ),
        # "refdynmat": (
        #     InputArray,
        #     {
        #         "dtype": float,
        #         "default": np.zeros(0, float),
        #         "help": "Portion of the refined dynamical matrix known up to now.",
        #     },
        # ),
    }

    dynamic = {}

    default_help = "Fill in."
    default_label = "BEC"

    def store(self, phonons):
        if phonons == {}:
            return
        self.mode.store(phonons.mode)
        self.pos_shift.store(phonons.deltax)
        #self.energy_shift.store(phonons.deltae)
        #self.output_shift.store(phonons.deltaw)
        self.prefix.store(phonons.prefix)
        self.asr.store(phonons.asr)
        self.Epolmatrix.store(phonons.Epolmatrix)
        self.Ipolmatrix.store(phonons.Ipolmatrix)
        self.Tpolmatrix.store(phonons.Tpolmatrix)
        #self.refdynmat.store(phonons.refdynmatrix)

    def fetch(self):
        rv = super(InputBECTensorsCalculator, self).fetch()
        rv["mode"] = self.mode.fetch()
        return rv
