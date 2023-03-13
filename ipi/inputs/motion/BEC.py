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
from ipi.utils.inputvalue import *
from ipi.inputs.thermostats import *
from ipi.inputs.initializer import *
from ipi.utils.units import *

__all__ = ["InputDynMatrix"]


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
        "Epolmatrix": (
            InputArray,
            {
                "dtype": float,
                "default": np.zeros(0, float),
                "help": "Portion of the electronic polarization matrix known up to now.",
            },
        ),
        "Ipolmatrix": (
            InputArray,
            {
                "dtype": float,
                "default": np.zeros(0, float),
                "help": "Portion of the ionic polarization matrix known up to now.",
            },
        ),
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
