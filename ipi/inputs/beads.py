"""Creates objects that deal with path integral beads."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np

from ipi.engine.beads import *
from ipi.utils.inputvalue import *
from ipi.utils.depend import *
from ipi.utils.units import *
from ipi.inputs.atoms import *


__all__ = ["InputBeads"]


class InputBeads(Input):

    """Beads input class.

    Handles generating the appropriate beads class from the xml input file,
    and generating the xml checkpoint tags and data from an instance of the
    object.

    Attributes:
       nbeads: An optional integer giving the number of beads. Defaults to 0.
       natoms: An optional integer giving the number of atoms. Defaults to 0.

    Fields:
        q: An optional array giving the bead positions. Defaults to an empty array with no elements.
        p: An optional array giving the bead momenta. Defaults to an empty array with no elements.
        m: An optional array giving the bead masses. Defaults to an empty array with no elements.
        IonsPol : an array giving the ionic polarization.
        ElecPol : an array giving the ionic polarization.
        TotalPol: an array giving the ionic polarization.
        names: An optional array giving the bead names. Defaults to an empty array with no elements.
    """

    attribs = {
        "natoms": (
            InputAttribute,
            {"dtype": int, "default": 0, "help": "The number of atoms."},
        ),
        "nbeads": (
            InputAttribute,
            {"dtype": int, "default": 0, "help": "The number of beads."},
        )
    }
    fields = {
        "q": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "The positions of the beads. In an array of size [nbeads, 3*natoms].",
                "dimension": "length",
            },
        ),
        "p": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "The momenta of the beads. In an array of size [nbeads, 3*natoms].",
                "dimension": "momentum",
            },
        ),
        "m": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "The masses of the atoms, in the format [m1, m2, ... ].",
                "dimension": "mass",
            },
        ),
        "names": (
            InputArray,
            {
                "dtype": str,
                "default": input_default(
                    factory=np.zeros, args=(0,), kwargs={"dtype": np.dtype("|U6")}
                ),
                "help": "The names of the atoms, in the format [name1, name2, ... ].",
            },
        ),
        "IonsPol": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "The ionic polarization expressed in cartesian coordinates",
            },
        ),
        "ElecPol": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "The electronic polarization expressed in cartesian coordinates",
            },
        ),
        "TotalPol": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "The total polarization expressed in cartesian coordinates",
            },
        ),
    }

    default_help = "Describes the bead configurations in a path integral simulation."
    default_label = "BEADS"

    def store(self, beads):
        """Takes a Beads instance and stores a minimal representation of it.

        Args:
           beads: A Beads object from which to initialise from.
        """

        super(InputBeads, self).store()
        self.natoms.store(beads.natoms)
        self.nbeads.store(beads.nbeads)

        self.q.store(dstrip(beads.q))
        self.p.store(dstrip(beads.p))
        self.m.store(dstrip(beads.m))
        self.IonsPol.store(dstrip(beads.IonsPol))
        self.ElecPol.store(dstrip(beads.ElecPol))
        self.TotalPol.store(dstrip(beads.TotalPol))
        self.names.store(dstrip(beads.names))
        
    def fetch(self):
        """Creates a beads object.

        Returns:
           A beads object of the appropriate type and with the appropriate
           properties given the attributes of the InputBeads object.
        """

        super(InputBeads, self).fetch()
        beads = Beads(self.natoms.fetch(), self.nbeads.fetch())

        # tries to fill up with as much data as available and valid
        q = self.q.fetch()
        if q.shape == (beads.nbeads, 3 * beads.natoms):
            beads.q = q
        elif beads.nbeads == 1 and q.shape == (3 * beads.natoms,):
            beads.q = q
        elif len(q) != 0:
            raise ValueError("Array shape mismatches for q in <beads> input.")

        p = self.p.fetch()
        if p.shape == (beads.nbeads, 3 * beads.natoms):
            beads.p = p
        elif beads.nbeads == 1 and p.shape == (3 * beads.natoms,):
            beads.p = p
        elif len(p) != 0:
            raise ValueError("Array shape mismatches for p in <beads> input.")

        m = self.m.fetch()
        if m.shape == (beads.natoms,):
            beads.m = m
        elif len(m) != 0:
            raise ValueError("Array shape mismatches for m in <beads> input.")

        n = self.names.fetch()
        if n.shape == (beads.natoms,):
            beads.names = n
        elif len(n) != 0:
            raise ValueError("Array shape mismatches for names in <beads> input.")
        
        IonsPol = self.IonsPol.fetch()
        if IonsPol.shape == (beads.nbeads,beads.natoms*3):
            beads.IonsPol = IonsPol
        elif len(IonsPol) != 0:
            raise ValueError("Array shape mismatches for IonsPol in <beads> input.")

        ElecPol = self.ElecPol.fetch()
        if ElecPol.shape == (beads.nbeads,beads.natoms*3):
            beads.ElecPol = ElecPol
        elif len(ElecPol) != 0:
            raise ValueError("Array shape mismatches for ElecPol in <beads> input.")

        TotalPol = self.TotalPol.fetch()
        if TotalPol.shape == (beads.nbeads,beads.natoms*3):
            beads.TotalPol = TotalPol
        elif len(TotalPol) != 0:
            raise ValueError("Array shape mismatches for TotalPol in <beads> input.")

        return beads
