
__all__ = ["Sequence"]

import numpy as np
from numpy.linalg import norm


from ipi.engine.motion import Motion
from ipi.utils.depend import *
from ipi.utils.softexit import softexit
from ipi import ipi_global_settings
from ase.io import read


class Sequence(Motion):

    """Calculator given a sequence of nuclear coordinates"""

    def __init__(
        self,
        input="",
    ):
        """Initialises Sequence.
        """
        
        super(Sequence, self).__init__(fixcom=False, fixatoms=None)
        self.input = input
        self.positions = None

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):

        super(Sequence, self).bind(ens, beads, nm, cell, bforce, prng, omaker)

        # Raises error for nbeads not equal to 1.
        if self.beads.nbeads > 1:
            raise ValueError(
                "Calculation not possible for number of beads greater than one."
            )
        self.ens = ens

    def step(self, step=None):
        """Executes one step."""
        if self.positions is None :
            self.positions = read(self.input,index=":")
        if step >= len(self.positions) :
            softexit.trigger(
                status="success",
                message="Finished sequence of configurations. Exiting simulation",
            )
        else :
            q = self.positions[step].positions.flatten()
            self.beads.q.set( q )
            fake = dstrip(self.ens.econs)
        pass
