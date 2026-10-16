# -*- coding: utf-8 -*-
import numpy as np
from ipi.utils import units

try:
    from .dummy import Dummy_driver
except ImportError:
    from dummy import Dummy_driver

__DRIVER_NAME__ = "VOTH_DW"
__DRIVER_CLASS__ = "DoubleWellVoth_driver"

class DoubleWellVoth_driver(Dummy_driver):

    def __init__(self, a=None, b=None, c=None, *args, **kwargs):

        # Default parameters (Hartree/bohr^n)
        if a is None:
            self.a = -0.03123777
            self.b =  0.00023614737
            self.c =  0.031240952
        else:
            self.a = a
            self.b = b
            self.c = c

        super().__init__(*args, **kwargs)

    '''def __call__(self, cell, pos):
        """
        pos has shape (natoms*1,) because DOFs 1 and 2 are fixed.
        """

        # x coordinate (1D)
        x = pos.reshape(-1)

        # ---- VOTH double-well potential ----
        a, b, c = self.a, self.b, self.c
        V = a*x**2 + b*x**3 + c*x**4
        pot = np.sum(V)

        # ---- Force in 1D ----
        Fx = -2*a*x - 3*b*x**2 - 4*c*x**3
        force = Fx.reshape(-1, 1)  # 1D force

        # ---- Virial MUST BE 3x3 ----
        vir = np.zeros((3,3))

        return pot, force, vir, "VothDW_1D"'''
    
    def __call__(self, cell, pos):
        x = pos.reshape(-1, 3)[:, 0]  # nur x-Koordinate

        a, b, c = self.a, self.b, self.c
        V = a*x**2 + b*x**3 + c*x**4
        pot = np.sum(V)

        Fx = -2*a*x - 3*b*x**2 - 4*c*x**3
        force = np.zeros_like(pos.reshape(-1, 3))
        force[:, 0] = Fx
        force = force.reshape(-1, 1)

        vir = np.zeros((3, 3))

        return pot, force, vir, "VothDW_1D"





