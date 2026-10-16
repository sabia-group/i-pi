# -*- coding: utf-8 -*-
import numpy as np
from ipi.utils import units
import os

try:
    from .dummy import Dummy_driver
except ImportError:
    from dummy import Dummy_driver

__DRIVER_NAME__ = "VOTH_OSC"
__DRIVER_CLASS__ = "Oscillating_Voth_driver"

class Oscillating_Voth_driver(Dummy_driver):

    def __init__(self, spline_file="spline_coeffs.dat", *args, **kwargs):

        # --- resolve path relative to this file ---
        pes_dir = os.path.dirname(os.path.abspath(__file__))
        spline_path = os.path.join(pes_dir, spline_file)

        data = np.loadtxt(spline_path, comments="#")

        self.x_knots = data[:, 0]
        self.c0 = data[:, 1]
        self.c1 = data[:, 2]
        self.c2 = data[:, 3]
        self.c3 = data[:, 4]

        self.n_intervals = len(self.x_knots) - 1

        super().__init__(*args, **kwargs)

    def __call__(self, cell, pos):
        x = pos.reshape(-1, 3)[:, 0]  # nur x-Koordinate
        
        i = np.searchsorted(self.x_knots, x, side="right") - 1
        i = np.clip(i, 0, self.n_intervals - 1)
        
        dx = x - self.x_knots[i]
        
        V  = ((self.c0[i] * dx + self.c1[i]) * dx + self.c2[i]) * dx + self.c3[i]
        dV = (3.0 * self.c0[i] * dx + 2.0 * self.c1[i]) * dx + self.c2[i]
        
        pot = np.sum(V)
        
        force = np.zeros_like(pos.reshape(-1, 3))
        force[:, 0] = -dV
        force = force.reshape(-1, 1)
        
        vir = np.zeros((3, 3))
        
        return pot, force, vir, "VOTH_OSC_1D"




