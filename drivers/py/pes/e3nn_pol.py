"""Interface with e3nn to run machine learning electric dipole"""

import sys
import torch
import json
import numpy as np
from ase.io import read 
from miscellaneous.elia.nn.get_type_onehot_encoding import symbols2x
from .dummy import Dummy_driver

class e3nn_pol(Dummy_driver):

    model_file : str
    info_file  : str
    atoms_file : str
    model      : torch.nn.Module
    info       : dict

    def __init__(self, args=None):

        self.error_msg = """I will insert an error message"""

        super().__init__(args)

    def check_arguments(self):
        """Check the arguments required to run the driver

        This loads the potential and atoms template in librascal
        """
        try:
            arglist = self.args.split(",")
        except ValueError:
            sys.exit(self.error_msg)

        if len(arglist) == 3:
            self.model_file = arglist[0] # file with the torch.nn.Module
            self.info_file  = arglist[1] # file with some json formatted information
            self.atoms_file = arglist[2] # file with the atomic species
        else:
            sys.exit(self.error_msg)

        self.model = torch.load(self.model_file)

        with open(self.info_file) as f:
            self.info = json.load(f)

        atoms = read(self.atoms_file)  
        symbols = atoms.get_chemical_symbols()
        self.x = symbols2x(symbols)

        pass
 

    def __call__(self, cell, pos):
        """Get energies, forces, and stresses from the librascal model"""

        pot, force, vir, extras= super().__call__(cell,pos)

        return pot, force, vir, extras
