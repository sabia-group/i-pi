"""Interface with e3nn to run machine learning electric dipole"""

import sys
import torch
import json
import numpy as np
from ase.io import read 
from .dummy import Dummy_driver
import importlib

# "args":["-m","e3nn_pol","-a","localhost","-p","0000","-u","-o",
#            "/home/stoccoel/google-personal/miscellaneous/miscellaneous/elia/nn/water/instructions.json,/home/stoccoel/google-personal/miscellaneous/miscellaneous/elia/nn/water/LiNbO3/MD/start.xyz"],

def get_class(module_name, class_name):
    try:
        # Import the module dynamically
        module = importlib.import_module(module_name)
        
        # Get the class from the module
        class_obj = getattr(module, class_name)
        
        # Create an instance of the class
        #instance = class_obj()
        
        return class_obj
    
    except ImportError:
        raise ValueError(f"Module '{module_name}' not found.")
    except AttributeError:
        raise ValueError(f"Class '{class_name}' not found in module '{module_name}'.")
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
    # return None

# # Usage example
# module_name = "my_module"  # Replace with your module's name
# class_name = "MyClass"     # Replace with the class name you want to allocate

# instance = create_instance(module_name, class_name)
# if instance:
#     instance.some_method()  # Call a method on the allocated instance


class e3nn_pol(Dummy_driver):

    def __init__(self, args=None):

        self.error_msg = """The parameters of 'e3nn_pol' are not correctly formatted. They should be two strings, separated by a comma."""

        super().__init__(args)

    def check_arguments(self):
        """Check the arguments required to run the driver

        This loads the potential and atoms template in librascal
        """
        try:
            arglist = self.args.split(",")
        except ValueError:
            sys.exit(self.error_msg)

        if len(arglist) == 2:
            # self.model_file = arglist[0] # file with the torch.nn.Module
            info_file  = arglist[0] # file with some json formatted information
            atoms_file = arglist[1] # file with the atomic species
        else:
            sys.exit(self.error_msg)

        # read instructions file        
        try :
            with open(info_file, "r") as json_file:
                instructions = json.load(json_file)
        except:
            raise ValueError("Error reading the instruction file '{:s}'".format(self.info_file))
        
        # check that 'instructions' has the correct format
        for k in ['kwargs','class','module'] :
            if k not in instructions:
                raise ValueError("Error: the '{:s}' key should be contained in '{:s}'".format(k,self.info_file))

        # wxtract values for the instructions
        kwargs = instructions['kwargs']
        cls    = instructions['class']
        mod    = instructions['module']

        # get the class to be instantiated
        class_obj = get_class(mod,cls)
        
        # instantiate class
        #try :
        self.model = class_obj(**kwargs)
        if not self.model :
            raise ValueError("Error instantiating class '{:s}' from module '{:s}'".format(cls,mod))

        # Store into the model the chemical species that will be used during the simulation.
        # This method is implemented in the parent class iPIinterface.
        self.model.store_chemical_species(atoms_file)

        pass
 

    def __call__(self, cell, pos):
        """Get energies, forces, stresses and extra quantities"""

        # Get vanishing pot, forces and vir
        pot, force, vir, extras = super().__call__(cell,pos)

        dipole = self.model.get(cell,pos,what="dipole")
        bec    = self.model.get(cell,pos,what="BEC")

        # I should check if this shape is okay
        # Axis of bec :
        #   1st: atoms index (0,1,2...)
        #   2nd: atom coordinate (x,y,z)
        #   3rd: polarization direction (x,y,z)
        bec = bec[0].T.reshape((-1,3,3)) 

        # reshape again
        bec = bec.reshape((-1,9)) 


        dipole = dipole[0]

        # I will convert the batch_size to the beads numbers
        # bec -> bec = bec[0] (batch_size=1)
        # bec -> bec = [bec]  (nbeads=1)

        # The same hold for the polarization
        volume = np.linalg.det(cell)
        polarization = dipole / volume

        extras = {
            "BEC" : bec.tolist(),
            "polarization" : polarization.tolist()
        }

        return pot, force, vir, json.dumps(extras)
