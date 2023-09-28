"""Interface with e3nn to run machine learning electric dipole"""

import sys
import json5 as json
import numpy as np
from ase.io import read 
from py.pes.dummy import Dummy_driver
import importlib
from miscellaneous.elia.nn.functions import get_function, bash_as_function
from miscellaneous.elia.functions import add_default


class bash_script(Dummy_driver):

    def __init__(self, args=None):
        self.error_msg = """The parameters of 'bash_script' are not correctly formatted. \
            They should a stringa, containing the filename of a json file."""
        super().__init__(args)

    def check_arguments(self):
        """Check the arguments required to run the driver

        This loads the potential and atoms template in librascal
        """
        try:
            arglist = self.args.split(",")
        except ValueError:
            sys.exit(self.error_msg)

        if len(arglist) == 1:
            infile = arglist[0] # file with some json formatted information
        else:
            sys.exit(self.error_msg) # to be modified


        # # read instructions file        
        try :
            with open(infile, "r") as json_file:
                instructions = json.load(json_file)
        except:
            raise ValueError("Error reading the instruction file '{:s}'".format(infile))
        
        # example of 'instructions'
        # {
        #    "run_once" : {   // python function
        #         "module" : "module_name" ,
        #         "function" : "function_name"
        #     },
        #     "preprocess" : {   // python function
        #         "module" : "module_name" ,
        #         "function" : "function_name"
        #     },
        #     "process" : { // bash script
        #         "path" : "filepath"
        #     },
        #     "postprocess" : { // python function
        #         "module" : "module_name" ,
        #         "function" : "function_name"
        #     }
        # }        
        keys = ["preprocess","process","postprocess"]
        for k in keys:
            if k not in instructions.keys():
                raise ValueError("missing key '{:s}' in file '{:s}'".format(k,infile))
            
        self.opts = {"obj":self}
        if "opts" in instructions.keys():
            opts = instructions["opts"]
            # priority to the file
            self.opts = add_default(opts,self.opts)
        
        if "run_once" in instructions.keys():
            run_once  = get_function(instructions["run_once"]["module"], instructions["run_once"]["function"])
            # usually read the chemical species
            self = run_once(self,opts=self.opts)

        self.preprocess  = get_function(instructions["preprocess"]["module"], instructions["preprocess"]["function"])
        self.postprocess = get_function(instructions["postprocess"]["module"],instructions["postprocess"]["function"])
        self.process     = bash_as_function(instructions["process"]["path"])

        

        pass
 
    def __call__(self, cell, pos):
        """Get energies, forces, stresses and extra quantities"""

        # Get vanishing pot, forces and vir
        pot, force, vir, extras = super().__call__(cell,pos)
        
        self.preprocess(cell,pos,opts=self.opts)
        self.process()
        output = self.postprocess(opts=self.opts)

        if "pot" in output:
            pot = output["pot"]
        if "force" in output:
            force = output["force"]
        if "vir" in output:
            pot = output["vir"]
        if "extras" in output:
            extras = output["extras"]

        return pot, force, vir, json.dumps(extras)
