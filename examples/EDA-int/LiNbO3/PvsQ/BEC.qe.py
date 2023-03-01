"""
   Purpose: Calculation of Born efefctive charges.
   Usage : Type 'python3 BEC.py --help' for all available options
   Author : Alaa Akkoush (June 2021)
"""

# -------------------Libraries------------------------------#
from argparse import ArgumentParser,ArgumentTypeError
import numpy as np
import os, sys
import time
from numpy import float64, zeros
from ase.io import read,write
from copy import copy,deepcopy
import pandas as pd

# constants
C = 1.6021766e-19  # in coulomb

def make_folder(folder):
    if not os.path.exists(folder):
        print("\t\tcreating folder '%s'"%(folder))
        os.mkdir(folder)

def shift(xyz,delta):
    if xyz=="x":
        return np.asarray([delta,0,0])
    elif xyz == "y":
        return np.asarray([0,delta,0])
    elif xyz == "z":
        return np.asarray([0,0,delta])
    else :
        raise ValueError("Wrong direction")

# def is_complete(file,show):
#     if os.path.exists(file):
#         with open(file) as f:
#             lines = f.readlines()
#             if np.any([ "Have a nice day." in lines[-i] for i in range(5) ]):
#                 if show:
#                     print("\t\tFHI-aims calculation is complete")
#                 return True
#             else:
#                 if show:
#                     print("\t\tFHI-aims calculation is not complete")
#                 #sys.exit(1)
#                 return False
#     else :
#         if show:
#             print("\t\tfile '%s' does not exist"%(file))
#             return False
#
# def postpro(file,show=True):
#     """Function to read outputs"""
#     #folder = get_folder(atom,xyz,dn)
#     p = None
#     volume = None
#     if is_complete(file,show):
#         with open(file) as f:
#             lines = f.readlines()
#             for line in lines:
#                 if line.rfind("| Cartesian Polarization ") != -1:
#                     p = float64(split_line(line)[-3:])  #
#                 if line.rfind("| Unit cell volume ") != -1:
#                     volume = float(split_line(line)[-2])
#             return p, volume
#     else :
#         return None,None
#
# def split_line(lines):
#     """Split input line"""
#     line_array = np.array(lines.strip().split(" "))
#     line_vals = line_array[line_array != ""]
#     return line_vals
#
# def get_folder(atom,xyz,dn):
#     return "BEC-I=%d-c=%s-d=%s"%(atom,xyz,dn)
#
# # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise ArgumentTypeError('Boolean value expected.')

def main():
    """main routine"""

    parser = ArgumentParser(description="BEC calculation with Quantum ESPRESSO")
    # parser.add_argument(
    #     "-x", "--executable", action="store",
    #     help="path to FHI-aims binary", default="/home/elia/Google-Drive/google-personal/q-e/PW/src/pw.x"
    # )
    # parser.add_argument(
    #     "-s",
    #     "--suffix",
    #     action="store",
    #     help="Call suffix for binary e.g. 'mpirun -np 12 '",
    #     default="mpirun -n 8",
    # )
    parser.add_argument(
        "-d", "--delta", action="store", type=float,
        help="displacement", default=0.1
    )
    # parser.add_argument(
    #     "-r", "--run", action="store", type=str2bool,
    #     help="perform DFT calculations", default=True
    # )
    # parser.add_argument(
    #     "-pp", "--postprocessing", action="store", type=str2bool,
    #     help="perform post-processing", default=False
    # )
    # parser.add_argument(
    #     "-g", "--geofile", action="store", type=str,
    #     help="original geometry file", default='results-ok/LiNbO3.nk=2.displaced.occ=fixed.n=0.scf.in'
    # )
    # parser.add_argument(
    #     "-c", "--controlfile", action="store", type=str,
    #     help="control file", default="control.in"
    # )
    # parser.add_argument(
    #     "-ps", "--pertspecies", action="store", type=str2bool,
    #     help="perturb all atoms of the same species", default=False
    # )
    # parser.add_argument(
    #     "-rs", "--restartsuffix", action="store", type=str,
    #     help="restart suffix", default="FHI-aims.restart"
    # )
    # parser.add_argument(
    #     "-p", "--polout", action="store", type=str,
    #     help="csv output file for polarization", default="pol.csv"
    # )
    # parser.add_argument(
    #     "-bec", "--becout", action="store", type=str,
    #     help="csv output file for BEC tensors", default="BEC.csv"
    # )
    # parser.add_argument(
    #     "-o", "--original", action="store", type=str,
    #     help="original output file with no displacement", default="FHI-aims.out"
    # )
    # parser.add_argument(
    #     "-w", "--overwrite", action="store", type=bool,
    #     help="overwrite previous calculations", default=False
    # )

    options = parser.parse_args()

    #options.qe_run = options.suffix + " " + options.executable
    #options.qeout = "qe-BEC.out"

    for config in range(0,6):
        make_folder("BEC")
        folder = "BEC/conf-n={:d}".format(config)
        make_folder(folder)

        data = read("{:s}/{:s}.n={:d}.scf.in".format("results-ok","LiNbO3.nk=2.displaced.occ=fixed",config))
        temp_file = "{:s}/{:s}".format(folder,"original.xyz")
        write(temp_file,data,format="xyz")

        for atom in [0,2,4]:
            for xyz in [ "x","y","z" ]:
                print("\t atom={:d} | dir='{:s}'".format(atom,xyz))
                # if options.pertspecies :
                #     raise ValueError("Not yet implemented")
                #     # S2Imap  = data.symbols.indices() # Species to Index map
                #     # Species = data.get_chemical_symbols()
                # else :
                newdata = copy(data)
                newdata.positions[atom,:] += shift(xyz,options.delta)
                temp_file = "{:s}/BEC.atom={:d}.dir={:s}.xyz".format(folder,atom,xyz)
                write(temp_file,newdata,format="xyz")

                make_folder("{:s}/results".format(folder))
                make_folder("{:s}/outdir".format(folder))

                os.system("cp raven.sh {:s}/.".format(folder))
                os.system("cp scf.sh {:s}/.".format(folder))
                os.system("cp var.sh {:s}/.".format(folder))
                os.system("cp raven.BEC.sh {:s}/.".format(folder))
                os.system("cp scf.BEC.sh {:s}/.".format(folder))

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()
