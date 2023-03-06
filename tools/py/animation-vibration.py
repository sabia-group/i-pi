
"""Functions used to read input configurations and print trajectories
in the XYZ format.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

from argparse import ArgumentParser

import os

import numpy as np

import ipi.utils.mathtools as mt
from ipi.utils.depend import dstrip
from ipi.utils.units import Elements
from ipi.utils.units import Elements
from ipi.utils.io.backends import io_xyz
from ase.io import read,write
from ase.cell import Cell
from copy import copy
from ipi.utils.messages import verbosity, warning, info
from numpy.linalg import inv
import tempfile
import pandas as pd


def print_cell(cell,tab="\t\t"):
    string = tab+"{:14s} {:1s} {:^10s} {:^10s} {:^10s}".format('','','x','y','z')
    for i in range(3):
        string += "\n"+tab+"{:14s} {:1d} : {:>10.6f} {:>10.6f} {:>10.6f}".format('lattice vector',i+1,cell[i,0],cell[i,1],cell[i,2])
    return string

def prepare_parser():

    parser = ArgumentParser(description="Prepare the xyz files for a vibrational mode animation.")
    parser.add_argument(
        "-i", "--input", action="store", type=str,
        help="input file with the relaxed position, around which the vibrational modes have been computed", default=None
    )
    parser.add_argument(
        "-e", "--eigenvec", action="store", type=str,
        help="file containing the eigen-vectors computed by i-PI", default=None
    )
    parser.add_argument(
        "-r", "--radix", action="store", type=str,
        help="radix name of the vibrational analysis output file", default=None
    )
    parser.add_argument(
        "-m", "--mode", action="store", type=int,
        help="index of the mode to consider (-1 means all)", default=-1
    )
    parser.add_argument(
        "-d", "--displacement", action="store", type=float,
        help="maximum dispacement to consider", default=0.5
    )
    # parser.add_argument(
    #     "-u", "--unit", action="store", type=str,
    #     help="measure unit of the displacement (A,bohr)", default="A"
    # ) 
    parser.add_argument(
        "-n", "--number", action="store", type=int,
        help="number of configurations to be computed", default=30
    ) 
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="radix of the output file (xyz) with all the configurations", default=None
    ) 
    parser.add_argument(
        "-od", "--outdir", action="store", type=str,
        help="output directory where a file (xyz) for each configuration will be saved", default='animation'
    )    
       
    options = parser.parse_args()

    return options

def main():
    """main routine"""

    options = prepare_parser()

    # read the input file using ase
    print("\n\treading cell from file '%s'"%(options.input))
    data = read(options.input)

    #print("\n\trelaxed positions")
    #print(data.positions)

    print("\n\treading cell from file '%s'"%(options.input))
    eigenvec = np.loadtxt(options.eigenvec,skiprows=1)
    #print("\n\teigenvectors")
    #print(eigen)

    if options.radix is not None :

        print("\treading dynamical matrix 'D'")
        dynmat = np.loadtxt(options.radix+".phonons.dynmat",skiprows=1)

        print("\treading eigenvalues matrix 'E'")
        eigval = np.loadtxt(options.radix+".phonons.eigval",skiprows=1)

        print("\treading eigenvector matrix 'V'")
        eigvec = np.loadtxt(options.radix+".phonons.eigvec",skiprows=1)

        eigval = np.diag(eigval)

        print("\tchecking that D@V = E@V")
        res = np.sqrt(np.square(dynmat @ eigvec - eigval @ eigvec).sum())
        print("\t\t | D@V - E@V | = {:>10.6f}".format(res))

        del dynmat, eigvec, eigval, res

    if options.number <= 0 :
        raise ValueError("'numbers' (-n, --numbers) has to be greater than 0")
    if options.displacement <= 0 :
        raise ValueError("'displacement' (-d, --displacement) has to be greater than 0.0")
    if options.mode < 0 and options.mode != -1 :
        raise ValueError("'mode' (-m, --mode) has to be greater than or equal to 0 (-1 means all modes)")
    if options.mode > len(eigenvec) :
        raise ValueError("'mode' (-m, --mode) has to be smaller than the number of vibratinal modes")
    if options.outdir is None and options.output is None:
        raise warning("outdir (-od,--outdir) and output (-o,--output) are both None: no output file will be produced")

    folder = tempfile.TemporaryDirectory().name
    print("\n\tcreating a temporary folder with name '%s'\n"%(folder))
    #os.mkdir(os.getcwd()+"/tmp")
    os.mkdir(folder)    
    positions = data.positions.copy()
    if options.mode == -1 :
        modes = np.arange(len(eigenvec))
    else :
        modes = [options.mode]

    mode_files = pd.DataFrame(columns=np.arange(options.number),index=np.arange(len(modes)))
    for nm,mode in enumerate(modes):
        print("\tcomputing configurations for mode %d"%(mode))
        vec = eigenvec[mode,:]
        for n in range(options.number):
            phase = float(n) / float(options.number)
            #print("\t\tn={:>3d} phase={:>10.6f}".format(n,phase))
            displacement = options.displacement * np.sin( phase*2*np.pi ) * vec
            data.positions = positions + displacement.reshape((-1,3))
            file = "{:s}/{:s}.m={:d}.c={:d}.xyz".format(folder,os.path.splitext(options.input)[-2],mode,n)
            mode_files.at[nm,n] = file
            print("\t\twriting configuration %i to file %s"%(n,file))
            write(file,data,format="xyz")

    if options.outdir is not None :
        if not os.path.exists(options.outdir):
            print("\n\tcreating folder '%s'\n"%(options.outdir))
            os.mkdir(options.outdir)
        print("\n\tcopying files from the temporary folder to '%s'\n"%(options.outdir))
        #os.system("cp %s/* %s/."%(folder,options.outdir))
    
    if options.output is not None :

        for nm,mode in enumerate(modes):
            file = "{:s}/{:s}.m={:d}.xyz".format(options.outdir,options.output,mode)
            print("\tconcatenating all files to '%s'"%(file))

            filenames = list(mode_files.iloc[nm,:])
            # if os.path.exists(options.output):
            #     os.remove(options.output)

            with open(file, 'w') as outfile:
                for fname in filenames:
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)
            
    print("\n\tremoving temporary folder")
    #os.system("rm -rf %s"%(os.getcwd()+"/tmp"))
    os.system("rm -rf %s"%(folder))

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()