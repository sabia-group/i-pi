
"""Functions used to read input configurations and print trajectories
in the XYZ format.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

from argparse import ArgumentParser,ArgumentTypeError

import sys, re, os

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


def print_cell(cell,tab="\t\t"):
    string = tab+"{:14s} {:1s} {:^10s} {:^10s} {:^10s}".format('','','x','y','z')
    for i in range(3):
        string += "\n"+tab+"{:14s} {:1d} : {:>10.6f} {:>10.6f} {:>10.6f}".format('lattice vector',i+1,cell[i,0],cell[i,1],cell[i,2])
    return string

def main():
    """main routine"""

    parser = ArgumentParser(description="Rotate the vectors and tensors from generic to trianglar cell parameters."+\
        "\nThe output quantity are saved in the 'rotated' folder wth the same original file name.")
    parser.add_argument(
        "-in", "--input", action="store", type=str,
        help="input file, FHI-aims formatted", default="geometry.in"
    )
    parser.add_argument(
        "-cs", "--cellsep", action="store", type=str,
        help="separator for the cell (output) file", default=","
    )  
    parser.add_argument(
        "-f", "--folder", action="store", type=str,
        help="output folder", default="rotated"
    )  
    parser.add_argument(
        "-oc", "--outcell", action="store", type=str,
        help="output file for the cell", default="cell.csv"
    )  
    parser.add_argument(
        "-v", "--vector", action="store", type=str,
        help="file containing the vectors (positions) to be rotated (csv or FHI-aims formatted)", default=None
    )
    parser.add_argument(
        "-vs", "--vecsep", action="store", type=str,
        help="separator for the vectors file", default=","
    )   
    parser.add_argument(
        "-t", "--tensor", action="store", type=str,
        help="file containing the tensor (BEC) to be rotated (csv formatted)", default=None
    )
    parser.add_argument(
        "-ts", "--tensep", action="store", type=str,
        help="separator for the tensor file", default=","
    )    
    # parser.add_argument(
    #     "-s",
    #     "--suffix",
    #     action="store",
    #     help="Call suffix for binary e.g. 'mpirun -np 12 '",
    #     default="",
    # )
    # parser.add_argument(
    #     "-d",
    #     "--delta",
    #     action="store",
    #     type=float,
    #     nargs=2,
    #     dest="delta",
    #     help="finite difference poles, defualt is [0.01]",
    #     default=[0.01],
    # )
    # parser.add_argument(
    #     "-r", "--run", action="store", type=str2bool,
    #     help="perform DFT calculations", default=True
    # )
    # parser.add_argument(
    #     "-pp", "--postprocessing", action="store", type=str2bool,
    #     help="perform post-processing", default=True
    # )
    # parser.add_argument(
    #     "-g", "--geofile", action="store", type=str,
    #     help="original geometry file", default='geometry.in'
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

    options = parser.parse_args()

    # read the input file using ase
    print("\n\treading cell from file '%s'"%(options.input))
    data = read(options.input)

    print("\n\toriginal lattice vectors")
    print(print_cell(data.cell))

    # get the fractional coordinates
    #frac = data.get_scaled_positions()      # fractional coordinates
    # construct the new cell parameters/lattice vectors, automatically created in the upper triangular form
    newcell = Cell.new(data.cell.cellpar()) # upper triangular

    print("\n\trotated lattice vectors (in triangular form):")
    print(print_cell(newcell))

    R = np.asarray(newcell).T @ inv(np.asarray(data.cell).T)

    print("\n\trotation matrix:")
    print(print_cell(R.T))

    outfile = options.folder+"/"+options.outcell
    print("\n\tsaving rotated cell to '%s'"%(outfile))
    np.savetxt(outfile,np.asarray(newcell),delimiter=options.cellsep)

    if options.vector is not None:
        print("\n\treading vector file '%s'"%(options.vector))
        if not os.path.splitext(options.vector):
            raise ValueError("%s does not exist"%(options.vector))
        else:
            ext = os.path.splitext(options.vector)[-1][1:]
            if ext == "in":
                positions = read(options.vector).positions
            elif ext in ["csv","txt"]:
                positions = np.loadtxt(options.vector,delimiter=options.vecsep)
                if len(positions.shape) == 1:
                    positions = positions.reshape((1,-1))
            else:
                raise ValueError("'%s' extension not supported"%(ext))

        print("\trotating vectors")
        positions = (R @ positions.T).T
        outfile = options.folder+"/"+options.vector

        print("\tsaving rotated vectors to '%s'"%(outfile))
        if options.vector == options.input:
            options.outcell = None
            if ext != "in":
                raise ValueError("wrong extensions")

            newdata = copy(data)
            newdata.cell = Cell(newcell)
            newdata.set_positions(positions,apply_constraint=False)
            if not os.path.exists(options.folder):
                print("\n\tcreating '%s' folder"%(options.folder))
                os.makedirs(options.folder)
            write(outfile,newdata)

        else :
            np.savetxt(outfile,positions,delimiter=options.vecsep)

    # if options.outcell is not None :
    #     outfile = options.folder+"/"+options.outcell
    #     print("\n\tsaving rotated cell to '%s'"%(outfile))
    #     np.savetxt(outfile,np.asarray(newcell),delimiter=options.cellsep)

    if options.tensor is not None:
        print("\n\treading tensor file '%s'"%(options.tensor))
        if not os.path.splitext(options.tensor):
            raise ValueError("%s does not exist"%(options.tensor))
        else:
            ext = os.path.splitext(options.tensor)[-1][1:]
            if ext in ["csv","txt"]:
                tBEC = np.loadtxt(options.tensor,delimiter=options.tensep) # temporary BEC
                n,l = tBEC.shape
                if l not in [3,9]:
                    raise ValueError("BEC with a wrong size")
                BEC = np.zeros((n,3,3))
                if l == 9 :
                    for i in range(n):
                        BEC[i,:,:] = tBEC[i,:].reshape((3,3))
                elif l == 3 :
                    temp = np.zeros((3,3))
                    np.fill_diagonal(temp,tBEC[i,:])
                    BEC[i,:,:] = temp
                else :
                    raise ValueError("BEC with a wrong size: coding error!")
            else:
                raise ValueError("'%s' extension not supported"%(ext))

            print("\trotating tensors")
            BEC = R @ BEC @ inv(R)
            BEC = BEC.reshape((len(BEC),-1))
                        
            outfile = options.folder+"/"+options.tensor
            print("\tsaving rotated tensors to '%s'"%(outfile))
            np.savetxt(outfile,BEC,delimiter=options.tensep)



    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()