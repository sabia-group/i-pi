
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


def print_cell(cell,tab="\t\t"):
    string = tab+"{:14s} {:1s} {:^10s} {:^10s} {:^10s}".format('','','x','y','z')
    for i in range(3):
        string += "\n"+tab+"{:14s} {:1d} : {:>10.6f} {:>10.6f} {:>10.6f}".format('lattice vector',i+1,cell[i,0],cell[i,1],cell[i,2])
    return string

def main():
    """main routine"""

    parser = ArgumentParser(description="Rotate vectors (and tensors) from a generic matrix to an upper triangular cell matrix."+\
        "\nThe output quantity are saved in the 'rotated' folder with the same original file name.")
    parser.add_argument(
        "-in", "--input", action="store", type=str,
        help="input file, FHI-aims formatted", default="geometry.in"
    )
    parser.add_argument(
        "-v", "--vector", action="store", type=str,
        help="file containing the vectors (positions) to be rotated", default=None
    )
    parser.add_argument(
        "-fmt", "--format", action="store", type=str,
        help="format of the file containing the vectors (positions) to be rotated. Examples: espresso-in, aims", default='espresso-in'
    )
    parser.add_argument(
        "-t", "--tensor", action="store", type=str,
        help="file containing the tensor (BEC) to be rotated (csv formatted)", default=None
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
        "-cs", "--cellsep", action="store", type=str,
        help="separator for the cell (output) file", default="\t"
    )   
    parser.add_argument(
        "-or", "--outrot", action="store", type=str,
        help="output file for the rotation matrix", default="rotation.csv"
    ) 
    parser.add_argument(
        "-rs", "--rotsep", action="store", type=str,
        help="separator for the rotation matrix file", default="\t"
    )  
    parser.add_argument(
        "-vs", "--vecsep", action="store", type=str,
        help="separator for the vectors file", default="\t"
    ) 
    parser.add_argument(
        "-ts", "--tensep", action="store", type=str,
        help="separator for the tensor file", default="\t"
    )    
    
    options = parser.parse_args()

    if options.folder == '.' :
        raise ValueError("\n\tPlease specify a different folder (using the flag '-f foldername'), so no file will be overwritten :)")

    # read the input file using ase
    print("\n\treading cell from file '%s'"%(options.input))
    data = read(options.input)

    print("\n\toriginal lattice vectors (matrix 'O'):")
    print(print_cell(data.cell))

    # get the fractional coordinates
    # frac = data.get_scaled_positions()      # fractional coordinates
    # construct the new cell parameters/lattice vectors, automatically created in the upper triangular form
    L = Cell.new(data.cell.cellpar()) # lower triangular
    print("\n\trotated lattice vectors in lower triangular form computed using ase (matrix 'L'):")
    print(print_cell(L))

    transformation = """
    \t\t     | A 0 0 |           | F E D |
    \t\t L = | B C 0 |  -->  R = | 0 C B |
    \t\t     | D E F |           | 0 0 A |"""

    print("\n\ttransforming LOWER triangular cell parameters (matrix 'L') to UPPER triangular cell parameters (matrix 'U'):\n"+transformation)

    U = np.zeros((3,3))
    U[0,0] = L[2,2] # F
    U[1,1] = L[1,1] # C
    U[2,2] = L[0,0] # A

    U[0,1] = L[2,1] # E
    U[0,2] = L[2,0] # D
    U[1,2] = L[1,0] # B

    del L # I do not need it any mode

    print("\n\trotated lattice vectors in upper triangular form (matrix 'U'):")
    print(print_cell(U))

    print("\n\tcomputing the rotation matrix 'R' transforming the original lattice vectors (matrix 'O') to the upper triangular form (matrix 'U')")
    print("\n\t\tU^t = R @ O^t  -->  R = U^t @ O^-1t")
    print("\n\t\tPay attention that the lattice vectors are stored as row vectors!")
    print("\n\t\tWe need to transpose the matrices shown above")
    
    R = np.asarray(U).T @ inv(np.asarray(data.cell).T)

    print("\n\trotation matrix 'R':")
    print(print_cell(R))

    print("\n\tchecking that 'R' is orthogonal: R @ R^t = id")
    print(print_cell(R@R.T))

    if not os.path.exists(options.folder):
        print("\n\tcreating '%s' folder"%(options.folder))
        os.makedirs(options.folder)

    outfile = options.folder+"/"+options.outcell
    print("\n\tsaving rotated cell to '%s'"%(outfile))
    np.savetxt(outfile,np.asarray(U),delimiter=options.cellsep)

    outfile = options.folder+"/"+options.outrot
    print("\n\tsaving roation matrix to '%s'"%(outfile))
    np.savetxt(outfile,np.asarray(R),delimiter=options.rotsep)

    #in_format = None
    if options.vector is not None:
        print("\n\treading file containing vectors: '%s'"%(options.vector))
        if not os.path.splitext(options.vector):
            raise ValueError("%s does not exist"%(options.vector))
        else:
            ext = os.path.splitext(options.vector)[-1][1:]
            if options.format in ["csv","txt"] or ext in ["csv","txt"]:
                positions = np.loadtxt(options.vector,delimiter=options.vecsep)
                if len(positions.shape) == 1:
                    positions = positions.reshape((1,-1))
            else :
                temp = read(options.vector,format=options.format)
                positions = temp.positions
                #in_format = temp.format
            # else:
            #     raise ValueError("'%s' extension not supported"%(ext))

        print("\trotating vectors (saved as row vectors): (R @ V^t)^t")
        positions = (R @ positions.T).T
        outfile = options.folder+"/"+options.vector

        print("\tsaving rotated vectors to '%s'"%(outfile))
        if options.vector == options.input:
            #options.outcell = None
            newdata = copy(data)
            newdata.cell = Cell(U)
            newdata.set_positions(positions,apply_constraint=False)
            write(outfile,newdata,format=options.format)

        else :
            np.savetxt(outfile,positions,delimiter=options.vecsep)

    if options.tensor is not None:
        print("\n\treading file containing tensors: '%s'"%(options.tensor))
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

            print("\trotating tensors 'T': R^t @ T @ R")
            # BEC = R @ BEC @ inv(R)
            BEC = R.T @ BEC @ R
            BEC = BEC.reshape((len(BEC),-1))
                        
            outfile = options.folder+"/"+options.tensor
            print("\tsaving rotated tensors to '%s'"%(outfile))
            np.savetxt(outfile,BEC,delimiter=options.tensep)



    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()