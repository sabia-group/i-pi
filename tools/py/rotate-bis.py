#!/usr/bin/env python3
"""Functions used to read input configurations and print trajectories
in the XYZ format.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

from argparse import ArgumentParser

import os
import ast

import numpy as np

from ase.io import read
#import inspect
import sys
  
# appending the parent directory path

# importing the methods
  

txt_files = ["csv","txt","tab"]

def print_cell(cell,tab="\t\t"):
    string = tab+"{:14s} {:1s} {:^10s} {:^10s} {:^10s}".format('','','x','y','z')
    for i in range(3):
        string += "\n"+tab+"{:14s} {:1d} : {:>10.6f} {:>10.6f} {:>10.6f}".format('lattice vector',i+1,cell[i,0],cell[i,1],cell[i,2])
    return string

def prepare_parser():

    parser = ArgumentParser(description="Rotate vectors (and tensors) from a generic matrix to an upper triangular cell matrix."+\
        "\nThe output quantity are saved in the 'rotated' folder with the same original file name.")
    
    parser.add_argument(
        "-p", "--path", action="store", type=str,
        help="i-PI path", default="/home/elia/Google-Drive/google-personal/i-pi-sabia/"
    )

    parser.add_argument(
        "-c", "--cell", action="store", type=str,
        help="file containing the cell patameters", default="geometry.in"
    )
    parser.add_argument(
        "-cf", "--cell-format", action="store", type=str,
        help="format of the file containing the cell", default=None
    )

    parser.add_argument(
        "-v", "--vectors", action="store", type=str,
        help="file containing the vectors (positions) to be rotated", default=None
    )
    parser.add_argument(
        "-vf", "--vectors-format", action="store", type=str,
        help="format of the file containing the vectors (positions)", default=None
    )
    parser.add_argument(
        "-vs", "--vectors-separator", action="store", type=str,
        help="separator for the vectors file", default=" "
    ) 
    

    parser.add_argument(
        "-t", "--tensors", action="store", type=str,
        help="file containing the ([1,1] rank) tensors to be rotated (in txt format)", default=None
    )
    parser.add_argument(
        "-tf", "--tensors-format", action="store", type=str,
        help="format of the file containing the the tensors", default=None
    )
    parser.add_argument(
        "-ts", "--tensors-separator", action="store", type=str,
        help="separator for the tensors file", default=" "
    )  

    parser.add_argument(
        "-vo", "--vectors-original-basis", action="store", type=str,
        help="original basis of the vectors", default=None
    )
    parser.add_argument(
        "-vd", "--vectors-destination-basis", action="store", type=str,
        help="destination basis of the vectors", default=None
    )     

    parser.add_argument(
        "-to", "--tensors-original-basis", action="store", type=list, nargs="*",
        help="original basis of the tensors", default=None
    )
    parser.add_argument(
        "-td", "--tensors-destination-basis", action="store", type=list, nargs="*",
        help="destination basis of the tensors", default=None
    )     


    parser.add_argument(
        "-f", "--folder", action="store", type=str,
        help="output folder", default="rotated"
    ) 

    options = parser.parse_args()

    options.tensors_original_basis = ast.literal_eval(''.join(options.tensors_original_basis[0]))
    options.tensors_destination_basis = ast.literal_eval(''.join(options.tensors_destination_basis[0]))
    
    return options

def check_values(options):
    if options.folder == '.' :
        raise ValueError("\n\tPlease specify a different folder (using the flag '-f foldername'), so no file will be overwritten :)")

def main():
    """main routine"""

    options = prepare_parser() 
    if  options.path is not None :
        sys.path.append(options.path)
    from ipi.engine.cell import Cell
    check_values(options)

    # read the input file using ase
    print("\n\treading cell from file '%s'"%(options.cell))
    data = read(options.cell,format=options.cell_format)

    print("\n\toriginal lattice vectors")
    print(print_cell(data.cell))

    print("\n\tallocating 'Cell' class")
    cell = Cell(np.asarray(data.cell).T)


    #in_format = None
    if options.vectors is not None:
        print("\n\treading file containing vectors: '%s'"%(options.vectors))
        if not os.path.splitext(options.vectors):
            raise ValueError("%s does not exist"%(options.vectors))
        else:
            ext = os.path.splitext(options.vectors)[-1][1:]
            try :
                if ext in txt_files:
                    positions = np.loadtxt(options.vectors,delimiter=options.vectors_separator)
                    if len(positions.shape) == 1:
                        positions = positions.reshape((1,-1))
                else :
                    positions = read(options.vectors,format=options.vectors_format).positions
            except: 
                raise ValueError("error reading file '{:s}'".format(options.vectors))

        print("\trotating vectors")
        positions = cell.change_basis(  v=positions.T,\
                                        orig=options.vectors_original_basis,\
                                        dest=options.vectors_destination_basis).T
        
        outfile = options.folder+"/"+options.vectors
        print("\tsaving rotated vectors to '%s'"%(outfile))
        np.savetxt(outfile,positions,delimiter=options.vectors_separator)

    if options.tensors is not None:
        print("\n\treading file containing tensors: '%s'"%(options.tensors))
        if not os.path.splitext(options.tensors):
            raise ValueError("%s does not exist"%(options.tensors))
        else:
            ext = os.path.splitext(options.tensors)[-1][1:]
            if ext in txt_files:
                tBEC = np.loadtxt(options.tensors,delimiter=options.tensors_separator) # temporary BEC
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
            for i in range(len(BEC)):
                BEC[i] = cell.change_basis(  M=BEC[i],\
                                            orig=options.tensors_original_basis,\
                                            dest=options.tensors_destination_basis)
        
            #BEC = R.T @ BEC @ R
            BEC = BEC.reshape((len(BEC),-1))
                        
            outfile = options.folder+"/"+options.tensors
            print("\tsaving rotated tensors to '%s'"%(outfile))
            np.savetxt(outfile,BEC,delimiter=options.tensors_separator)



    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()
