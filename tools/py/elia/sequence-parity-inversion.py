# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

import argparse
import numpy as np
from ase.io import read,write
import os
import re
import matplotlib.pyplot as plt
from functions import get_all_system_permutations, flatten_list
import itertools

description = "Given a nuclear configuration, this script generate its parity-inverted configuration (some atoms could be swapped).\
               Then a sequence of intermediate configuration is generated.\
               This is useful for computing the energy/polarization at varying configuration when the latter spans from one spontaneous polarization ground state\
               to the other. In thiw way it is possible to see the energy barrier to induce a polarization reversal, and see how some properties varyies along this path."

def prepare_parser():

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-q", "--positions", action="store", type=str,
        help="input file with the nuclear positions"#, default=None
    )
    parser.add_argument(
        "-pi", "--parity-inverted", action="store", type=str,
        help="input file with the parity inverted nuclear positions", default=None
    )
    parser.add_argument(
        "-if", "--input-format", action="store", type=str,
        help="input format of the positions files", default=None
    )
    parser.add_argument(
        "-n", "--number", action="store", type=int,
        help="number of intermediate configurations to be generated", default=10
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="output file with the intermediate configurations (xyz format)", default="sequence.xyz"
    )       

    return parser.parse_args()

def get_sequence(A,B,N):
    sequence = [np.zeros(A.shape)]*(N+2)
    # N = 0 -> t=0,1
    # N = 1 -> t=0,0.5,1
    for n in range(N+2):
        t = float(n)/(N+1)
        sequence[n] = A*(1-t) + t*B
    return sequence

def get_dist(sequence):
    A = np.asarray(sequence[:-1])
    B = np.asarray(sequence[1:])
    return np.sum(np.square(A-B))

def main():
    """main routine"""

    ###
    # prepare/read input arguments
    print("\n\tReding input arguments")
    options = prepare_parser()

    file = options.positions
    print("\n\tReding positions from file '{:s}'".format(file))
    data = read(file,format=options.input_format)

    positions = data.positions
    atoms     = np.asarray(data.get_chemical_symbols())
    Na        = len(atoms)


    if options.parity_inverted is None :

        cell      = np.asarray(data.cell).T
        v = cell.sum(axis=1)
        inv_pos = v - positions
        inv_pos[0,:] = - positions[0,:]
        inv_pos -= ( inv_pos[0,:] - positions[0,:])
        # with open("inverted.xyz", "w") as f:                    
        #     f.write(str(Na)+"\n")
        #     f.write("# inverted configuration\n")
        #     pos = inv_pos*1.8897259886
        #     for ii in range(Na):
        #         f.write("{:>2s} {:>20.12e} {:>20.12e} {:>20.12e}\n".format(atoms[ii],*pos[ii,:]))

        # positions -= positions[0,:]
        # with open("start.xyz", "w") as f:                    
        #     f.write(str(Na)+"\n")
        #     f.write("# inverted configuration\n")
        #     pos = positions
        #     for ii in range(Na):
        #         f.write("{:>2s} {:>20.12e} {:>20.12e} {:>20.12e}\n".format(atoms[ii],*pos[ii,:]))

        # exit

        species   = np.unique(atoms)
        index     = {key: list(np.where(atoms == key)[0]) for key in species}

        trial     = get_all_system_permutations(atoms)

        fi = flatten_list(index.values())

        # replace = np.full(len(atoms),np.nan,dtype=int)
        dist = np.zeros(len(trial))
        ipos = np.zeros(inv_pos.shape)
        for n,tr in enumerate(trial):
            ## t = { s:i for i,s in zip(tr,species) }
            # replace[fi] = flatten_list(list(tr))
            #ipos = inv_pos[replace]
            ## print(inv_pos - ipos)

            # replace[fi] = flatten_list(list(tr))
            j = flatten_list(list(tr))
            ipos[fi] = inv_pos[j]

            sequence = get_sequence(positions,ipos,10)
            dist[n] = get_dist(sequence)

        ii = np.argmin(dist)

        # replace[fi] = flatten_list(list(trial[ii]))
        ipos[fi] = inv_pos[flatten_list(list(trial[ii]))]

    else :
        file = options.parity_inverted
        print("\n\tReding parity-inverted positions from file '{:s}'".format(file))
        ipos = read(file,format=options.input_format).positions


    sequence = get_sequence(positions,ipos,options.number)

    
    with open(options.output, "w") as f:
        for i in range(len(sequence)):
            
            f.write(str(Na)+"\n")
            f.write("# configuration {:d}\n".format(i))

            pos = sequence[i]#*1.8897259886
            for ii in range(Na):
                f.write("{:>2s} {:>20.12e} {:>20.12e} {:>20.12e}\n".format(atoms[ii],*pos[ii,:]))
           


        # for n in range(len(sequence)):
        #     write(file,sequence[n],format="xyz")


    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()