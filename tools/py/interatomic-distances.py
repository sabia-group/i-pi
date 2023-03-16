
"""Functions used to read input configurations and print trajectories
in the XYZ format.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

from argparse import ArgumentParser

import os

import numpy as np
from numpy.linalg import norm

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
import ast

def prepare_parser():

    parser = ArgumentParser(description="Compute the interatomic distances during a molecular dynamics simulation.")
    parser.add_argument(
        "-i", "--input", action="store", type=str,
        help="input file (xyz format) with all nuclear positions", default=None
    )
    parser.add_argument(
        "-u", "--units", action="store", type=str,
        help="units of the input file", default="a.u."
    )
    parser.add_argument(
        "-c", "--couples", action="store", type=list, nargs="*",
        help="couples of atoms whose distance has to be computed", default=None
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="name of the output file", default="distances.csv"
    )   
    parser.add_argument(
        "-p", "--plot", action="store", type=str,
        help="name of the output file for plot", default="distances.png"
    )  
       
    return parser.parse_args()

def get_all_couples(N):
    out = list()
    for i in range(N):
        for j in range(i+1,N):
            out.append([i,j])
    return out

def check_couples(couples,N):
    temp = ast.literal_eval(''.join(couples[0]))
    arr = np.asarray(temp)
    if len(arr.shape) != 2:
        raise ValueError("'couples' are provided with a wrong shape")

    if np.any(arr[:,0] < 0)  or np.any(arr[:,0] >= N) or np.any(arr[:,1] < 0) or np.any(arr[:,1] >= N) :
        raise ValueError("some indexes in 'couples' are out of bound")

    return arr

def get_c(i,j):
    return "[%d,%d]"%(i,j)

def initialization(options):

    # read the input file using ase
    print("\n\tpositions from file '%s'"%(options.input))
    options.data = read(options.input)
    
    print("\tpreparing/checking couples of atoms")
    N = len(options.data.positions)
    if options.couples is None:
        options.couples = get_all_couples(N)
    else :
        options.couples = check_couples(options.couples,N)
    
    options.couples = np.asarray(options.couples)

    return options

def main():
    """main routine"""

    options = prepare_parser()
    options = initialization(options)

    options.configurations = read(options.input,index=':')
    N = len(options.configurations)

    columns = [ get_c(i,j) for i,j in zip(options.couples[:,0],options.couples[:,1])]
    distances = pd.DataFrame(columns=columns,index=np.arange(N))

    for n in range(N):
        for i,j in zip(options.couples[:,0],options.couples[:,1]):
            c = get_c(i,j)
            A = options.configurations[n].positions[i,:] # position of the atom A
            B = options.configurations[n].positions[j,:] # position of the atom B
            distances.at[n,c] = norm(A-B)

    print("\t\twriting distances to file %s"%(options.output))
    distances.to_csv(options.output,index=False)

    print("\n\tJob done :)\n")

def plot():

    import matplotlib.pyplot as plt

    options = prepare_parser()
    options = initialization(options)
    distances = pd.read_csv(options.output)

    fig = plt.figure()
    for c in options.couples:
        c = get_c(*c)
        y = distances[c]-np.mean(distances[c])
        plt.plot(y,label=c)

    plt.legend()
    plt.grid()
    plt.savefig(options.plot)
       
    print("\n\tJob done :)\n")


if __name__ == "__main__":
    main()
    plot()