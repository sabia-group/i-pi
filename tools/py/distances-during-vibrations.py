
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
import ast


def print_cell(cell,tab="\t\t"):
    string = tab+"{:14s} {:1s} {:^10s} {:^10s} {:^10s}".format('','','x','y','z')
    for i in range(3):
        string += "\n"+tab+"{:14s} {:1d} : {:>10.6f} {:>10.6f} {:>10.6f}".format('lattice vector',i+1,cell[i,0],cell[i,1],cell[i,2])
    return string

def prepare_parser():

    parser = ArgumentParser(description="Compute the distance of the ions during a vibrational mode.")
    parser.add_argument(
        "-i", "--input", action="store", type=str,
        help="input file with the relaxed position, around which the virbational modes have been computed", default=None
    )
    parser.add_argument(
        "-e", "--eigenvec", action="store", type=str,
        help="file containing the eigen-vectors computed by i-PI", default=None
    )
    # parser.add_argument(
    #     "-r", "--radix", action="store", type=str,
    #     help="radix name of the vibrational analysis output file", default=None
    # )
    parser.add_argument(
        "-m", "--mode", action="store", type=int,
        help="index of the mode to consider (-1 means all)", default=0
    )
    parser.add_argument(
        "-d", "--displacement", action="store", type=float,
        help="maximum dispacement to consider", default=0.5
    )
    parser.add_argument(
        "-c", "--couples", action="store", type=list, nargs="*",
        help="couples of atoms whose distance has to be computed", default=None
    )
    # parser.add_argument(
    #     "-u", "--unit", action="store", type=str,
    #     help="measure unit of the displacement (A,bohr)", default="A"
    # ) 
    parser.add_argument(
        "-n", "--number", action="store", type=int,
        help="number of configurations to be computed", default=20
    ) 
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="radix of the output file (xyz) with all the configurations", default=None
    ) 
    # parser.add_argument(
    #     "-od", "--outdir", action="store", type=str,
    #     help="output directory where a file (xyz) for each configuration will be saved", default=None
    # )    
       
    options = parser.parse_args()

    return options

def get_all_couples(N):
    out = list()
    for i in range(N):
        for j in range(i,N):
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

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def get_c(i,j):
    return "[%d,%d]"%(i,j)

def initialization(options):

    # read the input file using ase
    print("\n\treading cell from file '%s'"%(options.input))
    data = read(options.input)

    #print("\n\trelaxed positions")
    #print(data.positions)

    print("\n\treading cell from file '%s'"%(options.input))
    eigenvec = np.loadtxt(options.eigenvec,skiprows=1)
    #print("\n\teigenvectors")
    #print(eigen)

    if options.number <= 0 :
        raise ValueError("'numbers' (-n, --numbers) has to be greater than 0")
    if options.displacement <= 0 :
        raise ValueError("'displacement' (-d, --displacement) has to be greater than 0.0")
    if options.mode < 0 and options.mode != -1 :
        raise ValueError("'mode' (-m, --mode) has to be greater than or equal to 0 (-1 means all modes)")
    if options.mode > len(eigenvec) :
        raise ValueError("'mode' (-m, --mode) has to be smaller than the number of vibratinal modes")

    # folder = tempfile.TemporaryDirectory().name
    # print("\n\tcreating a temporary folder with name '%s'\n"%(folder))
    # #os.mkdir(os.getcwd()+"/tmp")
    # os.mkdir(folder)    

    positions = data.positions.copy()
    if options.mode == -1 :
        modes = np.arange(len(eigenvec))
    else :
        modes = [options.mode]

    
    print("\tpreparing/checking couples of atoms")
    N = len(positions)
    if options.couples is None:
        options.couples = get_all_couples(N)
    else :
        options.couples = check_couples(options.couples,N)

    return data,eigenvec,modes,positions

def main():
    """main routine"""

    options = prepare_parser()

    data,eigenvec,modes,positions = initialization(options)

    columns = [ get_c(i,j) for i,j in zip(options.couples[:,0],options.couples[:,1])]
    distances = pd.DataFrame(columns=columns,index=np.arange(options.number))
    #distances["A"] = options.couples[:,0]
    #distances["B"] = options.couples[:,1]

    for mode in modes:
        print("\tcomputing configurations for mode %d"%(mode))
        vec = eigenvec[mode,:]
        for n in range(options.number):
            phase = float(n) / float(options.number)
            #print("\t\tn={:>3d} phase={:>10.6f}".format(n,phase))
            displacement = options.displacement * np.sin( phase*2*np.pi ) * vec
            data.positions = positions + displacement.reshape((-1,3))
        
            for i,j,nAB in zip(options.couples[:,0],options.couples[:,1],np.arange(len(options.couples))):
                c = get_c(i,j)
                A = data.positions[i,:] # position of the atom A
                B = data.positions[j,:] # position of the atom B
                distances.at[n,c] = norm(A-B)

        file = "{:s}.m={:d}.csv".format(options.output,mode)
        print("\t\twriting distances for mode %i to file %s"%(n,file))
        distances.to_csv(file,index=False)

    print("\n\tJob done :)\n")

def plot():

    import matplotlib.pyplot as plt

    options = prepare_parser()
    data,eigenvec,modes,positions = initialization(options)

    for mode in modes:
        file = "{:s}.m={:d}.csv".format(options.output,mode)
        png = "{:s}.m={:d}.png".format(options.output,mode)

        print("\treading distances for mode %d from file '%s'"%(mode,file))
        
        distances = pd.read_csv(file)
        couples = distances.columns

        fig = plt.figure()
        for c in couples :
            plt.plot(distances[c]-np.mean(distances[c]),label=c)

        plt.legend()
        plt.grid()
        plt.savefig(png)
       
    print("\n\tJob done :)\n")


if __name__ == "__main__":
    main()
    plot()