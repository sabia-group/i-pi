
"""Functions used to read input configurations and print trajectories
in the XYZ format.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

from argparse import ArgumentParser

import os

import numpy as np

#import ipi.utils.mathtools as mt
#from ipi.utils.depend import dstrip
#from ipi.utils.units import Elements
#from ipi.utils.units import Elements
#from ipi.utils.io.backends import io_xyz
from ase.io import read#,write
#from ase.cell import Cell
#from copy import copy
#from ipi.utils.messages import verbosity, warning, info
#from numpy.linalg import inv
#import tempfile
#import pandas as pd
#import cmath
import matplotlib.pyplot as plt

def norm(v):
    return np.sqrt(v@v)

def print_cell(cell,tab="\t\t"):
    string = tab+"{:14s} {:1s} {:^10s} {:^10s} {:^10s}".format('','','x','y','z')
    for i in range(3):
        string += "\n"+tab+"{:14s} {:1d} : {:>10.6f} {:>10.6f} {:>10.6f}".format('lattice vector',i+1,cell[i,0],cell[i,1],cell[i,2])
    return string

def prepare_parser():

    parser = ArgumentParser(description="Prepare the xyz files for a vibrational mode animation.")
    parser.add_argument(
        "-r", "--relaxed", action="store", type=str,
        help="input file with the relaxed position, around which the virbational modes have been computed", default=None
    )
    parser.add_argument(
        "-p", "--positions", action="store", type=str,
        help="input file with the position of all the configurations (in 'xyz' format)", default=None
    )
    parser.add_argument(
        "-v", "--eigenvec", action="store", type=str,
        help="file containing the eigenvectors computed by i-PI", default=None
    )
    parser.add_argument(
        "-e", "--eigenval", action="store", type=str,
        help="file containing the eigenvalues computed by i-PI", default=None
    )
    parser.add_argument(
        "-t", "--timestep", action="store", type=float,
        help="time step of the MD simulation (a.u.)", default=None
    )
    parser.add_argument(
        "-c", "--config", action="store", type=str,
        help="configuration output file (csv)", default='configurations.csv'
    ) 
    parser.add_argument(
        "-d", "--displacements", action="store", type=str,
        help="displacements output file (csv)", default='displacements.csv'
    ) 
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="output file", default="projections.csv"
    ) 
       
    options = parser.parse_args()

    return options

def main():
    """main routine"""

    options = prepare_parser()

    # read the input file using ase
    print("\n\treading relaxed positions from file '%s'"%(options.relaxed))
    data = read(options.relaxed)

    #print("\n\trelaxed positions")
    #print(data.positions)

    print("\n\treading eigenvectors from file '%s'"%(options.eigenvec))
    eigenvec = np.loadtxt(options.eigenvec,skiprows=1)
    Nmodes = len(eigenvec)
    #print("\n\teigenvectors")
    #print(eigen)

    print("\n\treading eigenvalues from file '%s'"%(options.eigenval))
    eigenval = np.loadtxt(options.eigenval,skiprows=1)
 
    print("\n\treading configuations from file '%s'"%(options.positions))
    Na = data.get_global_number_of_atoms()
    Nc = 0
    configurations = list()

    tempfile = 'temporary.xyz'
    with open(options.positions,'r') as file :
        while True :        
            try :
                lines = [next(file) for _ in range(Na+2)] 
                try :           
                    #print("\n\treading configuation %d"%(Nc+1),end="\r")
                    temp = open(tempfile,'w')         
                    for l in lines:
                        temp.write(l)                   
                    temp.close() 
                    configurations.append(read(tempfile,format="xyz").positions)
                    Nc += 1
                except :
                    raise ValueError("some error occurred")                
            except:
                break
    os.remove(tempfile)

    print("\n\twriting configuations to file '%s'"%(options.config))
    configs = np.zeros(shape=(len(configurations),3*len(configurations[0])))
    for i in range(len(configs)):
        configs[i,:] = configurations[i].flatten()
    np.savetxt(fname=options.config,X=configs,delimiter=',')

    print("\n\twriting displacements to file '%s'"%(options.displacements))
    relaxed = np.asarray(data.positions).flatten()
    displs = np.zeros(shape=configs.shape)
    for i in range(len(displs)):
        displs[i,:] -= relaxed
    np.savetxt(fname=options.displacements,X=displs,delimiter=',')

    print("\n\tcomputing initial phase mismatch for each mode")
    A   = displs [0,:]#.reshape((-1,3)) # it should be Na x 3
    relaxed = relaxed.reshape((-1,3)) # it should be Na x 3

    # let's compute the phases of each mode
    phases = np.zeros(shape=(Nmodes))
    for mode in range(Nmodes):
        B = eigenvec[:,mode]
        phases[mode] = np.arccos( ( A @ B ) / ( norm(A) * norm (B) ) ) # phase in rad

    #print("\n\tcomputing initial phase mismatch for each ion")
    #displs = displs.reshape((Nc,3*Na))
    #eigenvec = eigenvec.reshape((Na,3)) # normalized per columns
    projections = np.zeros(shape=(Nc,Nmodes))
    for n in range(Nc): # cycle over al configurations
        for mode in range(Nmodes):
            # u(t) = v exp(-iwt+p)
            Tevolution = np.exp( -1.j * ( n * eigenval[mode] * options.timestep - phases[mode] ) )
            u = eigenvec[:,mode] * np.real( Tevolution ) # the real part of u(t)
            d = displs[n,:] # the displacement
            projections[n,mode] = d @ u / ( norm(d) * norm (u) )

    print("\n\twriting projections to file '%s'"%(options.output))
    np.savetxt(fname=options.output,X=projections,delimiter=',')

    plt.figure()
    for mode in range(Nmodes):
        plt.plot(projections[:,mode])
    plt.grid()
    plt.legend()
    plt.show()

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()