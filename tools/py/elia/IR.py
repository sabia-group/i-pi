# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

import argparse
import numpy as np
import matplotlib.pyplot as plt

def prepare_parser():
    """set up the script input parameters"""

    parser = argparse.ArgumentParser(description="Compute the Infra Red (IR) Raman intensities of the vibrational modes.")

    parser.add_argument(
        "-z", "--born_charges", action="store", type=str,
        help="input file with the Born Effective Charges Z*", default=None
    )
    parser.add_argument(
        "-m", "--modes", action="store", type=str,
        help="input file with the vibrational modes computed by i-PI", default=None
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="output file with the IR intensities", default="IR.txt"
    ) 
    parser.add_argument(
        "-p", "--plot", action="store", type=str,
        help="output file of the plot of the IR intensities", default=None
    ) 
    parser.add_argument(
        "-n", "--number", action="store", type=int,
        help="index of the Born Effective Charges to be plotted", default=0
    ) 
    parser.add_argument(
        "-w", "--eigenvalues", action="store", type=str,
        help="input file with the eigenvalues computed by i-PI", default=None
    ) 
       
    options = parser.parse_args()

    return options

def read_input(options):
    """read the input arrays from file"""

    import os

    class Data: pass
    data = Data()

    file = options.born_charges
    if not os.path.exists(file):
        raise ValueError("'{:s}' does not exists".format(file))
    data.Z = np.loadtxt(file)

    # check all the BEC have the same size
    lenght = [ len(i) for i in data.Z ]
    result = lenght.count(lenght[0]) == len(lenght)
    if not result :
        raise ValueError("Born Effective Charges should have the same size for all the steps")
    
    # check that the BEC lenght is a multiple of 9
    N = len(data.Z[0])
    if N % 9 != 0 :
        raise ValueError("Born Effective Charges with wrong size")
    
    Na = int( N / 9 ) # number of atoms
    Nmd = len(data.Z) # number of MD steps
    temp = np.full((Nmd,3,Na*3),np.nan)
    # MD steps
    for i in range(Nmd): 
        # polarization components
        for j in range(3): 
            temp[i,j,:] = data.Z[i,j::3]
    data.Z = temp

    file = options.modes
    if not os.path.exists(file):
        raise ValueError("'{:s}' does not exists".format(file))
    data.modes = np.loadtxt(file)

    if len(data.modes) != data.Z.shape[2]:
        raise ValueError("Vibrational modes and Born Effective Charges shapes do not match")
    
    file = options.eigenvalues
    if file is not None and ( options.plot is not None or options.plot != "") :
        data.w = np.sqrt(np.loadtxt(file))
        if len(data.w) != len(data.modes):
            raise ValueError("Vibrational modes and eigenvalues shapes do not match")

    return data

def compute(data):
    """compute the IR activities"""

    class Results: pass
    results = Results()

    # number of Molecular Dynamics steps
    Nmd = len(data.Z)
    # number of vibrational modes
    Nmodes = len(data.modes)

    # IR Raman intensities (for each mode and MD step)
    results.IR = np.full((Nmd,Nmodes),np.nan)

    # derivative of the polarization w.r.t. normal/vibrational modes
    results.dP_dQ = np.full((Nmd,3,Nmodes),np.nan)

    # derivative of the cartesian coordinates w.r.t. normal modes
    dRdQ = data.modes #np.linalg.inv(data.modes)
    results.dP_dQ = data.Z @ dRdQ 

    # IR Raman activities
    # row: MD step
    # col: mode 
    results.IR = np.square(results.dP_dQ.sum(axis=1)) # sum over cartesian components

    return results


def main():
    """main routine"""

    print("\n\tScript to compute the IR Raman activities\n\tfrom the vibrational modes and the Born Effective Charge tensors\n")

    # prepare/read input arguments
    print("\tReading script input arguments")
    options = prepare_parser()

    # read input argumfilesents
    print("\tReading input files: '{:s}' and '{:s}'".format(options.born_charges,options.modes))
    data = read_input(options)

    # compute IR activity
    print("\tComputing IR intesities")
    results = compute(data)

    # print IR intesity to file
    print("\tSaving IR intesities to file '{:s}'".format(options.output))
    np.savetxt(options.output,results.IR)
    
    # produce plot of the IR intesities
    if options.plot is not None :
        print("\tPlotting IR {:d}-th intesity to file '{:s}'".format(options.number,options.plot))
        
        fig, ax = plt.subplots(figsize=(10,6))

        x = data.w / 1.5198298e-4
        y = results.IR[options.number,:]
        ax.bar(x,y,color="blue",width=2e-1) #label=str(n)
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        ax.set_ylabel('IR Raman intensity (a.u.)')
        ax.set_xlabel('frequency (THz)')

        plt.tight_layout()
        file = options.plot
        print("\tsaving plot to file '{:s}'".format(file))
        plt.savefig(file)
    

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()