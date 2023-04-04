# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import argparse
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def prepare_parser():

    parser = argparse.ArgumentParser(description="Compute the time-dependent vibrational modes occupations given the velocities of a MD simulation.")

    parser.add_argument(
        "-v", "--velocities", action="store", type=str,
        help="input file with the velocities of all the configurations (in 'xyz' format)", default=None
    )
    parser.add_argument(
        "-m", "--modes", action="store", type=str,
        help="file containing the vibrational modes computed by i-PI", default=None
    )
    parser.add_argument(
        "-o", "--occupations", action="store", type=str,
        help="output file with the modes occupation", default="occupations.txt"
    ) 

    parser.add_argument(
        "-c", "--compute", action="store", type=str2bool,
        help="whether the modes occupations are computed", default=True
    )

    parser.add_argument(
        "-p", "--plot", action="store", type=str,
        help="output file for the modes occupation plot", default=None
    )

    parser.add_argument(
        "-s", "--signal", action="store", type=str,
        help="output file for the velocities projected on the vibrational modes", default=None
    )
       
    options = parser.parse_args()

    return options

def main():
    """main routine"""

    ###
    # prepare/read input arguments
    options = prepare_parser()

    occupations = None
    if options.compute :

        from ase.io import read
        from scipy.signal import hilbert

        ###
        # reading velocities
        print("\n\treading velocities from file '{:s}'".format(options.velocities))
        velocities = read(options.velocities,index=":")
        Nconf = len(velocities)
        print("\tread {:d} configurations".format(Nconf))

        ###
        # reading vibrational modes
        print("\n\treading vibrational modes from file '{:s}'".format(options.modes))
        modes = np.loadtxt(options.modes)
        Nmodes = len(modes)
        print("\tread {:d} modes".format(Nmodes))

        if modes.shape[0] != modes.shape[1]:
            raise ValueError("matrix of vibrational modes is not square")
    
        if not np.all(np.asarray([ velocities[i].positions.flatten().shape for i in range(Nconf)]) == Nmodes) :
            raise ValueError("some configurations do not have the correct shape")

        # ###
        # # reading eigenvalues
        # print("\n\treading eigenvalues from file '{:s}'".format(options.eigenvalues))
        # eigvals = np.loadtxt(options.eigenvalues)
        # Nvals = len(eigvals)
        # print("\tread {:d} eigenvalues".format(Nvals))
        # if Nvals != Nmodes:
        #     raise ValueError("the number fo eigenvalues is not equal to the number of vibrational modes")
        
        ###
        # flatten the velocities
        for n in range(Nconf):
            velocities[n] = velocities[n].positions.flatten()
        velocities = np.asarray(velocities)
        
        ###
        # project on phonon modes
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#scipy.signal.hilbert
        print("\n\tprojecting velocities on vibrational modes")
        signal = velocities @ modes

        if options.signal is not None :
            print("\tsaving velocities projected on the vibrational modes to file '{:s}'".format(options.signal))
            np.savetxt(options.signal,signal,delimiter=" ",fmt="%20.12e")

        print("\tcomputing the analytic signal of the velocities along the vibrational modes")
        analytic_signal = hilbert(signal,axis=0)

        print("\tcomputing the time-dependent occupations of the vibrational modes")
        occupations = np.absolute(analytic_signal)
        
        ###
        # save occupations to file  
        print("\tsaving modes occupations to file '{:s}'".format(options.occupations))
        np.savetxt(options.occupations,occupations,delimiter=" ",fmt="%20.12e")
    
    if options.plot is not None:

        import matplotlib.pyplot as plt

        if occupations is None :
            print("\n\treading modes occupations from file '{:s}'".format(options.occupations))
            occupations = np.loadtxt(options.occupations)
        
        print("\n\tplotting modes occupations")

        plt.figure()
        Nmodes = occupations.shape[1]
        for n in range(Nmodes):
            plt.plot(occupations[:,n],label=str(n))
        
        plt.grid()
        plt.legend()

        print("\n\tsaving plot to file '{:s}'".format(options.plot))
        plt.savefig(options.plot)


    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()