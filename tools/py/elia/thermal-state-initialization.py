# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

import argparse
from argparse import RawTextHelpFormatter,SUPPRESS
from functions import str2bool
from classes import Data

# This script project a MD trajectoies onto the vibrational modes.

# Input:
# - MD trajectories (positions and velocities)
# - some output files produced by the vibrational analysis
# - the equilibrium nuclear configuration w.r.t. the vibrational analysis has been performed
# - the time (in a.u.) at which the nuclear configuration are evaluated
# The latter is necessary in order to compute the displacements w.r.t. this configuration.

# Some comments:
# The projection on the vibrational modes is just a change of variables.
# Actually it means to change from cartesian coordinates to angle-action variables (more or less).

# Output:
#  - the amplitudes of the modes in two different formats:
#  - A-amplitudes.txt (dimension of lenght x mass^{1/2})
#  - B-amplitudes (dimensionless)
#         (The A-amplitudes have to be considered in case the vibrational modes are written in terms of the dynamical matrix eigenvector and the nuclear masses.
#         The B-amplitudes have to be considered in case the vibrational modes are written in terms of the (dimensionless and orthonormalized) normal modes.
#         Ask to the author for more details.)
#  - the "initial phases" of the modes (the angle variable are actually wt+\phi and not just \phi).
#  - the energy of each vibrational modes
#  - the (classical) occupation number of each vibrational modes (i.e. the zero-point-energy is neglected)

# The script can produce as additional output:
#  - some plots of the vibrationalmodes energies
#         (The energies of the vibrational modes can be used to check the equipartition theorem is valid, i.e. whether the system has thermalized.)

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de


def prepare_parser():

    parser = argparse.ArgumentParser(description="",\
                                     formatter_class=RawTextHelpFormatter,\
                                        usage=SUPPRESS)

    parser.add_argument(
        "-T", "--temperature", action="store", type=float,metavar='\b',
        help="temperature (Kelvin)"#, default=None
    )
    parser.add_argument(
        "-r", "--relaxed", action="store", type=str,metavar='\b',
        help="input file with the relaxed/original configuration ('xyz' format, a.u.)"#, default=None
    )
    parser.add_argument(
        "-m", "--modes", action="store", type=str,metavar='\b',
        help="folder containing the vibrational modes computed by i-PI"#, default=None
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,metavar='\b',
        help="prefix for the output files", default="start"
    ) 
    parser.add_argument(
        "-x", "--random", action="store", type=str,metavar='\b',
        help="file containing the state of the random number generator", default="random.state.pickle"
    ) 
       
    options = parser.parse_args()

    return options

def main():
    """main routine"""

    ###
    # prepare/read input arguments
    # print("\n\tReding input arguments")
    options = prepare_parser()

    ###
    # compute occupations
    if options.compute :

        # read input files
        print("\n\tReding input files for computation")
        data = Data(options,what="compute")
        
        print("\n\tComputing occupations")
        data.compute()

        data.save(options.output)    

    ###
    # plot occupations
    if options.plot:

        # read input files
        print("\n\tReding input files for plot")
        data = Data(options,what="plot")

        print("\n\tPlotting normalized energy per mode")
        data.plot(options)

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()



import random
import numpy as np
import pickle 


s = random.getstate()

with open('data.pickle', 'wb') as f:
    pickle.dump(s, f)


for i in range(10):
    # random.setstate(data)
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    random.setstate(data)
    del data
    print(random.random()*2*np.pi)


print("\n\tJob done :)\n")