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


description =  \
" \n\
\tThis script project a MD trajectoies onto the vibrational modes.\n\
\n\tInput:\n\
\t- MD trajectories (positions and velocities)\n\
\t- some output files produced by the vibrational analysis\n\
\t- the equilibrium nuclear configuration w.r.t. the vibrational analysis has been performed\n\
\t- the time (in a.u.) at which the nuclear configuration are evaluated\n\
\tThe latter is necessary in order to compute the displacements w.r.t. this configuration.\n\
\n\tSome comments:\n\
\tThe projection on the vibrational modes is just a change of variables.\n\
\tActually it means to change from cartesian coordinates to angle-action variables (more or less).\n\
\n\tOutput:\n\
\t - the amplitudes of the modes in two different formats:\n\
\t - A-amplitudes.txt (dimension of lenght x mass^{1/2})\n\
\t - B-amplitudes (dimensionless)\n\
\t\t(The A-amplitudes have to be considered in case the vibrational modes are written in terms of the dynamical matrix eigenvector and the nuclear masses.\n\
\t\tThe B-amplitudes have to be considered in case the vibrational modes are written in terms of the (dimensionless and orthonormalized) normal modes.\n\
\t\tAsk to the author for more details.)\n\
\t - the \"initial phases\" of the modes (the angle variable are actually wt+\phi and not just \phi).\n\
\t - the energy of each vibrational modes\n\
\t - the (classical) occupation number of each vibrational modes (i.e. the zero-point-energy is neglected)\n\
\n\tThe script can produce as additional output:\n\
\t - some plots of the vibrationalmodes energies\n\
\t\t(The energies of the vibrational modes can be used to check the equipartition theorem is valid, i.e. whether the system has thermalized.)\n\
\n\tauthor: Elia Stocco\
\n\temail : stocco@fhi-berlin.mpg.de"

# - the characteristic mass of the vibrationa modes\n\
# - the characteristic lenght of the vibrationa modes\n\

def prepare_parser():

    parser = argparse.ArgumentParser(description=description,\
                                     formatter_class=RawTextHelpFormatter,\
                                        usage=SUPPRESS)

    parser.add_argument(
        "-q", "--positions", action="store", type=str,metavar='\b',
        help="input file with the positions of all the configurations (in 'xyz' format)"#, default=None
    )
    parser.add_argument(
        "-r", "--relaxed", action="store", type=str,metavar='\b',
        help="input file with the relaxed/original configuration (in 'xyz' format)"#, default=None
    )
    parser.add_argument(
        "-M", "--masses", action="store", type=str,metavar='\b',
        help="input file with the nuclear masses (in 'txt' format)"#, default=None
    )
    parser.add_argument(
        "-v", "--velocities", action="store", type=str,metavar='\b',
        help="input file with the velocities of all the configurations (in 'xyz' format)"#, default=None
    )
    parser.add_argument(
        "-m", "--modes", action="store", type=str,metavar='\b',
        help="folder containing the vibrational modes computed by i-PI"#, default=None
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,metavar='\b',
        help="output folder", default="output"
    ) 
    parser.add_argument(
        "-pr", "--property", action="store", type=str,metavar='\b',
        help="file containing the property at each MD step computed by i-PI"#, default=None
    )

    parser.add_argument(
        "-c", "--compute", action="store", type=str2bool,metavar='\b',
        help="whether the modes occupations are computed", default=True
    )
    parser.add_argument(
        "-p", "--plot", action="store", type=str,metavar='\b',
        help="output file for the modes occupation plot", default=None
    )
    parser.add_argument(
        "-t", "--t-min", action="store", type=int,metavar='\b',
        help="minimum time to be plotted", default=0
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