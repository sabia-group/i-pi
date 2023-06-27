# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

import argparse
import numpy as np
import pandas as pd
from functions import get_property_header

description = "Given a nuclear configuration, this script generate its parity-inverted configuration (some atoms could be swapped).\
               Then a sequence of intermediate configuration is generated.\
               This is useful for computing the energy/polarization at varying configuration when the latter spans from one spontaneous polarization ground state\
               to the other. In thiw way it is possible to see the energy barrier to induce a polarization reversal, and see how some properties varyies along this path."

def prepare_parser():

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-i", "--input", action="store", type=str,
        help="input file", default="i-pi.properties.out"
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="output file (csv format)", default="i-pi.properties.csv"
    )
    parser.add_argument(
        "-f", "--format", action="store", type=str,
        help="output format", default='%20.10e'
    )

    return parser.parse_args()

def main():
    """main routine"""

    ###
    # prepare/read input arguments
    print("\n\tReding input arguments")
    options = prepare_parser()

    file = options.input
    print("\tReding properties from file '{:s}'".format(file))
    data = np.loadtxt(file)

    print("\tReading properties names from file '{:s}'".format(file))
    names = get_property_header(file)

    print("\tCreating dataframe")
    df = pd.DataFrame(data=data,columns=names)

    file = options.output
    print("\tSaving dataframe to file '{:s}'".format(file))
    df.to_csv(file, float_format=options.format,index=False)

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()