#!/usr/bin/env python3

""" getproperty.py

Parses a property output file and - if present - outputs the column(s)
corresponding to the desired property. Relies on the infrastructure of i-pi,
so the ipi package should be installed in the Python module directory, or
the i-pi main directory must be added to the PYTHONPATH environment variable.

Syntax:
   geproperty.py propertyfile propertyname [skip]
"""


import sys
import re
import numpy as np
from ipi.utils.messages import warning

def getproperty(inputfile, propertyname,data=None,skip="0"):

    import re
    import numpy as np

    if type(propertyname) in [list,np.ndarray]: 
        out = dict()
        data = np.loadtxt(inputfile)
        for p in propertyname:
            out[p] = getproperty(inputfile,p,data,skip=skip)
        return out
    
    print("searching for '{:s}'".format(propertyname))

    skip = int(skip)

    propertyname = " " + propertyname + " "

    # opens & parses the input file
    ifile = open(inputfile, "r")

    # now reads the file one frame at a time, and outputs only the required column(s)
    icol = 0
    while True:
        try:
            line = ifile.readline()
            if len(line) == 0:
                raise EOFError
            while "#" in line :  # fast forward if line is a comment
                line = line.split(":")[0]
                if propertyname in line :
                    cols = [ int(i)-1 for i in re.findall(r"\d+", line) ]                    
                    if len(cols) == 1 :
                        icol += 1
                        output = data[:,cols[0]]
                    elif len(cols) == 2 :
                        icol += 1
                        output = data[:,cols[0]:cols[1]+1]
                    elif len(cols) != 0 :
                        raise ValueError("wrong string")
                    if icol > 1 :
                        raise ValueError("Multiple instances for '{:s}' have been found".format(propertyname))                    
                # get new line
                line = ifile.readline()
                if len(line) == 0:
                    raise EOFError
            if icol <= 0:
                print("Could not find " + propertyname + " in file " + inputfile)
                raise EOFError
            else :
                return np.asarray(output)

        except EOFError:
            break


def main(inputfile, propertyname="potential", skip="0"):
    skip = int(skip)

    # opens & parses the input file
    ifile = open(inputfile, "r")

    reprop = re.compile(" ([0-9]*) *--> " + propertyname)
    #    reunit = re.compile("{(.*)}")

    # now reads the file one frame at a time, and outputs only the required column(s)
    icol = -1
    step = 0
    while True:
        try:
            line = ifile.readline()
            if len(line) == 0:
                raise EOFError
            while line[0] == "#":  # fast forward if line is a comment
                rm = reprop.search(line)
                if not (rm is None):
                    if icol >= 0:
                        warning(
                            "Multiple instances of the specified property "
                            + propertyname
                            + " have been found"
                        )
                        raise EOFError
                    icol = int(rm.group(1)) - 1
                line = ifile.readline()
                if len(line) == 0:
                    raise EOFError
            if icol < 0:
                warning("Could not find " + propertyname + " in file " + inputfile)
                raise EOFError
            line = line.split()
            if step >= skip:
                print(line[icol])
            step += 1
        except EOFError:
            break


if __name__ == "__main__":
    main(*sys.argv[1:])
