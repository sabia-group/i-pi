# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

import argparse
import numpy as np
from ase.io import read, write
from ase import Atoms
import json

def prepare_parser():
    description= "Create a sequence of nuclear configurations (xyz format) where the nuclei are displaced along the vibrational mode.\
                The output file can be used by i-PI using the 'Sequence' class to compute (for example) the numerical differences of the polarization along the vibrational mode."
        
    parser = argparse.ArgumentParser(description=description)

    
    # parser.add_argument(
    #     "-q", "--positions", action="store", type=str,
    #     help="input file with the positions of all the configurations (in 'xyz' format)", default=None
    # )
    parser.add_argument(
        "-q", "--positions", action="store", type=str,
        help="input file (xyz format) with the positions w.r.t. the displacement are performed", default=None
    )
    parser.add_argument(
        "-u", "--units", action="store", type=str,
        help="units of the input file", default="angstrom"
    )
    parser.add_argument(
        "-d", "--displacement", action="store", type=float,
        help="displacement along the vibrational modes (a.u.)", default=1
    )
    parser.add_argument(
        "-m", "--modes", action="store", type=str,
        help="folder containing the (normalized) vibrational modes computed by i-PI", default=None
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="output file with displaced positions (xyz format, and in a.u.)", default="sequence.xyz"
    )      
    parser.add_argument(
        "-i", "--information", action="store", type=str,
        help="output file with informations about the displaced configurations", default="info.json"
    )     
    options = parser.parse_args()

    return options

def get_one_file_in_folder(folder,ext):
    import os
    files = list()
    for file in os.listdir(folder):
        if file.endswith(ext):
            files.append(os.path.join(folder, file))
    if len(files) == 0 :
        raise ValueError("no '*{:s}' files found".format(ext))
    elif len(files) > 1 :
        raise ValueError("more than one '*{:s}' file found".format(ext))
    return files[0]

class Data:

    tab = "\t\t"
    check = True
    thr = 0.1
    check_orth = 0.001
    check_norm = 0.001

    def __init__(self,options):

        ###
        # reading original position
        print("\n{:s}reading original/relaxed position from file '{:s}'".format(Data.tab,options.positions))
        atom = read(options.positions)#,format=options.positions_format)
        original = np.asarray(atom.positions)

        if options.units in ["angstrom","ang","A"]:
            original *= 1.88972612463 # from angstrom to a.u.

        ###
        # reading vibrational modes           
        file = options.modes
        print("\n{:s}reading vibrational modes from file '{:s}'".format(Data.tab,file))
        modes = np.loadtxt(file)
        Nmodes = len(modes)
        if modes.shape[0] != modes.shape[1] :
            raise ValueError("vibrational modes matrix is not square")

        # check that the eigenvectors are orthogonal (they could not be so)
        if Data.check_norm > 0.0:                
            print("{:s}checking that the normal modes are normalized, i.e. |N_s| = 1 ".format(Data.tab))
            res = np.linalg.norm(np.linalg.norm(modes,axis=0) - 1.0 )
            print("{:s} | |N_s| - 1 | = {:>20.12e}".format(Data.tab,res))
            if res > Data.check_norm :
                raise ValueError("the normal modes are not normalized")
            
        # # eigenvectors
        # file = options.eigvec
        # print("\n{:s}reading eigenvectors from file '{:s}'".format(Data.tab,file))
        # eigvec = np.loadtxt(file)
        # if eigvec.shape[0] != Nmodes or eigvec.shape[1] != Nmodes:
        #     raise ValueError("eigenvectors matrix with wrong size")
            
        # # check that the eigenvectors are orthogonal (they could not be so)
        # if Data.check_orth > 0.0:                
        #     print("{:s}checking that the eigenvectors are orthonormal, i.e. M @ M^t = Id".format(Data.tab))
        #     res = np.linalg.norm(eigvec @ eigvec.T - np.eye(Nmodes))
        #     print("{:s} | M @ M^t - Id | = {:>20.12e}".format(Data.tab,res))
        #     if res > Data.check_orth :
        #         raise ValueError("the eigenvectors are not orthonormal")

        if len(original) == Nmodes :
            raise ValueError("original position does not have the correct shape")
        
        self.modes    = modes
        # self.eigvec   = eigvec
        self.Nmodes   = Nmodes
        self.original = original
        self.atom     = atom

        pass

    def compute(self,options):

        print("{:s}computing displaced configurations".format(self.tab))
        N = self.Nmodes*2
        #output = [Atoms()]*N
        Na = len(self.original)
        atoms = self.atom.get_chemical_symbols()
        
        info = {"dir":[0]*N,"mode":[0]*N,"disp":options.displacement} 

        print("{:s}writing displaced configurations to file '{:s}'".format(Data.tab,options.output))
        with open(options.output,'w') as f:

            k = 0 
            for i in range(self.Nmodes):
                for j in [-1,1]:
                    f.write(str(Na)+"\n")
                    f.write("# mode {:d}, displacement {:s}{:d}\n".format(i,"" if j == -1 else "+",j))

                    pos = ( self.original.flatten() + j*options.displacement*self.modes[:,i]).reshape((-1,3))
                    for ii in range(Na):
                        f.write("{:>2s} {:>20.12e} {:>20.12e} {:>20.12e}\n".format(atoms[ii],*pos[ii,:]))
                    # write(f,pos.reshape((-1,3)))
                    info["dir"][k] = j
                    info["mode"][k] = i
                    k += 1

        # write(options.output,output,format="xyz")

        print("{:s}writing information to file '{:s}'".format(Data.tab,options.information))
        with open(options.information, 'w') as fp:
            json.dump(info, fp,indent=4)

        pass

def main():
    """main routine"""

    ###
    # prepare/read input arguments
    print("\n\tReding input arguments")
    options = prepare_parser()

    ###
    # read input
    print("\n\tReding input files")
    data = Data(options)
    # print(data.__dict__.keys()) # = dict_keys(['modes', 'Nmodes', 'original'])
    
    ###
    # compute displacements and write them to file
    print("\n\tComputing occupations")
    data.compute(options)
   
    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()