"""
   Purpose: Calculation of Born efefctive charges.
   Usage : Type 'python3 BEC.py --help' for all available options
   Author : Alaa Akkoush (June 2021)
"""

# -------------------Libraries------------------------------#
from argparse import ArgumentParser,ArgumentTypeError
import numpy as np
import os, sys
import time
from numpy import float64, zeros
from ase.io import read,write
from copy import copy,deepcopy
import pandas as pd

import subprocess

# constants
C = 1.6021766e-19  # in coulomb

def shift(xyz,delta):
    if xyz=="x":
        return np.asarray([delta,0,0])
    elif xyz == "y":
        return np.asarray([0,delta,0])
    elif xyz == "z":
        return np.asarray([0,0,delta])
    else :
        raise ValueError("Wrong direction")

def is_complete(file,show):
    if os.path.exists(file):
        with open(file) as f:
            lines = f.readlines()
            if np.any([ "Have a nice day." in lines[-i] for i in range(5) ]):
                if show:
                    print("\t\tFHI-aims calculation is complete")
                return True
            else:
                if show:
                    print("\t\tFHI-aims calculation is not complete")
                #sys.exit(1)
                return False
    else :
        if show:
            print("\t\tfile '%s' does not exist"%(file))
            return False

def postpro(file,show=True):
    """Function to read outputs"""
    #folder = get_folder(atom,xyz,dn)
    p = None
    volume = None
    if is_complete(file,show):
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if line.rfind("| Cartesian Polarization ") != -1:
                    p = float64(split_line(line)[-3:])  #
                if line.rfind("| Unit cell volume ") != -1:
                    volume = float(split_line(line)[-2])
            return p, volume
    else :
        return None,None

def split_line(lines):
    """Split input line"""
    line_array = np.array(lines.strip().split(" "))
    line_vals = line_array[line_array != ""]
    return line_vals

def get_folder(atom,xyz,dn):
    return "BEC-I=%d-c=%s-d=%s"%(atom,xyz,dn)

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def main():
    """main routine"""

    parser = ArgumentParser(description="BEC calculation with FHI-aims")
    parser.add_argument(
        "-x", "--executable", action="store",
        help="path to FHI-aims binary", default="/home/elia/Google-Drive/google-personal/FHIaims/build/aims.221103.mpi.x"
    )
    parser.add_argument(
        "-s",
        "--suffix",
        action="store",
        help="Call suffix for binary e.g. 'mpirun -n 8 '",
        default="mpirun -n 8",
    )
    parser.add_argument(
        "-d",
        "--delta",
        action="store",
        type=float,
        nargs="+",
        dest="delta",
        help="finite difference poles, defualt is [0.01]",
        default=[0.1],
    )
    parser.add_argument(
        "-r", "--run", action="store", type=str2bool,
        help="perform DFT calculations", default=True
    )
    parser.add_argument(
        "-pp", "--postprocessing", action="store", type=str2bool,
        help="perform post-processing", default=False
    )
    parser.add_argument(
        "-g", "--geofile", action="store", type=str,
        help="original geometry file", default='geometry.in'
    )
    parser.add_argument(
        "-c", "--controlfile", action="store", type=str,
        help="control file", default="control.in"
    )
    parser.add_argument(
        "-ps", "--pertspecies", action="store", type=str2bool,
        help="perturb all atoms of the same species", default=False
    )
    parser.add_argument(
        "-rs", "--restartsuffix", action="store", type=str,
        help="restart suffix", default="FHI-aims.restart"
    )
    parser.add_argument(
        "-p", "--polout", action="store", type=str,
        help="csv output file for polarization", default="pol.csv"
    )
    parser.add_argument(
        "-bec", "--becout", action="store", type=str,
        help="csv output file for BEC tensors", default="BEC.csv"
    )
    parser.add_argument(
        "-o", "--original", action="store", type=str,
        help="original output file with no displacement", default="FHI-aims.out"
    )
    parser.add_argument(
        "-w", "--overwrite", action="store", type=bool,
        help="overwrite previous calculations", default=False
    )

    options = parser.parse_args()

    options.aims_run = options.suffix + " " + options.executable
    options.aimsout = "FHI-aims.out"

    # cmd = "{:s} > {:s} ".format(options.aims_run,options.aimsout)
    script = 'run.BEC.sh'
    # with open(script,'w') as s:
    #     s.write('source ~/.bashrc\n')
    #     s.write('source ~/.elia\n')
    #     s.write(cmd)
    # os.system('chmod +x {:s}'.format(script))

    if options.run : # perform DFT calculations
        print("\n\tComputing BEC tensors")
        data    = read(options.geofile)

        for atom in range(data.get_global_number_of_atoms()):
            for dn,delta in enumerate(options.delta):
                for xyz in [ "x","y","z" ]:
                    print("\n\t I={:<2d} | c={:<3d} | d='{:<1s}'".format(atom,dn,xyz))
                    if options.pertspecies :
                        raise ValueError("Not yet implemented")
                        # S2Imap  = data.symbols.indices() # Species to Index map
                        # Species = data.get_chemical_symbols()
                    else :
                        newdata = deepcopy(data)
                        newdata.positions[atom,:] += shift(xyz,delta)


                    folder = get_folder(atom,xyz,dn)
                    if not os.path.exists(folder):
                        print("\t\tcreating folder '%s'"%(folder))
                        os.mkdir(folder)

                    os.system("cp {:s} {:s}/.".format(script,folder))

                    # if not options.overwrite :
                    #     file = folder + "/" + options.aimsout
                    #     print("\t\treading output file '%s'"%(file))
                    #     if is_complete(file,show=False):
                    #         print("\t\tcomputation completed")
                    #         print("\t\t'overwrite' flag set to 'false': skipping computation")
                    #         continue
                    # else:
                    #     if os.path.exists(file):
                    #         print("\t\tcomputation not completed")
                    #         print("\t\t'overwrite' flag set to 'true': computing again")
                    #         continue


                    newcpfile = folder + "/control.in"
                    print("\t\tcopying '%s' to '%s'"%(options.controlfile,newcpfile))
                    os.popen('cp %s %s'%(options.controlfile,newcpfile))

                    newgeofile = folder + "/geometry.in"
                    print("\t\twriting new geometry file to '%s'"%(newgeofile))
                    write(newgeofile,newdata)

                    print("\t\tcopying restart files to folder '%s'"%(folder))
                    os.popen('cp %s* %s/.'%(options.restartsuffix,folder))

                    # print("\t\trunning calculation, output printed to '%s'"%(options.aimsout))
                    # os.chdir(folder)

                    # #subprocess.Popen("./{:s}".format(script)).wait()
                    # #subprocess.run(["./{:s}".format(script)], shell=True)

                    # os.chdir("..")

                    # if is_complete(folder + "/" + options.aimsout,show=False):
                    #     print("\t\tcomputation completed")
                    # else :
                    #     print("\t\tcomputation not completed")

                    # print("\t\tremoving restart files from folder '%s'"%(folder))
                    # os.popen('rm %s/%s*'%(folder,options.restartsuffix))

    if options.postprocessing : # Post-Processing
        print("\n\tPost-Processing")

        data = read(options.geofile)
        N    = data.get_global_number_of_atoms()
        P = pd.DataFrame(columns=["atom","delta","xyz","px","py","pz"])
        for atom in range(N):
            for dn,delta in enumerate(options.delta):
                for xyz in [ "x","y","z" ]:
                    folder = get_folder(atom,xyz,dn)

                    print("\n\t I={:<2d} | c={:<3d} | d='{:<1s}'".format(atom,dn,xyz))
                    if not os.path.exists(folder):
                        print("\t\tfolder '%s' does not exist"%(folder))
                        continue

                    file = folder + "/" + options.aimsout
                    print("\t\treading output file '%s'"%(file))
                    p, V = postpro(file,show=False)

                    if p is None or V is None:
                        print("\t\tcomputation not completed")
                        continue
                    else :
                        print("\t\tread polarization and volume")
                        row = {"atom":atom,"delta":delta,"xyz":xyz,\
                            "px":p[0],"py":p[1],"pz":p[2]}#,"V":V}
                        P = P.append(row,ignore_index=True)

        print("\n\tSaving polarizations to file '%s'"%(options.polout))
        P.to_csv(options.polout,index=False)

        print("\tComputing BEC tensors")
        P0,V = postpro(options.original,show=False)
        born_factor = (V * 1e-20) / C

        columns =  ["atom","name","delta",\
                    "Zxx","Zxy","Zxz",\
                    "Zyx","Zyy","Zyz",\
                    "Zzx","Zzy","Zzz"]
        BEC = pd.DataFrame(columns=columns)
        for atom in range(N):

            for delta in options.delta:
                row = dict(zip(columns, [None]*len(columns)))
                row["atom"] = atom
                row["name"] = data.get_chemical_symbols()[atom]
                row["delta"] = delta

                for dir_xyz in [ "x","y","z" ]:
                    P1 = P.where( P["atom"] == atom).where(P["delta"] == delta ).where(P["xyz"] == dir_xyz).dropna()
                    if len(P1) != 1 :
                        raise ValueError("Found more than one row for atom=%d, delta=%f, xyz=%s"%(atom,delta,dir_xyz))
                    i = P1.index[0]

                    for n,pol_xyz in enumerate([ "x","y","z" ]):
                        BECcol = "Z%s%s"%(pol_xyz,dir_xyz)
                        Pcol = "p%s"%pol_xyz

                        p1 = P1.at[i,Pcol]
                        p0 = P0[n]
                        if p1 is None or p0 is None:
                            raise ValueError("Polarization is None")

                        # compute BEC
                        # the first index indicate the polarization component
                        # the secondi indicate the displacement
                        row[BECcol] = born_factor * ( p1 - p0) / delta  # fix this if options.pertspecie == True

                if np.any( [ j is None for j in row.values() ]):
                    raise ValueError("Found None value")
                BEC = BEC.append(row,ignore_index=True)

        print("\tSaving BEC tensors to file '%s'"%(options.becout))
        BEC.to_csv(options.becout,index=False)

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()
