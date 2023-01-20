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

# ------------------------PreProcessing-------------------------------#

# def preprocess(geofile, controlfile):
#     """Checking that inputs are found"""
#     if os.path.exists(geofile):
#         print("geometry.in was found")
#         with open(geofile) as geo:
#             for line in geo:
#                 if line.startswith("atom_frac"):
#                     frac2atom(geofile)
#                     break
#                     # call transform
#     else:
#         print("Error: Cannot find geometry.in.\n")
#         sys.exit(1)
#     if os.path.exists(controlfile):
#         print("control.in was found")
#     else:
#         print("Error: Cannot find control.in.\n")
#         sys.exit(1)

# def shiftgeo(filename, c, delta, options):
#     """Function to shift geometries + save in corresponding directories"""

#     print(
#         "Shifting the geometry in direction  "
#         + str(c)
#         + " for delta  "
#         + str(delta)
#         + "\n"
#     )
#     lattice = []
#     fdata = []
#     element = []
#     lattice = []
#     folder = options.name + "_disp_" + str(delta)
#     if not os.path.exists(folder):
#         os.mkdir(folder)
#     with open(filename) as f:
#         ii = 0
#         i = 0
#         for line in f:
#             t = line.split()
#             if len(t) == 0:
#                 continue
#             if t[0] == "#":
#                 continue
#             if t[0] == "constrain_relaxation":
#                 continue
#             if t[0] == "lattice_vector":
#                 lattice += [(float(t[1]), float(t[2]), float(t[3]))]
#             elif t[0] == "atom":
#                 if line.rfind(options.name) != -1:
#                     i = i + 1
#                     if options.position:
#                         if i == options.position:
#                             t[c] = float(t[c]) + delta
#                             fdata += [(float(t[1]), float(t[2]), float(t[3]))]
#                             element += [(str(t[4]))]
#                             ii = ii + 1
#                         else:
#                             fdata += [(float(t[1]), float(t[2]), float(t[3]))]
#                             element += [(str(t[4]))]
#                     else:
#                         t[c] = float(t[c]) + delta
#                         fdata += [(float(t[1]), float(t[2]), float(t[3]))]
#                         element += [(str(t[4]))]
#                         ii = ii + 1

#                 else:
#                     fdata += [(float(t[1]), float(t[2]), float(t[3]))]
#                     element += [(str(t[4]))]
#             else:
#                 continue
#     fdata = np.array(fdata)
#     lattice = np.array(lattice)
#     element = np.array(element)

#     new_geo = open(folder + "/geometry.in", "w")
#     new_geo.write(
#         """#
#     lattice_vector """
#         + ((" %.8f" * 3) % tuple(lattice[0, :]))
#         + """
#     lattice_vector """
#         + ((" %.8f" * 3) % tuple(lattice[1, :]))
#         + """
#     lattice_vector """
#         + ((" %.8f" * 3) % tuple(lattice[2, :]))
#         + """
#     #
#     """
#     )
#     for i in range(0, len(fdata)):
#         new_geo.write(
#             "atom" + ((" %.8f" * 3) % tuple(fdata[i, :])) + " " +
#             element[i] + "\n"
#         )

#     new_geo.close()
#     return ii

# def precontrol(filename, delta, options):
#     """Function to copy and edit control.in"""
#     aimsout = "aims.out"
#     folder = options.name + "_disp_" + str(delta)
#     f = open(filename, "r")  # read control.in template
#     template_control = f.read()
#     f.close
#     if not os.path.exists(folder):
#         os.mkdir(folder)
#     new_control = open(folder + "/control.in", "w")
#     new_control.write(
#         template_control
#         + "KS_method serial \n"
#         + "output polarization    "
#         + str(1)
#         + " {} {} {}\n".format(options.nx[0], options.nx[1], options.nx[2])
#         + "output polarization    "
#         + str(2)
#         + " {} {} {}\n".format(options.ny[0], options.ny[1], options.ny[2])
#         + "output polarization    "
#         + str(3)
#         + " {} {} {}\n".format(options.nz[0], options.nz[1], options.nz[2])
#     )
#     new_control.close()
#     os.chdir(folder)
#     # Change directoy
#     if options.run_aims:
#         os.system(
#             options.run_aims + " > " + aimsout
#         )  # Run aims and pipe the output into a file named 'filename'
#     os.chdir("..")
#     time.sleep(2.4)

# ------------------------Post Processing-------------------------------#

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
        help="path to FHI-aims binary", default="/home/elia/Google-Drive/google-personal/FHIaims/build/aims.221103.scalapack.mpi.x"
    )
    parser.add_argument(
        "-s",
        "--suffix",
        action="store",
        help="Call suffix for binary e.g. 'mpirun -np 12 '",
        default="",
    )
    parser.add_argument(
        "-d",
        "--delta",
        action="store",
        type=float,
        nargs=2,
        dest="delta",
        help="finite difference poles, defualt is [0.01]",
        default=[0.01],
    )
    parser.add_argument(
        "-r", "--run", action="store", type=str2bool,
        help="perform DFT calculations", default=False
    )
    parser.add_argument(
        "-pp", "--postprocessing", action="store", type=str2bool,
        help="perform post-processing", default=True
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

    options = parser.parse_args()

    options.aims_run = options.suffix + " " + options.executable 
    options.aimsout = "FHI-aims.out"

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
                    
                    
                    newcpfile = folder + "/control.in"
                    print("\t\tcopying '%s' to '%s'"%(options.controlfile,newcpfile))
                    os.popen('cp %s %s'%(options.controlfile,newcpfile)) 

                    newgeofile = folder + "/geometry.in"
                    print("\t\twriting new geometry file to '%s'"%(newgeofile))
                    write(newgeofile,newdata)

                    print("\t\tcopying restart files to folder '%s'"%(folder))
                    os.popen('cp %s* %s/.'%(options.restartsuffix,folder)) 

                    print("\t\trunning calculation, output printed to '%s'"%(options.aimsout))
                    os.chdir(folder)
                    os.system(options.aims_run + " > " + options.aimsout)
                    os.chdir("..")

                    if is_complete(folder + "/" + options.aimsout,show=False):
                        print("\t\tcomputation completed")
                    else :
                        print("\t\tcomputation not completed")

                    print("\t\tremoving restart files from folder '%s'"%(folder))
                    os.popen('rm %s/%s*'%(folder,options.restartsuffix)) 

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

