# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

import argparse
import numpy as np
from ase.io import read
import os
import re
import matplotlib.pyplot as plt

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
        "-q", "--positions", action="store", type=str,
        help="input file with the positions of all the configurations (in 'xyz' format)"#, default=None
    )
    parser.add_argument(
        "-r", "--relaxed", action="store", type=str,
        help="input file with the relaxed/original configuration (in 'xyz' format)"#, default=None
    )
    parser.add_argument(
        "-M", "--masses", action="store", type=str,
        help="input file with the nuclear masses (in 'txt' format)"#, default=None
    )
    parser.add_argument(
        "-v", "--velocities", action="store", type=str,
        help="input file with the velocities of all the configurations (in 'xyz' format)"#, default=None
    )
    parser.add_argument(
        "-m", "--modes", action="store", type=str,
        help="folder containing the vibrational modes computed by i-PI"#, default=None
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="output folder", default="output"
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
        "-t", "--t-min", action="store", type=int,
        help="minimum time to be plotted", default=0
    )
    parser.add_argument(
        "-pr", "--properties", action="store", type=str,
        help="file containing the properties at each MD step computed by i-PI", default=None
    )
       
    options = parser.parse_args()

    return options

def get_one_file_in_folder(folder,ext):
    files = list()
    for file in os.listdir(folder):
        if file.endswith(ext):
            files.append(os.path.join(folder, file))
    if len(files) == 0 :
        raise ValueError("no '*{:s}' files found".format(ext))
    elif len(files) > 1 :
        raise ValueError("more than one '*{:s}' file found".format(ext))
    return files[0]

def getproperty(inputfile, propertyname,data=None,skip="0"):

    def check(p,l):
        if not l.find(p) :
            return False # not found
        elif l[l.find(p)-1] != " ":
            return False # composite word
        elif l[l.find(p)+len(p)] == "{":
            return True
        elif l[l.find(p)+len(p)] != " " :
            return False # composite word
        else :
            return True

    if type(propertyname) in [list,np.ndarray]: 
        out   = dict()
        units = dict()
        data = np.loadtxt(inputfile)
        for p in propertyname:
            out[p],units[p] = getproperty(inputfile,p,data,skip=skip)
        return out,units
    
    print("\tsearching for '{:s}'".format(propertyname))

    skip = int(skip)

    # propertyname = " " + propertyname + " "

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
                if check(propertyname,line):
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

                    l = line
                    p = propertyname
                    if l[l.find(p)+len(p)] == "{":
                        unit = l.split("{")[1].split("}")[0]
                    else :
                        unit = "atomic_unit"

                # get new line
                line = ifile.readline()
                if len(line) == 0:
                    raise EOFError
            if icol <= 0:
                print("Could not find " + propertyname + " in file " + inputfile)
                raise EOFError
            else :
                return np.asarray(output),unit

        except EOFError:
            break

class Data:

    tab = "\t\t"
    check = True
    thr = 0.1
    check_orth = True
    fmt = "%20.12e"
    ofile = {"energy":"energy.txt",\
             "Aamp":"A-amplitudes.txt",\
             "Bamp":"B-amplitudes.txt",\
             "violin":"violin.csv"}

    def __init__(self,\
                 options,\
                 what="compute"):
        
        self.displacements = None
        self.velocities = None
        self.eigvals = None
        self.dynmat = None
        self.eigvec = None
        self.modes = None
        self.Nmodes = None
        self.Nconf = None

        if not os.path.isdir(options.modes):
            raise ValueError("'--modes' should be a folder")
    
        if what == "compute" :

            ###
            # reading original position
            print("{:s}reading original/relaxed position from file '{:s}'".format(self.tab,options.relaxed))
            relaxed = read(options.relaxed)

            if options.masses is None :
                print("{:s}storing nuclear masses from the original/relaxed position file using ASE".format(self.tab))
                masses = relaxed.get_masses()
            else:
                print("{:s}reading masses from file '{:s}'".format(self.tab,options.masses))
                masses = np.loadtxt(options.masses)
                if len(masses) == len(relaxed.positions) :
                    # set masses
                    M = np.zeros((3 * len(masses)), float)
                    M[ 0 : 3 * len(masses) : 3] = masses
                    M[ 1 : 3 * len(masses) : 3] = masses
                    M[ 2 : 3 * len(masses) : 3] = masses
                    masses = M

                elif len(masses) != 3 * len(relaxed.positions):            
                    raise ValueError("wrong number of nuclear masses")
                            
            # positions
            relaxed = relaxed.positions
            Nmodes = relaxed.shape[0] * 3

            ###
            # reading positions
            print("{:s}reading positions from file '{:s}'".format(self.tab,options.positions))
            positions = read(options.positions,index=":")
            Nconf = len(positions) 

            ###
            # reading velocities
            print("{:s}reading velocities from file '{:s}'".format(self.tab,options.velocities))
            velocities = read(options.velocities,index=":")
            Nvel = len(velocities)
            print("{:s}read {:d} configurations".format(self.tab,Nconf))
            if Nvel != Nconf :
                raise ValueError("number of velocities and positions configuration are different")

            ###
            # reading vibrational modes
            
            print("{:s}searching for '*.mode' file in folder '{:s}'".format(self.tab,options.modes))
            
            # modes
            file = get_one_file_in_folder(folder=options.modes,ext=".mode")
            print("{:s}reading vibrational modes from file '{:s}'".format(self.tab,file))
            modes = np.loadtxt(file)
            if modes.shape[0] != Nmodes or modes.shape[1] != Nmodes :
                raise ValueError("vibrational modes matrix with wrong size")
            
            # eigenvectors
            file = get_one_file_in_folder(folder=options.modes,ext=".eigvec")
            print("{:s}reading eigenvectors from file '{:s}'".format(self.tab,file))
            eigvec = np.loadtxt(file)
            if eigvec.shape[0] != Nmodes or eigvec.shape[1] != Nmodes:
                raise ValueError("eigenvectors matrix with wrong size")
            
            # check that the eigenvectors are orthogonal (they could not be so)
            if Data.check_orth :                
                print("{:s}checking that the eigenvectors are orthonormal, i.e. M @ M^t = Id".format(self.tab))
                res = np.linalg.norm(eigvec @ eigvec.T - np.eye(Nmodes))
                print("{:s} | M @ M^t - Id | = {:>20.12e}".format(self.tab,res))
                if res > Data.thr :
                    raise ValueError("the eigenvectors are not orthonormal")

            # hess
            file = get_one_file_in_folder(folder=options.modes,ext="phonons.hess")
            print("{:s}reading vibrational modes from file '{:s}'".format(self.tab,file))
            hess = np.loadtxt(file)
            if hess.shape[0] != Nmodes or hess.shape[1] != Nmodes:
                raise ValueError("hessian matrix with wrong size")
            
            # eigvals
            file = get_one_file_in_folder(folder=options.modes,ext=".eigval")
            print("{:s}reading vibrational modes from file '{:s}'".format(self.tab,file))
            eigvals = np.loadtxt(file)
            if len(eigvals) != Nmodes:
                raise ValueError("eigenvalues array with wrong size")
            
            # dynmat
            file = get_one_file_in_folder(folder=options.modes,ext=".dynmat")
            print("{:s}reading the dynamical matrix from file '{:s}'".format(self.tab,file))
            dynmat = np.loadtxt(file)
            if dynmat.shape[0] != Nmodes or dynmat.shape[1] != Nmodes:
                raise ValueError("dynamical matrix with wrong size")

            print("{:s}read {:d} modes".format(self.tab,Nmodes))                

            if modes.shape[0] != modes.shape[1]:
                raise ValueError("vibrtional mode matrix is not square")

            if not np.all(np.asarray([ positions[i].positions.flatten().shape for i in range(Nconf)]) == Nmodes) :
                raise ValueError("some configurations do not have the correct shape")
            
            # if self.check :
                #     print("\n{:s}Let's do a little test".format(self.tab))
                #     mode      = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".mode"))
                #     dynmat    = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".dynmat"))
                #     full_hess = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext="_full.hess"))
                #     eigvals    = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".eigvals"))
                #     eigvec    = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".eigvec"))
                #     hess      = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".hess"))
                    
                #     print("{:s}checking that D@V = E@V".format(self.tab))
                #     res = np.sqrt(np.square(dynmat @ eigvec - eigvals @ eigvec).sum())
                #     print("{:s} | D@V - E@V | = {:>20.12e}".format(self.tab,res))

                #     eigsys = np.linalg.eigh(mode)

                #     print("{:s}checking that eigvec(M) = M".format(self.tab))
                #     res = np.sqrt(np.square(eigsys[1] - mode).flatten().sum())
                #     print("{:s} | eigvec(H) - M | = {:>20.12e}".format(self.tab,res))

                #     print("{:s}checking that eigvals(H) = E".format(self.tab))
                #     res = np.sqrt(np.square( np.sort(eigsys[0]) - np.sort(eigvals)).sum())
                #     print("{:s} | eigvec(H) - E | = {:>20.12e}".format(self.tab,res))

                #     print("{:s}checking that H@eigvec(H) = eigvals(H)@eigvec(H)".format(self.tab))
                #     res = np.sqrt(np.square(eigsys[0] - eigvals).sum())
                #     print("{:s} | eigvec(H) - E | = {:>20.12e}".format(self.tab,res))
            
            ###
            # flatten the displacements
            for n in range(Nconf):
                positions[n] = positions[n].positions.flatten()
            displacements = np.asarray(positions) - np.asarray(positions[0])#- 1.88972612463*relaxed.flatten()

            ###
            # flatten the velocities
            for n in range(Nconf):
                velocities[n] = velocities[n].positions.flatten()
            velocities = np.asarray(velocities)
            
            
            # ###
            # # project on phonon modes
            # # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#scipy.signal.hilbert
            # print("\n\tprojecting displacements on vibrational modes")
            # signal = displacement @ modes

            # if options.signal is not None :
            #     print("\tsaving displacements projected on the vibrational modes to file '{:s}'".format(options.signal))
            #     np.savetxt(options.signal,signal,delimiter=" ",fmt="%20.12e")

            # print("\tcomputing the analytic signal of the displacements along the vibrational modes")
            # analytic_signal = hilbert(signal,axis=0)

            # print("\tcomputing the time-dependent occupations of the vibrational modes")
            # occupations = np.absolute(analytic_signal)
            
            # ###
            # # save occupations to file  
            # print("\tsaving modes occupations to file '{:s}'".format(options.occupations))
            # np.savetxt(options.occupations,occupations,delimiter=" ",fmt="%20.12e")

            # arrays
            self.displacements = displacements
            self.velocities = velocities
            # self.modes = modes
            self.hess = hess
            self.eigvals = eigvals
            self.masses = masses
            self.dynmat = dynmat
            self.eigvec = eigvec

            # information
            self.Nconf = Nconf
            self.Nmodes = Nmodes

            # M = np.eye(len(modes))
            # np.fill_diagonal(M,1.0/np.sqrt(self.masses))
            # a = M @ self.eigvec
            self.ortho_modes = modes
            self.modes = Data.massexp(self.masses,"-1/2") @ self.eigvec
            # b = self.modes.copy()
            # for i in range(len(b)):
            #     b[:,i] /= np.linalg.norm(b[:,i])
            # print( np.linalg.norm( b - self.ortho_modes ) )
            # print( np.linalg.norm( M @ hess @ M - self.dynmat ) )
            # eigsys = np.linalg.eigh(self.dynmat)
            # print( np.linalg.norm( ( eigsys[0] - eigvals ) ) )
            
            # M = np.eye(len(modes))
            # np.fill_diagonal(M,np.sqrt(self.masses))
            self.proj = self.eigvec.T @ Data.massexp(self.masses,"1/2")

        elif what == "plot" :

            # eigvals
            file = get_one_file_in_folder(folder=options.modes,ext=".eigval")
            print("{:s}reading vibrational modes from file '{:s}'".format(Data.tab,file))
            self.eigvals = np.loadtxt(file)
            self.Nmodes = len(self.eigvals)
    
            file = Data.output_file(options.output,Data.ofile["energy"])
            print("{:s}reading energy from file '{:s}'".format(Data.tab,file))
            self.energy = np.loadtxt(file)

            file = Data.output_file(options.output,Data.ofile["Aamp"])
            print("{:s}reading A-amplitudes from file '{:s}'".format(Data.tab,file))
            self.Aamplitudes = np.loadtxt(file)

            if np.any(self.Aamplitudes.shape != self.energy.shape):
                raise ValueError("energy and A-amplitudes matrix size do not match")
            
            t,u = getproperty(options.properties,["time"])
            self.time  = t["time"]
            self.units = u["time"]

        pass

    @staticmethod
    def potential_energy_per_mode(displ,proj,eigvals): #,hess=None,check=False):
        """return an array with the potential energy of each vibrational mode"""        

        # proj_displ = np.linalg.inv(modes) @ displ
        proj_displ = proj @ displ
        return 0.5 * ( np.square(proj_displ).T * eigvals ).T #, 0.5 * proj_displ * omega_sqr @ proj_displ
    
    @staticmethod
    def kinetic_energy_per_mode(vel,proj,eigvals): #,check=False):
        """return an array with the kinetic energy of each vibrational mode"""        

        N = len(eigvals)
        omega_inv = np.zeros((N,N))
        np.fill_diagonal(omega_inv,1.0/np.sqrt(eigvals))
        # proj_vel = omega_inv @ np.linalg.inv(modes) @ vel
        proj_vel = omega_inv @ proj @ vel
        return 0.5 * ( np.square(proj_vel).T * eigvals ).T #, 0.5 * ( proj_vel * eigvals ) * identity @ ( eigvals * proj_vel )

    @staticmethod
    def massexp(M,exp):
        out = np.eye(len(M))        
        if exp == "-1":
            np.fill_diagonal(out,1.0/M)
        elif exp == "1/2":
            np.fill_diagonal(out,np.sqrt(M))
        elif exp == "-1/2":
            np.fill_diagonal(out,1.0/np.sqrt(M))
        else :
            raise ValueError("'exp' value not allowed")
        return out       

    @staticmethod
    def A2B(A,N,M,E):
        """
        purpose:
            convert the A-amplitude [length x mass^{-1/2}] into B-amplitudes [length]

        input :
            A : A-amplitudes
            N : normal modes (normalized)
            M : masses
            E : eigevectors (of the dynamical matrix)

        output:
            B : B-amplitudes
        """
        
        # print("A shape : ",A.shape)
        # print("N shape : ",N.shape)
        # print("M shape : ",M.shape)
        # print("E shape : ",E.shape)

        B = np.diag( np.linalg.inv(N) @ Data.massexp(M,"-1/2") @ E ) * A
        # print("B shape : ",B.shape)
        return B

    def compute(self):
        
        arrays = [  self.displacements,\
                    self.velocities,\
                    self.modes, \
                    self.hess, \
                    self.eigvals, \
                    self.Nmodes, \
                    self.dynmat, \
                    self.eigvec, \
                    self.Nconf,\
                    self.masses,\
                    self.ortho_modes,\
                    self.proj ]
        
        if np.any( arrays is None ) :
            raise ValueError("Some arrays are missing")
        
        Vs = Data.potential_energy_per_mode(self.displacements.T,self.proj, self.eigvals) #, self.hess, check=True)
        Ks = Data.kinetic_energy_per_mode  (self.velocities.T,   self.proj, self.eigvals) #, self.masses, check=True)
        Es = Vs + Ks
        # Es_tot = Vs_tot + Ks_tot
        
        V = np.sum(Vs)
        K = np.sum(Ks)
        E = np.sum(Es)

        # V_tot = np.sum(Vs_tot)
        # K_tot = np.sum(Ks_tot)
        # E_tot = np.sum(Es_tot)        

        print("{:s}Summary:".format(self.tab))
        print("{:s}pot. energy = {:>20.12e}".format(self.tab,V))
        print("{:s}kin. energy = {:>20.12e}".format(self.tab,K))
        print("{:s}tot. energy = {:>20.12e}".format(self.tab,E))

        # self.occupations = (2 * Es.T / self.eigvals)
        self.energy = self.occupations = self.Aamplitudes = self.Bamplitudes = None 
    
        self.energy = Es.T
        # self.occupations = Es.T / np.sqrt( self.eigvals) # - 0.5
        self.Aamplitudes  = np.sqrt( 2 * Es.T / self.eigvals  )

        self.Bamplitudes = Data.A2B(A=self.Aamplitudes,\
                                    N=self.ortho_modes,\
                                    M=self.masses,\
                                    E=self.eigvec)

        out = {"energy":self.energy,\
               "occupations":self.occupations,\
               "A-amplitudes":self.Aamplitudes,\
               "B-amplitudes":self.Bamplitudes}
        
        # print("\n{:s}pot. energy (with off diag.) = {:>20.12e}".format(self.tab,V_tot))
        # print("\n{:s}kin. energy (with off diag.) = {:>20.12e}".format(self.tab,K_tot))
        # print("\n{:s}tot. energy (with off diag.) = {:>20.12e}".format(self.tab,E_tot))

        # print("\n{:s}Delta pot. energy = {:>20.12e}".format(self.tab,V-V_tot))
        # print("\n{:s}Delta kin. energy = {:>20.12e}".format(self.tab,K-K_tot))

        return out

    @staticmethod
    def output_folder(folder):
        if folder in ["",".","./"] :
            folder = "."
        elif not os.path.exists(folder) :
            print("\n\tCreating directory '{:s}'".format(folder))
            os.mkdir(folder)
        return folder
    
    @staticmethod
    def output_file(folder,what):
        folder = Data.output_folder(folder)
        return "{:s}/{:s}".format(folder,what)

    def save(self,folder):

        file = Data.output_file(folder,Data.ofile["energy"])
        print("{:s}saving energy to file '{:s}'".format(Data.tab,file))
        np.savetxt(file,self.energy, fmt=Data.fmt)

        file = Data.output_file(folder,Data.ofile["Aamp"])
        print("{:s}saving A-amplitudes to file '{:s}'".format(Data.tab,file))
        np.savetxt(file,self.Aamplitudes,fmt=Data.fmt)

        file = Data.output_file(folder,Data.ofile["Bamp"])
        print("{:s}saving B-amplitudes to file '{:s}'".format(Data.tab,file))
        np.savetxt(file,self.Bamplitudes,fmt=Data.fmt)

        pass

    def plot(self,options):

        if options.t_min > 0 :            
            print("\tSkipping the {:d} {:s}".format(options.t_min,self.units))
            i = np.where( self.time >= options.t_min )[0][0]
            print("\tthen skipping the first {:d} MD steps".format(i))
            self.Aamplitudes = self.Aamplitudes[i:,:]
            self.energy = self.energy[i:,:] 
            self.time   = self.time[i:]

        Ndof = self.Aamplitudes.shape[1]
        normalization = self.energy.sum(axis=1) / Ndof

        normalized_occupations = np.zeros(self.Aamplitudes.shape)
        for i in range(Ndof):
            normalized_occupations[:,i] = np.square(self.Aamplitudes[:,i])  * self.eigvals[i] / ( 2*normalization[i] )

        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(self.time,normalized_occupations)

        # plt.title('LiNbO$_3$ (NVT@$20K$,$\\Delta t = 1fs$,T$=20-50ps$,$\\tau=10fs$)')
        ax.set_ylabel("$A^2_s\\omega^2_s / \\left( 2 N \\right)$ with $N=E_{harm}\\left(t\\right)$")
        ax.set_xlabel("time ({:s})".format("fs" if self.units == "femtosecond" else "a.u."))
        ax.set_xlim(min(self.time),max(self.time))
        ylim = ax.get_ylim()
        ax.set_ylim(0,ylim[1])
        # ax.set_yscale("log")

        plt.grid()
        plt.tight_layout()
        plt.savefig(options.plot)

        ###
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

        mean = np.mean(normalized_occupations,axis=0)
        std = np.mean(normalized_occupations,axis=0)
        if len(mean) != Ndof or len(std) != Ndof:
            raise ValueError("wrong array size for barplot")

        fig, ax = plt.subplots(figsize=(10,6))
        w = np.sqrt(self.eigvals)
        # ax.scatter(x=w,y=mean,color="navy")
        ax.errorbar(x=w,y=mean,yerr=std,color="red",ecolor="navy",fmt="o")

        # plt.title('LiNbO$_3$ (NVT@$20K$,$\\Delta t = 1fs$,T$=20-50ps$,$\\tau=10fs$)')
        ax.set_ylabel("$A^2_s\\omega^2_s / \\left( 2 N \\right)$ with $N=E_{harm}\\left(t\\right)$")
        ax.set_xlabel("$\\omega$ (a.u.)")
        #ax.set_xlim(min(self.time),max(self.time))
        #ylim = ax.get_ylim()
        #ax.set_ylim(0,ylim[1])
        # ax.set_yscale("log")

        plt.grid()
        plt.tight_layout()
        tmp = os.path.splitext(options.plot)
        file = "{:s}.{:s}{:s}".format(tmp[0],"mean-std",tmp[1])
        # plt.show()
        plt.savefig(file)

        import pandas as pd
        df = pd.DataFrame(columns=["w","mean","std"])
        df["w"] = w
        df["mean"] = mean
        df["std"] = std
        file = file = Data.output_file(options.output,Data.ofile["violin"])
        df.to_csv(file,index=False)

        pass

def main():
    """main routine"""

    ###
    # prepare/read input arguments
    print("\n\tReding input arguments")
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