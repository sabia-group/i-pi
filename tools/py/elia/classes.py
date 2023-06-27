from functions import get_one_file_in_folder,getproperty
import os
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
from ipi.utils.units import unit_to_internal

class Data:

    tab = "\t\t"
    check = True
    thr = 0.1
    check_orth = True
    fmt = "%20.12e"
    ofile = {"energy":"energy.txt",\
             "phases":"phases.txt",\
             "occupations":"occupations.txt",\
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
                file = get_one_file_in_folder(folder=options.modes,ext=".masses")
            else :
                file = options.masses

            print("{:s}reading masses from file '{:s}'".format(self.tab,file))
            masses = np.loadtxt(file)
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
            
            # arrays
            self.displacements = displacements
            self.velocities = velocities
            self.hess = hess
            self.eigvals = eigvals
            self.masses = masses
            self.dynmat = dynmat
            self.eigvec = eigvec

            # information
            self.Nconf = Nconf
            self.Nmodes = Nmodes

            self.ortho_modes = modes
            self.modes = Data.massexp(self.masses,"-1/2") @ self.eigvec
            self.proj = self.eigvec.T @ Data.massexp(self.masses,"1/2")

        elif what == "plot" :

            # eigvals
            file = get_one_file_in_folder(folder=options.modes,ext=".eigval")
            print("{:s}reading vibrational modes from file '{:s}'".format(Data.tab,file))
            self.eigvals = np.loadtxt(file)
            self.Nmodes = len(self.eigvals)
    
            file = Data.output_file(options.output,Data.ofile["energy"])
            print("{:s}reading energy from file '{:s}'".format(Data.tab,file))
            self.energy = np.loadtxt(file).T

            file = Data.output_file(options.output,Data.ofile["Aamp"])
            print("{:s}reading A-amplitudes from file '{:s}'".format(Data.tab,file))
            self.Aamplitudes = np.loadtxt(file).T

            if np.any(self.Aamplitudes.shape != self.energy.shape):
                raise ValueError("energy and A-amplitudes matrix size do not match")
            
        t,u = getproperty(options.property,["time"])
        self.time  = t["time"]
        self.units = u["time"]

        if u["time"] not in ["a.u.","atomic_unit"]:
            print("{:s}'time' is not in 'atomic units' but in '{:s}'".format(Data.tab,u["time"]))
            factor = unit_to_internal("time","femtosecond",1)
            print("{:s}converting 'time' to 'atomic units' by multiplication for {:>14.10e}".format(Data.tab,factor))
            self.time *= factor
            self.units = "a.u."

        pass

    @staticmethod
    def project_displacement(displ,proj):
        return proj @ displ
    
    @staticmethod
    def project_velocities(vel,proj,eigvals):
        N = len(eigvals)
        omega_inv = np.zeros((N,N))
        np.fill_diagonal(omega_inv,1.0/np.sqrt(eigvals))
        return omega_inv @ proj @ vel
    
    @staticmethod
    def potential_energy_per_mode(proj_displ,eigvals): #,hess=None,check=False):
        """return an array with the potential energy of each vibrational mode"""        
        return 0.5 * ( np.square(proj_displ).T * eigvals ).T #, 0.5 * proj_displ * omega_sqr @ proj_displ
    
    @staticmethod
    def kinetic_energy_per_mode(proj_vel,eigvals): #,check=False):
        """return an array with the kinetic energy of each vibrational mode"""        
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
                    self.proj,\
                    self.time ]
        
        if np.any( arrays is None ) :
            raise ValueError("Some arrays are missing")
        
        proj_displ = Data.project_displacement(self.displacements.T,self.proj)
        proj_vel   = Data.project_velocities  (self.velocities.T,   self.proj, self.eigvals)
        
        Vs = Data.potential_energy_per_mode(proj_displ,self.eigvals)
        Ks = Data.kinetic_energy_per_mode  (proj_vel,  self.eigvals)
        Es = Vs + Ks
        # Es_tot = Vs_tot + Ks_tot
        
        # V = np.sum(Vs)
        # K = np.sum(Ks)
        # E = np.sum(Es)

        # V_tot = np.sum(Vs_tot)
        # K_tot = np.sum(Ks_tot)
        # E_tot = np.sum(Es_tot)        

        # print("{:s}Summary:".format(self.tab))
        # print("{:s}pot. energy = {:>20.12e}".format(self.tab,V))
        # print("{:s}kin. energy = {:>20.12e}".format(self.tab,K))
        # print("{:s}tot. energy = {:>20.12e}".format(self.tab,E))

        # self.occupations = (2 * Es.T / self.eigvals)
        self.energy = self.occupations = self.phases = self.Aamplitudes = self.Bamplitudes = None 
    
        self.energy = Es.T
        self.occupations = self.energy / np.sqrt( self.eigvals) # - 0.5 # hbar = 1 in a.u.
        self.Aamplitudes  = np.sqrt( 2 * Es.T / self.eigvals  )

        self.Bamplitudes = Data.A2B(A=self.Aamplitudes,\
                                    N=self.ortho_modes,\
                                    M=self.masses,\
                                    E=self.eigvec)
        
        self.phases = np.arctan2(-proj_vel,proj_displ) - np.outer(np.sqrt( self.eigvals) , self.time)

        out = {"energy":self.energy,\
               "occupations":self.occupations,\
               "phases":self.phases,\
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
        np.savetxt(file,self.energy.T, fmt=Data.fmt)

        file = Data.output_file(folder,Data.ofile["occupations"])
        print("{:s}saving occupations to file '{:s}'".format(Data.tab,file))
        np.savetxt(file,self.occupations.T, fmt=Data.fmt)

        file = Data.output_file(folder,Data.ofile["phases"])
        print("{:s}saving phases to file '{:s}'".format(Data.tab,file))
        np.savetxt(file,self.phases.T, fmt=Data.fmt)

        file = Data.output_file(folder,Data.ofile["Aamp"])
        print("{:s}saving A-amplitudes to file '{:s}'".format(Data.tab,file))
        np.savetxt(file,self.Aamplitudes.T,fmt=Data.fmt)

        file = Data.output_file(folder,Data.ofile["Bamp"])
        print("{:s}saving B-amplitudes to file '{:s}'".format(Data.tab,file))
        np.savetxt(file,self.Bamplitudes.T,fmt=Data.fmt)

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
        xlim = ax.get_xlim()
        ax.hlines(1.0,xlim[0],xlim[1],linestyle="dashed",color="black",alpha=0.5)
        ax.set_xlim(*xlim)
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
