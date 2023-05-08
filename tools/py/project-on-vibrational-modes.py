# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

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
        "-q", "--positions", action="store", type=str,
        help="input file with the positions of all the configurations (in 'xyz' format)", default=None
    )
    parser.add_argument(
        "-r", "--relaxed", action="store", type=str,
        help="input file with the relaxed/original configuration (in 'xyz' format)", default=None
    )
    parser.add_argument(
        "-M", "--masses", action="store", type=str,
        help="input file with the nuclear masses (in 'txt' format)", default=None
    )
    parser.add_argument(
        "-v", "--velocities", action="store", type=str,
        help="input file with the velocities of all the configurations (in 'xyz' format)", default=None
    )
    parser.add_argument(
        "-m", "--modes", action="store", type=str,
        help="folder containing the vibrational modes computed by i-PI", default=None
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

    # parser.add_argument(
    #     "-s", "--signal", action="store", type=str,
    #     help="output file for the velocities projected on the vibrational modes", default=None
    # )
       
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
    check_orth = True

    def __init__(self,\
                 options,\
                 compute:bool=True,\
                 plot:bool=True):
        
        self.displacements = None
        self.velocities = None
        self.eigvals = None
        self.dynmat = None
        self.eigvec = None
        self.modes = None
        self.Nmodes = None
        self.Nconf = None
    
        if compute :

            from ase.io import read
            import os

            ###
            # reading original position
            print("\n{:s}reading original/relaxed position from file '{:s}'".format(self.tab,options.relaxed))
            relaxed = read(options.relaxed)

            if options.masses is None :
                print("\n{:s}storing nuclear masses from the original/relaxed position file using ASE".format(self.tab))
                masses = relaxed.get_masses()
            else:
                print("\n{:s}reading masses from file '{:s}'".format(self.tab,options.masses))
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
            print("\n{:s}reading positions from file '{:s}'".format(self.tab,options.positions))
            positions = read(options.positions,index=":")
            Nconf = len(positions) 

            ###
            # reading velocities
            print("\n{:s}reading velocities from file '{:s}'".format(self.tab,options.velocities))
            velocities = read(options.velocities,index=":")
            Nvel = len(velocities)
            print("{:s}read {:d} configurations".format(self.tab,Nconf))
            if Nvel != Nconf :
                raise ValueError("number of velocities and positions configuration are different")

            ###
            # reading vibrational modes
            if os.path.isdir(options.modes):
                print("\n{:s}searching for '*.mode' file in folder '{:s}'".format(self.tab,options.modes))
                
                # modes
                file = get_one_file_in_folder(folder=options.modes,ext=".mode")
                print("\n{:s}reading vibrational modes from file '{:s}'".format(self.tab,file))
                modes = np.loadtxt(file)
                if modes.shape[0] != Nmodes or modes.shape[1] != Nmodes :
                    raise ValueError("vibrational modes matrix with wrong size")
                
                # eigenvectors
                file = get_one_file_in_folder(folder=options.modes,ext=".eigvec")
                print("\n{:s}reading eigenvectors from file '{:s}'".format(self.tab,file))
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
                print("\n{:s}reading vibrational modes from file '{:s}'".format(self.tab,file))
                hess = np.loadtxt(file)
                if hess.shape[0] != Nmodes or hess.shape[1] != Nmodes:
                    raise ValueError("hessian matrix with wrong size")
                
                # eigval
                file = get_one_file_in_folder(folder=options.modes,ext=".eigval")
                print("\n{:s}reading vibrational modes from file '{:s}'".format(self.tab,file))
                eigval = np.loadtxt(file)
                if len(eigval) != Nmodes:
                    raise ValueError("eigenvalues array with wrong size")
                
                # dynmat
                file = get_one_file_in_folder(folder=options.modes,ext=".dynmat")
                print("\n{:s}reading the dynamical matrix from file '{:s}'".format(self.tab,file))
                dynmat = np.loadtxt(file)
                if dynmat.shape[0] != Nmodes or dynmat.shape[1] != Nmodes:
                    raise ValueError("dynamical matrix with wrong size")

                print("\n{:s}read {:d} modes".format(self.tab,Nmodes))

            else:
                raise ValueError("'--modes' should be a folder")

            if modes.shape[0] != modes.shape[1]:
                raise ValueError("vibrtional mode matrix is not square")

            if not np.all(np.asarray([ positions[i].positions.flatten().shape for i in range(Nconf)]) == Nmodes) :
                raise ValueError("some configurations do not have the correct shape")
            
            # if self.check :
                #     print("\n{:s}Let's do a little test".format(self.tab))
                #     mode      = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".mode"))
                #     dynmat    = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".dynmat"))
                #     full_hess = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext="_full.hess"))
                #     eigval    = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".eigval"))
                #     eigvec    = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".eigvec"))
                #     hess      = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".hess"))
                    
                #     print("{:s}checking that D@V = E@V".format(self.tab))
                #     res = np.sqrt(np.square(dynmat @ eigvec - eigval @ eigvec).sum())
                #     print("{:s} | D@V - E@V | = {:>20.12e}".format(self.tab,res))

                #     eigsys = np.linalg.eigh(mode)

                #     print("{:s}checking that eigvec(M) = M".format(self.tab))
                #     res = np.sqrt(np.square(eigsys[1] - mode).flatten().sum())
                #     print("{:s} | eigvec(H) - M | = {:>20.12e}".format(self.tab,res))

                #     print("{:s}checking that eigval(H) = E".format(self.tab))
                #     res = np.sqrt(np.square( np.sort(eigsys[0]) - np.sort(eigval)).sum())
                #     print("{:s} | eigvec(H) - E | = {:>20.12e}".format(self.tab,res))

                #     print("{:s}checking that H@eigvec(H) = eigval(H)@eigvec(H)".format(self.tab))
                #     res = np.sqrt(np.square(eigsys[0] - eigval).sum())
                #     print("{:s} | eigvec(H) - E | = {:>20.12e}".format(self.tab,res))
            
            ###
            # flatten the displacements
            for n in range(Nconf):
                positions[n] = positions[n].positions.flatten()
            displacements = np.asarray(positions) - relaxed.flatten()

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
            self.modes = modes
            self.hess = hess
            self.eigval = eigval
            self.masses = masses
            self.dynmat = dynmat
            self.eigvec = eigvec

            # information
            self.Nconf = Nconf
            self.Nmodes = Nmodes

            M = np.eye(len(modes))
            np.fill_diagonal(M,1.0/np.sqrt(self.masses))
            a = M @ self.eigvec
            self.ortho_modes = modes
            self.modes = a
            # b = a.copy()
            # for i in range(len(b)):
            #     b[:,i] /= np.linalg.norm(b[:,i])
            # print( np.linalg.norm( (b - self.modes ).flatten() ) )
            # print( np.linalg.norm( ( M @ hess @ M - self.dynmat ).flatten() ) )
            # eigsys = np.linalg.eigh(self.dynmat)
            # print( np.linalg.norm( ( eigsys[0] - eigval ) ) )

        if plot :
            print("\n{:s}reading modes occupations from file '{:s}'".format(self.tab,options.occupations))
            occupations = np.loadtxt(options.occupations)

            self.occupations = occupations
            Nmodes = occupations.shape[1]
            self.Nmodes = Nmodes

        pass

    @staticmethod
    def potential_energy_per_mode(displ,modes,eigvals): #,hess=None,check=False):
        """return an array with the potential energy of each vibrational mode"""        

        proj_displ = np.linalg.inv(modes) @ displ
        
        # if check :
        #     omega_sqr = modes.T @ hess @ modes

        #     N = len(eigvals)
        #     eig_sqr = np.zeros((N,N))
        #     np.fill_diagonal(eig_sqr,eigvals)

        #     print("{:s}checking that N^t @ Phi @ N = W^2 (N is the non-normalized vibrational modes matrix)".format(Data.tab))
        #     res = np.linalg.norm(omega_sqr - eig_sqr)
        #     print("{:s} | N^t @ Phi @ N - W^2 | = {:>20.12e}".format(Data.tab,res))
            
        #     print("{:s}checking the off diagonal elements of A = N^t @ Phi @ N".format(Data.tab))
        #     omega_sqr_copy = omega_sqr.copy()
        #     np.fill_diagonal(omega_sqr_copy,0)
        #     res = np.linalg.norm(omega_sqr_copy)
        #     print("{:s} | A - diag(A) | = {:>20.12e}".format(Data.tab,res))

        return 0.5 * ( np.square(proj_displ).T * eigvals ).T #, 0.5 * proj_displ * omega_sqr @ proj_displ
    
    @staticmethod
    def kinetic_energy_per_mode(vel,modes,eigvals): #,check=False):
        """return an array with the kinetic energy of each vibrational mode"""        

        N = len(eigvals)
        omega_inv = np.zeros((N,N))
        np.fill_diagonal(omega_inv,1.0/np.sqrt(eigvals))
        proj_vel = omega_inv @ np.linalg.inv(modes) @ vel
        
        # if check :
        #     omega = np.fill_diagonal(omega_inv,np.sqrt(eigvals))

        #     print("{:s}checking that E^t @ M @ E = Id".format(Data.tab))
        #     res = np.linalg.norm(identity - np.eye(len(identity),1))
        #     print("{:s} | E^t @ M @ E - Id | = {:>20.12e}".format(Data.tab,res))
            
        #     print("{:s}checking the off diagonal elements of A = E^t @ M @ E".format(Data.tab))
        #     identity_copy = identity.copy()
        #     np.fill_diagonal(identity_copy,0)
        #     res = np.linalg.norm(identity_copy)
        #     print("{:s} | A - diag(A) | = {:>20.12e}".format(Data.tab,res))

        return 0.5 * ( np.square(proj_vel).T * eigvals ).T #, 0.5 * ( proj_vel * eigvals ) * identity @ ( eigvals * proj_vel )


    def compute_occ(self):
        
        arrays = [  self.displacements,\
                    self.velocities,\
                    self.modes, \
                    self.hess, \
                    self.eigval, \
                    self.Nmodes, \
                    self.dynmat, \
                    self.eigvec, \
                    self.Nconf,\
                    self.masses ]
        
        if np.any( arrays is None ) :
            raise ValueError("Some arrays are missing")
        
        Vs = Data.potential_energy_per_mode(self.displacements.T,self.modes, self.eigval) #, self.hess, check=True)
        Ks = Data.kinetic_energy_per_mode  (self.velocities.T,   self.modes, self.eigval) #, self.masses, check=True)
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

        # self.occupations = (2 * Es.T / self.eigval)
        self.energy = Es.T
        self.occupations = Es.T / np.sqrt( self.eigval) # - 0.5
        self.amplitudes  = np.sqrt( 2 * Es.T / self.eigval  )
        
        # print("\n{:s}pot. energy (with off diag.) = {:>20.12e}".format(self.tab,V_tot))
        # print("\n{:s}kin. energy (with off diag.) = {:>20.12e}".format(self.tab,K_tot))
        # print("\n{:s}tot. energy (with off diag.) = {:>20.12e}".format(self.tab,E_tot))

        # print("\n{:s}Delta pot. energy = {:>20.12e}".format(self.tab,V-V_tot))
        # print("\n{:s}Delta kin. energy = {:>20.12e}".format(self.tab,K-K_tot))

        return self.occupations.copy()

def main():
    """main routine"""

    ###
    # prepare/read input arguments
    print("\n\tReding input arguments")
    options = prepare_parser()

    ###
    # read input argumfilesents
    print("\n\tReding input files")
    data = Data(options,compute=options.compute,plot=options.plot is not None)
    
    ###
    # compute occupations
    if options.compute :
        print("\n\tComputing occupations")
        occ = data.compute_occ() # occupations are stored in 'data.occupations'
        np.savetxt("occupations.txt",occ,fmt="%20.12e")
        np.savetxt("energy.txt",data.energy,fmt="%20.12e")
        np.savetxt("amplitudes.txt",data.amplitudes,fmt="%20.12e")

    ###
    # plot occupations
    if options.plot:

        import matplotlib.pyplot as plt

        print("\n\tplotting modes occupations")

        plt.figure()
        for n in range(data.Nmodes):
            plt.plot(data.occupations[:,n],label=str(n))
        
        plt.yscale('log')
        plt.grid()
        plt.legend()

        print("\n\tsaving plot to file '{:s}'".format(options.plot))
        plt.savefig(options.plot)


    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()