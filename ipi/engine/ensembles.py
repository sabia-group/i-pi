"""Contains the classes that deal with the different dynamics required in
different types of ensembles.

Holds the algorithms required for normal mode propagators, and the objects to
do the constant temperature and pressure algorithms. Also calculates the
appropriate conserved energy quantity for the ensemble of choice.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np
from numpy import linalg

from ipi.utils.messages import warning,verbosity
from ipi.utils.depend import dd
from ipi.utils.units import Constants
from ipi.engine.thermostats import *
from ipi.engine.barostats import *
from ipi.engine.motion.alchemy import *
from ipi.engine.forces import Forces, ScaledForceComponent
#from ipi.engine.motion.polarization import *

__all__ = ["Ensemble", "ensemble_swap"]

# IMPORTANT - THIS MUST BE KEPT UP-TO-DATE WHEN THE ENSEMBLE CLASS IS CHANGED


def ensemble_swap(ens1, ens2):
    """Swaps the definitions of the two ensembles, by
    exchanging all of the inner properties."""

    if ens1.temp != ens2.temp:
        ens1.temp, ens2.temp = ens2.temp, ens1.temp
    if ens1.pext != ens2.pext:
        ens1.pext, ens2.pext = ens2.pext, ens1.pext
    if np.linalg.norm(ens1.stressext - ens2.stressext) > 1e-10:
        tmp = dstrip(ens1.stressext).copy()
        ens1.stressext[:] = ens2.stressext
        ens2.stressext[:] = tmp
    if len(ens1.bweights) != len(ens2.bweights):
        raise ValueError(
            "Cannot exchange ensembles that have different numbers of bias components"
        )
    if len(ens1.hweights) != len(ens2.hweights):
        raise ValueError(
            "Cannot exchange ensembles that are described by different forces"
        )
    if not np.array_equal(ens1.bweights, ens2.bweights):
        ens1.bweights, ens2.bweights = (
            dstrip(ens2.bweights).copy(),
            dstrip(ens1.bweights).copy(),
        )
    if not np.array_equal(ens1.hweights, ens2.hweights):
        ens1.hweights, ens2.hweights = (
            dstrip(ens2.hweights).copy(),
            dstrip(ens1.hweights).copy(),
        )


class Ensemble(dobject):

    """Base ensemble class.

    Defines the thermodynamic state of the system.

    Depend objects:
        temp: The system's temperature.
        pext: The systems's pressure
        stressext: The system's stress tensor
        bias: Explicit bias forces
    """

    def __init__(
        self,
        eens=0.0,
        econs=0.0,
        temp=None,
        pext=None,
        stressext=None,
        bcomponents=None,
        bweights=None,
        hweights=None,
        time=0.0,
        Eamp=None,
        Efreq=None,
        Ephase=None,
        Epeak=None,
        Esigma=None,
        BEC=None,
        cpol=True
    ):
        """Initialises Ensemble.

        Args:
            temp: The temperature.
            fixcom: An optional boolean which decides whether the centre of mass
                motion will be constrained or not. Defaults to False.
        """
        dself = dd(self)

        dself.temp = depend_value(name="temp")
        if temp is not None:
            self.temp = temp
        else:
            self.temp = -1.0

        dself.stressext = depend_array(name="stressext", value=np.zeros((3, 3), float))
        if stressext is not None:
            self.stressext = np.reshape(np.asarray(stressext), (3, 3))
        else:
            self.stressext = -1.0

        dself.pext = depend_value(name="pext")
        if pext is not None:
            self.pext = pext
        else:
            self.pext = -1.0

        dself.eens = depend_value(name="eens")
        if eens is not None:
            self.eens = eens
        else:
            self.eens = 0.0

        # the bias force contains two bits: explicit biases (that are meant to represent non-physical external biasing potentials)
        # and hamiltonian weights (that will act by scaling different physical components of the force). Both are bound as components
        # of the "bias force" evaluator, but their meaning (and the wiring further down in bind()) differ.

        # these are the additional bias components
        if bcomponents is None:
            bcomponents = []
        self.bcomp = bcomponents
        self.bias = Forces()

        # and their weights
        if bweights is None or len(bweights) == 0:
            bweights = np.ones(len(self.bcomp))

        dself.bweights = depend_array(name="bweights", value=np.asarray(bweights))

        # weights of the Hamiltonian scaling
        if hweights is None:
            hweights = np.ones(0)
        self.hweights = np.asarray(hweights)

        # ES

        if Epeak is not None and Epeak < 0 :
            raise ValueError("Epeak < 0: the peak of the external electric field can only be positive")    
        if Esigma is not None and Esigma < 0 :
            raise ValueError("Esigma < 0: the standard deviation of the gaussian envelope function of the external electric field has to be positive") 

        # Internal time counter
        dself.time = depend_value(name="time",value=time)

        # ES: we do I need to specify default values here too?
        dself.Eamp   = depend_array(name="Eamp"  ,value=Eamp   if Eamp   is not None else np.zeros(3))
        dself.Efreq  = depend_value(name="Efreq" ,value=Efreq  if Efreq  is not None else 0.0 )
        dself.Ephase = depend_value(name="Ephase",value=Ephase if Ephase is not None else 0.0 )
        dself.Epeak  = depend_value(name="Epeak" ,value=Epeak  if Epeak  is not None else 0.0)
        dself.Esigma = depend_value(name="Esigma",value=Esigma if Esigma is not None else np.inf)
        dself.BEC    = depend_array(name="BEC"   ,value=BEC    if BEC    is not None else np.zeros(0))

        self.cpol = cpol

    def copy(self):
        return Ensemble(
            eens=self.eens,
            econs=0.0,
            temp=self.temp,
            pext=self.pext,
            stressext=dstrip(self.stressext).copy(),
            bcomponents=self.bcomp,
            bweights=dstrip(self.bweights).copy(),
            hweights=dstrip(self.hweights).copy(),
            time=self.time,
        )

    def bind(
        self,
        beads,
        nm,
        cell,
        bforce,
        fflist,
        output_maker,
        elist=[],
        xlpot=[],
        xlkin=[],
    ):
        self.beads = beads
        self.cell = cell
        self.forces = bforce
        self.nm = nm
        dself = dd(self)
        self.output_maker = output_maker

        # this binds just the explicit bias forces
        self.bias.bind(
            self.beads,
            self.cell,
            self.bcomp,
            fflist,
            open_paths=nm.open_paths,
            output_maker=self.output_maker,
        )

        dself.econs = depend_value(name="econs", func=self.get_econs)
        # dependencies of the conserved quantity
        dself.econs.add_dependency(dd(self.nm).kin)
        dself.econs.add_dependency(dd(self.forces).pot)
        dself.econs.add_dependency(dd(self.bias).pot)
        dself.econs.add_dependency(dd(self.nm).vspring)
        dself.econs.add_dependency(dself.eens)

        # pipes the weights to the list of weight vectors
        i = 0
        for fc in self.bias.mforces:
            if fc.weight != 1:
                warning(
                    "The weight given to forces used in an ensemble bias are given a weight determined by bias_weight"
                )
            dpipe(dself.bweights, dd(fc).weight, i)
            i += 1

        # add Hamiltonian REM bias components
        if len(self.hweights) == 0:
            self.hweights = np.ones(len(self.forces.mforces))

        dself.hweights = depend_array(name="hweights", value=np.asarray(self.hweights))

        # we use ScaledForceComponents to replicate the physical forces without (hopefully) them being actually recomputed
        for ic in range(len(self.forces.mforces)):
            sfc = ScaledForceComponent(self.forces.mforces[ic], 1.0)
            self.bias.add_component(self.forces.mbeads[ic], self.forces.mrpc[ic], sfc)
            dd(sfc).scaling._func = lambda i=ic: self.hweights[i] - 1
            dd(sfc).scaling.add_dependency(dself.hweights)

        self._elist = []

        for e in elist:
            self.add_econs(e)

        dself.lpens = depend_value(
            name="lpens", func=self.get_lpens, dependencies=[dself.temp]
        )
        dself.lpens.add_dependency(dd(self.nm).kin)
        dself.lpens.add_dependency(dd(self.forces).pot)
        dself.lpens.add_dependency(dd(self.bias).pot)
        dself.lpens.add_dependency(dd(self.nm).vspring)

        # extended Lagrangian terms for the ensemble
        self._xlpot = []
        for p in xlpot:
            self.add_xlpot(p)

        self._xlkin = []
        for k in xlkin:
            self.add_xlkin(k)

        # I need cptime to be defined here, and not in TimeDependentIntegrator
        dself.cptime = depend_value(name="cptime",value=0)
        dself.Eenvelope  = depend_value(name="Eenvelope" ,value=0.0,func=self._get_Eenvelope,dependencies=[dself.cptime,dself.Epeak,dself.Esigma])
        dself.Efield = depend_array(name="Efield",value=np.zeros(3, float),func=self._get_Efield,dependencies=[dself.cptime,dself.Eamp,dself.Efreq,dself.Ephase,dself.Eenvelope])
        
    
        # ES: polarization(s) for each beads
        #all_q = [dd(self.beads[i]).q for i in range(self.beads.nbeads)]
        #print(len(self.beads.q._dependants))
        val = np.full(self.beads.nbeads,np.zeros(3,dtype=float)) if self.beads.nbeads > 1 else np.zeros(3,dtype=float)
        dself.IonsPol  = depend_array(name="IonsPol" , func=lambda:self._get_pol(what="ions") ,value=val,dependencies=[dself.time,dd(self.beads).q])
        dself.ElecPol  = depend_array(name="ElecPol" , func=lambda:self._get_pol(what="elec") ,value=val,dependencies=[dself.time,dd(self.beads).q])
        dself.TotalPol = depend_array(name="TotalPol", func=lambda:self._get_pol(what="total"),value=val,dependencies=[dself.time,dd(self.beads).q])
        
        # print(dself.ElecPol)
        # print(dself.ElecPol)
        # dself.beads.q.taint(True)
        # print(dself.ElecPol)

        # ES: Ensemble polarization(s): the average over the bead of the previous quantities
        dself.EnsIonsPol  = depend_array(name="EnsIonsPol" , func=lambda:self._get_enspol(what="ions") ,value=np.zeros(3,dtype=float),dependencies=[dself.IonsPol ,dself.time])
        dself.EnsElecPol  = depend_array(name="EnsElecPol" , func=lambda:self._get_enspol(what="elec") ,value=np.zeros(3,dtype=float),dependencies=[dself.ElecPol ,dself.time])
        dself.EnsTotalPol = depend_array(name="EnsTotalPol", func=lambda:self._get_enspol(what="total"),value=np.zeros(3,dtype=float),dependencies=[dself.TotalPol,dself.time])

        dself.EDAenergy = depend_value(name="EDAenergy", func=self._get_EDAenergy,value=0.0,dependencies=[dd(self.cell).V,dself.EnsTotalPol,dself.Efield])
        #dself.Eenthalpy = depend_value(name="Eenthalpy", func=self._get_Eenthalpy,value=0.0,dependencies=[dself.econs,dself.EDAenergy])

        #dself.Efieldcart = depend_array(name="Efieldcart" , func=lambda:self.cell.lv2cart(self.Efield) ,value=np.zeros(3,dtype=float),dependencies=[dself.Efield])
        #dself.Efieldcart = depend_array(name="Efieldcart" , func=lambda:self.cell.change_basis(v=self.Efield,orig="lv",dest="cart"),value=np.zeros(3,dtype=float),dependencies=[dself.Efield])
        #dself.BEC = depend_array(name="BEC" , func=lambda:self._get_BEC() ,value=np.zeros((3,3),dtype=float),dependencies=[dself.BEC])
        
    def add_econs(self, e):
        self._elist.append(e)
        dd(self).econs.add_dependency(e)

    def add_xlpot(self, p):
        self._xlpot.append(p)
        dd(self).lpens.add_dependency(p)

    def add_xlkin(self, k):
        self._xlkin.append(k)
        dd(self).lpens.add_dependency(k)

    def get_econs(self):
        """Calculates the conserved energy quantity for constant energy
        ensembles.
        """

        eham = self.nm.vspring + self.nm.kin + self.forces.pot
        
        eham += self.bias.pot  # bias

        for e in self._elist: # add thermostat and barostat
            eham += e.get()

        return eham + self.eens

    def get_lpens(self):
        """Returns the ensemble probability (modulo the partition function)
        for the ensemble.
        """

        lpens = self.forces.pot + self.bias.pot + self.nm.kin + self.nm.vspring

        # inlcude terms associated with an extended Lagrangian integrator of some sort
        for p in self._xlpot:
            lpens += p.get()
        for k in self._xlkin:
            lpens += k.get()

        lpens *= -1.0 / (Constants.kb * self.temp * self.beads.nbeads)
        return lpens

    def _get_EDAenergy(self):
        pol = self.cell.change_basis(v=self.EnsTotalPol,orig="rlv",dest="lv") # total polarization in cartesian coordinates
        return float(self.cell.V * np.dot(pol , self.Efield))

    # def _get_Eenthalpy(self):
    #     return self.econs - self.EDAenergy

    def _get_enspol(self,what=None):
        """Return the ensemble average of the polarization(s)"""
        pol = self._get_pol(what)
        if self.beads.nbeads > 1 :
            return np.asarray(pol).mean(axis=0)
        else :
            return pol

    def _get_Eenvelope(self):
        """Gte the gaussian envelope function of the external electric field"""
        # https://en.wikipedia.org/wiki/Normal_distribution
        if self.Epeak > 0.0 and self.Esigma != np.inf :
            x = self.cptime # indipendent variable
            u = self.Epeak  # mean value
            s = self.Esigma # standard deviation
            return np.exp( - 0.5 * ((x-u)/s)**2 ) # the returned maximum value is 1, when x = u
        else :
            return None

    def _get_Efield(self):
        """Get the value of the external electric field (w.r.t. the lattice vectors)"""
        if self.Eenvelope is not None :
            return self.Eamp * np.cos( self.Efreq * self.cptime + self.Ephase) * self.Eenvelope#(self.cptime)
        else :
            return self.Eamp * np.cos( self.Efreq * self.cptime + self.Ephase)

    # def _lv2cart(self,BEC):
    #     """Get the BEC tensors expressed w.r.t. the lattice vectors and returns the same tensor expressed in cartesian coordinates"""
    #     R = np.asarray(self.cell.h)#.copy()
    #     for i in range(3):
    #         R[:,i] = R[:,i] / linalg.norm(R[:,i])
    #     return linalg.inv(R) @ BEC @ R

    def _get_BEC(self):
        """Return the BEC tensors (cart,cart).
        The BEC tensor are stored in a compact form.
        This method trasform the BEC tensors into another data structure, suitable for computation.
        A lambda function is also returned to perform fast matrix multiplication.
        """

        N = len(self.BEC)      # lenght of the BEC array
        Na = self.beads.natoms # number of atoms

        if N == Na:     # scalar BEC
            Z = np.zeros((Na,3,3))
            for i in range(Na):
                for j in range(3):
                    Z[i,j,j] = self.BEC[i]
            return Z #self._lv2cart(Z)
            #lambda a,b : a*b # element-wise (matrix) multplication (only the diagonal elements have been allocated)

        elif N == 3*Na: # diagonal BEC
            Z = np.zeros((Na,3,3))
            temp = self.BEC.reshape((Na,3))
            for i in range(Na):
                for j in range(3):
                    Z[i,j,j] = temp[i,j]
            return Z # self._lv2cart(Z)
            #lambda a,b : a*b # element-wise (matrix) multplication (only the diagonal elements have been allocated)
        
        elif N == 9*Na: # all-components BEC
            Z = np.zeros((Na,3,3))
            temp = self.BEC.reshape((Na,3,3))
            for i in range(Na):
                Z[i,:,:] = temp[i,:,:]
            return Z # self._lv2cart(Z) # rows-by-columns (matrix) multplication (all the elements have been allocated)

        else :
            raise ValueError("BEC tensor with wrong size!")

    def _get_pol(self,what,bead=None):
        """Return the polarization vector of all the beads as a list of np.array"""
        self._check_pol()

        # check that bead is a correct value
        N = self.beads.nbeads
        if bead is not None:
            if bead < 0:
                raise ValueError("Error in get_pol: 'beads' is negative") 
            if bead >= N :
                raise ValueError("Error in get_pol: 'beads' is greater than the number of beads") 
        
        # return the polarization
        if what in ["total","elec","ions"]:
            if not self.cpol:
                return np.asarray([0,0,0]) # the polarization is not computed by the driver, then return [0,0,0]
            else :
                pol = [self.forces.extras["polarization"][i][what] for i in range(N)]
                return pol[0] if bead is None else pol[bead] 
            
        elif what == "all":
            # ES: pay attention, these following have to be in the same order to line 530 in /ipi/engine/outputs.py
            pol = [ np.asarray(list(self.forces.extras["polarization"][i]["ions"])+\
                                list(self.forces.extras["polarization"][i]["elec"])+\
                                list(self.forces.extras["polarization"][i]["total"])) for i in range(N)]
            return pol if bead is None else pol[bead] 
        
        else:
            raise ValueError("Error in get_pol: '"+what+"' is not a 'polarization' key") 
        

    def _check_pol(self):
        """Check that the polarization is correctly formatted."""
        # check whether the driver returned to i-pi the polarization values

        msg = "Error in _check_pol"

        if self.cpol :
            if "polarization" not in self.forces.extras :
                raise warning(msg+": polarization is not returned to i-pi (or at least not accessible in _check_pol)") 
        else : # the polarization is not computed by the driver, then skip this check
            return True

        N = self.beads.nbeads
        # check whether the number of polarization values is correct, i.e. equal to the number of beads 
        # this should be done in ForceComponents.extra_gather (/ipi/engine/forces.py)
        if len(self.forces.extras["polarization"]) != N:
            raise ValueError(msg+": number of polarization values (accessed in _check_pol) should be equal to number of beads")

        # check whether the total, electronic, and ionic polarizations are all available
        for word in ["total","ions","elec"]:
            if not np.all([word in self.forces.extras["polarization"][i] for i in range(N)]):
                raise ValueError(msg+": "+word+" polarization not present for all the beads")

        # check whether the polarizations (total, electronic, and ionic) are identically vanishing
        from numpy import linalg as LA
        for word in ["total","ions","elec"]:
            if np.all([ LA.norm(self.forces.extras["polarization"][i][word]) == 0 for i in range(N)]):
                warning(word+" polarization is vanishing for all the beads",verbosity.high)
        return True

    # I moved this method from TimeDependentIntegrator (no longer available) to Ensemble
    def _check_time(self):
        """Check that self.cptime is equal to self.ensemble.time.
        Pay attention that this is not always true all over the simulation!
        These variable have to be equal only before and after the Integration procedure.
        In fact, this method is called only in Dynamics.step, after self.integrator.step(step).
        The two variable are also forces to be equal before the INtegration procedure at each step.

        This method should always return True, but perhaps future code changes could "break" this.
        Better to be sure that everythin is fine :) """

        thr_time_comparison = 0.1
        if abs(self.cptime - self.time) > thr_time_comparison:
            raise ValueError("Error in EDAIntegrator._check_time: the 'continous' time of EDAIntegrator does not match"+\
                "Ensemble.time (up to a threshold).\nThis seems to be a coding error, not due to wrong input parameters."+\
                "\nRead the description of the function in file ipi/engine/motion/dynamics.py."+\
                "\nAnd then, if you still have problem, you can write me an email to stocco@fhi-berlin.mpg.de.\nBye :)")
        return True

