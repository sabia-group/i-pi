from ipi.utils.messages import verbosity,warning
import numpy as np

__all__ = ["get_pol","_check_pol"]

def get_pol(self,what,bead=None):
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
        pol = [self.forces.extras["polarization"][i][what] for i in range(N)]
        return pol if bead is None else pol[bead] 
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

    if "polarization" not in self.forces.extras:
        raise ValueError(msg+": polarization is not returned to i-pi (or at least not accessible in _check_pol)")

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