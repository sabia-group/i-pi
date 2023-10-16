""" Small functions/classes providing access to driver PES to be called from driver.py """

from .dummy import Dummy_driver
from .harmonic import Harmonic_driver
from .rascal import Rascal_driver
from .e3nn_pol import e3nn_pol
from .bash_script import bash_script

# __all__ = ["__drivers__", "Dummy_driver", "Harmonic_driver", "Rascal_driver","e3nn_pol","bash_script"]

# dictionary linking strings
__drivers__ = {
    "dummy": Dummy_driver,
    "harmonic": Harmonic_driver,
    "rascal": Rascal_driver,
    "e3nn_pol": e3nn_pol,
    "bash_script":bash_script,
}
