import numpy as np
from typing import Union, List
from ipi.utils.messages import warning
from ipi.pes.dummy import Dummy_driver
from ipi.pes.tools import Instructions

__DRIVER_NAME__ = "confining"
__DRIVER_CLASS__ = "ConfiningPotential"


# ---------------------- #
class ConfiningPotential(Instructions,Dummy_driver):
    """
    Spherical Lennard-Jones potential driver.
    Driver implementing a Spherical Lennard-Jones (LJ) potential.
    Parameters must be passed via a dictionary or a JSON file with the following keys:
    {
        "sigma": 2.569,                 // sigma parameter of the LJ potential
        "epsilon": 2.754,               // epsilon parameter of the LJ potential

        "sigma_unit": "angstrom",       // unit of sigma      (supported: angstrom, atomic_unit[bohr] + derived quantities)
        "epsilon_unit": "kilocal/mol",  // unit of epsilon    (supported: electronvolt, atomic_unit[Hartree], j/mol, cal/mol, kelvin + derived quantities)

        "symbols": ["O"],               // list of chemical symbols to whom the potential applies
        "first_power": 9,               // first power of the LJ potential
        "second_power": 3               // second power of the LJ potential
    }
    Attention: no default parameters are provided, so you must specify all of them.
    You can just copy and paste the above example and modify the values as needed.
    """

    def __init__(
        self,
        template: str,  # template file to extract atomic symbols if symbols not provided
        mode: str,  # mode of the confining potential, e.g. "LJ-wall"
        instructions: dict,  # instructions for the potential
        # species: list = None,  # list of species to which the potential applies
        symbols: list = None,  # list of atomic symbols
        has_energy=True,
        has_forces=True,
        has_stress=False,
        *args,
        **kwargs,
    ):
        assert has_energy, "Radialdriver requires energy calculation."
        assert has_forces, "Radialdriver requires forces calculation."
        assert not has_stress, "Radialdriver does not support stress calculation."

        super().__init__(instructions=instructions, *args, **kwargs)
        self.species = self.instructions["species"] if "species" in self.instructions else None
        del self.instructions

        mode = str(mode).lower()
        if mode == "lj-wall":
            self.potential = LJWall(instructions)
        elif mode == "morse-wall":
            self.potential = MorseWall(instructions)
        else:
            raise ValueError(f"Unknown mode {mode}.")

        # Initialize symbols
        self.atoms = None
        if symbols is None:
            # Let's try to use ASE ...
            try:
                from ase.io import read

                atoms = read(template)
                self.symbols = atoms.get_chemical_symbols()
                assert atoms.positions.shape == (
                    len(self.symbols),
                    3,
                ), "Positions shape mismatch."
            except ImportError:
                warning("Could not find or import the ASE module")
        else:
            # ... but the user can also provide symbols directly
            self.symbols = symbols            
        self.species = self.species if self.species is not None else list(set(self.symbols))

    def compute_structure(self, cell: np.ndarray, pos: np.ndarray):
        """
        Core method that calculates energy and forces for given atoms using
        the spherical Lennard-Jones potential.
        """
        
        assert cell.shape == (
            3,
            3,
        ), "Cell must be a list of 3x3 matrices."
        assert pos.ndim == 2 and pos.shape == (
            len(self.symbols),
            3,
        ), "Position must be a list of Nx3 matrices."

        potential, forces, vir, extras = super().compute_structure(cell, pos)

        # atoms to be considered
        indices = [n for n, symbol in enumerate(self.symbols) if symbol in self.species]

        potential, some_forces = self.potential.compute_energy_and_forces(pos[indices])

        forces[indices, :] = some_forces

        return potential, forces, vir, extras


# ---------------------- #
class LennardJonesPotential(Instructions):
    """
    Spherical Lennard-Jones potential driver.
    Driver implementing a Spherical Lennard-Jones (LJ) potential.
    Parameters must be passed via a dictionary or a JSON file with the following keys:
    {
        "sigma": 2.569,                 // sigma parameter of the LJ potential
        "epsilon": 2.754,               // epsilon parameter of the LJ potential

        "sigma_unit": "angstrom",       // unit of sigma      (supported: angstrom, atomic_unit[bohr] + derived quantities)
        "epsilon_unit": "kilocal/mol",  // unit of epsilon    (supported: electronvolt, atomic_unit[Hartree], j/mol, cal/mol, kelvin + derived quantities)

        "symbols": ["O"],               // list of chemical symbols to whom the potential applies
        "first_power": 9,               // first power of the LJ potential
        "second_power": 3               // second power of the LJ potential
    }
    Attention: no default parameters are provided, so you must specify all of them.
    You can just copy and paste the above example and modify the values as needed.
    """

    dimensions = {
        "sigma": "length",
        "epsilon": "energy",
    }

    def lennard_jones(self, r: np.ndarray):
        """
        Computes potential and analytical forces using NumPy.

        Returns:
            potential (float), forces (ndarray)
        """

        assert r.ndim == 1, "Input positions must be a 1D array of distances."

        sigma = float(self.instructions["sigma"])
        epsilon = float(self.instructions["epsilon"])
        first_power = int(self.instructions["first_power"])
        second_power = int(self.instructions["second_power"])

        # Check if any atom is outside the spherical potential
        if np.any(r <= 0):
            raise ValueError("Some atoms are wrongly located.")

        # Calculate the potential
        sr = sigma / r
        potential = epsilon * (sr**first_power - sr**second_power)
        potential = np.sum(potential)

        # Correct derivative of the potential including the 2/15 factor
        forces = epsilon * (
            -first_power * (sigma**first_power) / (r ** (first_power + 1))
            + second_power * (sigma**second_power) / (r ** (second_power + 1))
        )

        assert forces.shape == r.shape, "Forces shape mismatch."

        return potential, forces


# ---------------------- #
class LJWall(LennardJonesPotential):
    """
    Driver implementing a Lennard-Jones wall potential.
    Parameters must be passed via a dictionary or a JSON file with the following keys:
    {
        "z_plane": 10,                  // z-coordinate of the plane
        "z_plane_unit": "angstrom",     // unit of the z-coordinate (supported: angstrom, atomic_unit[bohr] + derived quantities)
    }
    Attention: no default parameters are provided, so you must specify all of them.
    You can just copy and paste the above example and modify the values as needed.
    """

    dimensions = {
        **LennardJonesPotential.dimensions,
        **{
            "z_plane": "length",
        },
    }

    def compute_energy_and_forces(self, pos: np.ndarray) -> tuple:
        """
        pos : np.ndarray
            Atomic positions (in atomic units).
        """
        z = pos[:, 2]
        dz = np.abs(z - self.instructions["z_plane"])  # Distance from the plane
        potential, forces_z = self.lennard_jones(dz)
        forces = np.zeros_like(pos)
        forces[:, 2] = np.sign(z - self.instructions["z_plane"]) * forces_z
        return potential, forces


# ---------------------- #
class MorsePotential(Instructions):

    dimensions = {
        "D0": "energy",
        "a": "length",
        "z0": "length",
    }

    def morse_potential(self, r: np.ndarray):
        """
        Computes potential and analytical forces using NumPy.

        Returns:
            potential (float), forces (ndarray)
        """

        assert r.ndim == 1, "Input positions must be a 1D array of distances."

        D0 = float(self.instructions["D0"])
        a = 1.0/float(self.instructions["a-1"])
        z0 = float(self.instructions["z0"])

        if np.any(r <= 0):
            raise ValueError("Some atoms are wrongly located.")

        # Exponential term
        exp_term = np.exp(-a * (r - z0))

        # Potential
        potential = D0 * (1.0 - exp_term) ** 2
        potential = np.sum(potential)

        # Analytical force: F = -dV/dr
        forces = -2.0 * D0 * a * (1.0 - exp_term) * exp_term

        assert forces.shape == r.shape, "Forces shape mismatch."

        return potential, forces


# ---------------------- #
class MorseWall(MorsePotential):
    """
    Driver implementing a Lennard-Jones wall potential.
    Parameters must be passed via a dictionary or a JSON file with the following keys:
    {
        "z_plane": 10,                  // z-coordinate of the plane
        "z_plane_unit": "angstrom",     // unit of the z-coordinate (supported: angstrom, atomic_unit[bohr] + derived quantities)
    }
    Attention: no default parameters are provided, so you must specify all of them.
    You can just copy and paste the above example and modify the values as needed.
    """

    dimensions = {
        **MorsePotential.dimensions,
        **{
            "z_plane": "length",
        },
    }

    def compute_energy_and_forces(self, pos: np.ndarray) -> tuple:
        """
        pos : np.ndarray
            Atomic positions (in atomic units).
        """
        z = pos[:, 2]
        dz = np.abs(z - self.instructions["z_plane"])  # Distance from the plane
        potential, forces_z = self.morse_potential(dz)
        forces = np.zeros_like(pos)
        forces[:, 2] = np.sign(z - self.instructions["z_plane"]) * forces_z
        return potential, forces
