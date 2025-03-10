from typing import Dict, Tuple
import torch
from e3nn.util.jit import compile_mode
from e3nn.util import jit
from ase.io import read
from ase import Atoms

torch.set_default_dtype(torch.float64)


# ---------------------------------------#
def compute_dipole_jacobian(
    dipole: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:

    if dipole.ndim != 1:
        raise RuntimeError("Dipole must have dimension == 1.")

    if positions.ndim != 2:
        raise RuntimeError("Positions must have dimension == 2.")

    if not positions.requires_grad:
        raise RuntimeError("Positions must have requires_grad=True.")

    # Preallocate tensor for Jacobian
    jacobian = torch.zeros(
        (dipole.shape[0], positions.shape[0], positions.shape[1]),
        dtype=positions.dtype,
        device=positions.device,
    )

    for i in range(dipole.shape[0]):
        gradient = torch.autograd.grad(
            outputs=[dipole[i]],
            inputs=[positions],
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]

        if gradient is not None:
            jacobian[i] = gradient  # Assign directly instead of appending

    return jacobian


# ---------------------------------------#
@compile_mode("script")
class PSwater(torch.nn.Module):

    def __init__(self, device="cpu", *argc, **kwargs):
        self.device = device
        super().__init__(*argc, **kwargs)

    def atoms2data(self, atoms: Atoms) -> Dict[str, torch.Tensor]:
        data = {
            "positions": torch.from_numpy(atoms.get_positions()),
            # add more stuffs if you need
        }
        return data

    def evaluate(self, atoms: Atoms) -> Dict[str, torch.Tensor]:
        data = self.atoms2data(atoms)
        return self(data)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Evaluates the potential energy surface (PES), forces, dipole moment, and Born effective charges for a water monomer.

        Args:
            cell: Simulation cell (not used in this example).
            pos: Positions of the atoms in the water monomer.

        Returns:
            float: potential energy.
            numpy.ndarray: forces on the atoms.
            numpy.ndarray: virial tensor (not used in this example).
            dict: additional properties including dipole moment and Born effective charges.
        """

        r1 = data["positions"].clone().detach()
        r1.requires_grad_(True)  # angstrom
        pot, force = self.calculate_energy_forces(r1)  #
        dipole, Z = self.calculate_dipole_born_charges(r1)

        results = {
            "energy": pot,
            "forces": force,
            "stress": torch.zeros((3, 3)),
            "dipole": dipole,
            "BEC": Z.reshape((3, 9)).T,
        }

        return results

    def calculate_pes(self, r1: torch.Tensor) -> torch.Tensor:
        """Calculates the potential energy surface (PES) for a water monomer.

        Args:
            r1 (torch.Tensor): Positions of the atoms in the water monomer.

        Returns:
            torch.Tensor: Potential energy.
        """
        ROH1 = torch.norm(r1[1] - r1[0] - 0.98)  # **2
        ROH2 = torch.norm(r1[2] - r1[0] - 0.98)  # **2
        return ROH1 + ROH2

        # eO  = 2*r1[0]**2
        # eH1 = r1[1]**2
        # eH2 = r1[2]**2
        # return torch.sum(eO - eH1 - eH2)

    def calculate_charges(self, r1: torch.Tensor) -> torch.Tensor:
        """Calculates the partial charges for a water monomer.

        Args:
            r1 (torch.Tensor): Positions of the atoms in the water monomer.

        Returns:
            torch.Tensor: Partial charges on the atoms.
        """
        return torch.stack([torch.tensor(-2.0), torch.tensor(1.0), torch.tensor(1.0)])

    def calculate_energy_forces(
        self, r1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the potential energy surface (PES) and forces for a water monomer.

        Args:
            r1 (torch.Tensor): Positions of the atoms in the water monomer (in angstrom).

        Returns:
            torch.Tensor: potential energy.
            torch.Tensor: forces on the atoms.
        """
        # with torch.autograd.set_detect_anomaly(False):
        r1.requires_grad_(True)
        pes = self.calculate_pes(r1)
        pes.backward()
        forces = -r1.grad
        return pes, forces

    def calculate_dipole_born_charges(
        self, r1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the dipole moment and Born effective charges for a water monomer.

        Args:
            r1 (torch.Tensor): Positions of the atoms in the water monomer.

        Returns:
            torch.Tensor: Dipole moment.
            torch.Tensor: Born effective charges.
        """
        r1.requires_grad_(True)
        charges = self.calculate_charges(r1)
        dipole = torch.sum(charges * r1, dim=0)
        assert dipole.shape == (3,)
        bec = compute_dipole_jacobian(dipole, r1)
        # test = bec.sum(dim=1)
        # if torch.mean(test) > 1e-6:
        #     warning(
        #         "Born effective charges does not satify acoustic sum rule: ",
        #         torch.mean(test),
        #     )
        return dipole, bec


# ---------------------------------------#
# save the models
model = PSwater()
model.eval()
torch.save(model, "pswater.model")
print("print eager model to 'pswater.model'")

model_compiled = jit.compile(model)
torch.jit.save(model_compiled, "pswater_compiled.model")
print("print compiled model to 'pswater_compiled.model'")

# ---------------------------------------#
# test the model in eager mode
print("testing eager model")
model = torch.load("pswater.model")
atoms = read("start.xyz")
results = model.evaluate(atoms)

batch = {
    "positions": torch.from_numpy(atoms.get_positions()),
    # add more stuffs if you need
}
results = model(batch)

# ---------------------------------------#
# test the model in compiled mode
print("testing compiled model")
model_compiled = torch.jit.load("pswater_compiled.model")
results = model.evaluate(atoms)
batch = {
    "positions": torch.from_numpy(atoms.get_positions()),
    # add more stuffs if you need
}
results = model(batch)

# ---------------------------------------#
print("everything was okay. Bye :)")
