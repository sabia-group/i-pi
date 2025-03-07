from typing import Dict, Tuple
import torch
from e3nn.util.jit import compile_mode
from e3nn.util import jit
from ase.io import read
from ase import Atoms
from pswater_constants import ConstantData

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
class PSwater(ConstantData, torch.nn.Module):

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
            float: Potential energy.
            numpy.ndarray: Forces on the atoms.
            numpy.ndarray: Virial tensor (not used in this example).
            dict: Additional properties including dipole moment and Born effective charges.
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
        ROH1 = r1[1] - r1[0]
        ROH2 = r1[2] - r1[0]
        RHH = r1[1] - r1[2]
        dROH1 = torch.norm(ROH1)
        dROH2 = torch.norm(ROH2)
        dRHH = torch.norm(RHH)
        costh = torch.dot(ROH1, ROH2) / (dROH1 * dROH2)

        exp1 = torch.exp(-self.alphaoh * (dROH1 - self.roh))
        exp2 = torch.exp(-self.alphaoh * (dROH2 - self.roh))
        Va = self.deoh * (exp1 * (exp1 - 2.0) + exp2 * (exp2 - 2.0))
        Vb = self.phh1 * torch.exp(-self.phh2 * dRHH)

        x1 = (dROH1 - self.reoh) / self.reoh
        x2 = (dROH2 - self.reoh) / self.reoh
        x3 = costh - self.costhe

        # Initialize a list to store tensor values for each step
        fmat_list = [torch.zeros(3)]  # fmat(0,1:3) = 0.d0
        fmat_list.append(torch.ones(3))  # fmat(1,1:3) = 1.d0

        # Compute fmat recursively without in-place operations
        for j in range(2, 16):
            fmat_list.append(
                torch.stack(
                    [
                        fmat_list[j - 1][0] * x1,
                        fmat_list[j - 1][1] * x2,
                        fmat_list[j - 1][2] * x3,
                    ]
                )
            )

        # Convert list to a single tensor
        fmat = torch.stack(fmat_list)

        efac = torch.exp(
            -self.b1 * ((dROH1 - self.reoh) ** 2 + (dROH2 - self.reoh) ** 2)
        )

        sum0 = torch.tensor(0.0)

        for j in range(1, 244):  # Fortran (2:245) → Python (1, 244)
            inI = self.idx[j][0]
            inJ = self.idx[j][1]
            inK = self.idx[j][2]

            sum0 += (
                self.c5z[j]
                * (fmat[inI, 0] * fmat[inJ, 1] + fmat[inJ, 0] * fmat[inI, 1])
                * fmat[inK, 2]
            )

        Vc = 2.0 * self.c5z[0] + efac * sum0
        e1 = Va + Vb + Vc
        e1 += 0.44739574026257
        e1 *= 0.00285914375100642899  # cm-1 to Kcal/mol

        # return unit_to_internal("energy", "kilocal/mol", e1)
        return e1

    def calculate_charges(self, r1: torch.Tensor) -> torch.Tensor:
        """Calculates the partial charges for a water monomer.

        Args:
            r1 (torch.Tensor): Positions of the atoms in the water monomer.

        Returns:
            torch.Tensor: Partial charges on the atoms.
        """
        ROH1 = r1[1] - r1[0]
        ROH2 = r1[2] - r1[0]
        # RHH = r1[1] - r1[2]
        dROH1 = torch.norm(ROH1)
        dROH2 = torch.norm(ROH2)
        # dRHH = torch.norm(RHH)
        costh = torch.dot(ROH1, ROH2) / (dROH1 * dROH2)
        efac = torch.exp(
            -self.b1 * ((dROH1 - self.reoh) ** 2 + (dROH2 - self.reoh) ** 2)
        )

        x1 = (dROH1 - self.reoh) / self.reoh
        x2 = (dROH2 - self.reoh) / self.reoh
        x3 = costh - self.costhe

        fmat_list = [
            torch.zeros(3, dtype=r1.dtype, device=r1.device) for _ in range(16)
        ]
        fmat_list[1] = torch.ones(3, dtype=r1.dtype, device=r1.device)
        for j in range(2, 16):
            fmat_list[j] = torch.stack(
                [
                    fmat_list[j - 1][0] * x1,
                    fmat_list[j - 1][1] * x2,
                    fmat_list[j - 1][2] * x3,
                ]
            )
        fmat = torch.stack(fmat_list)

        P1 = torch.tensor(0.0)
        P2 = torch.tensor(0.0)
        for j in range(2, 84):
            inI = self.idx[j][0]
            inJ = self.idx[j][1]
            inK = self.idx[j][2]
            P1 = P1 + self.coefD[j] * fmat[inI, 0] * fmat[inJ, 1] * fmat[inK, 2]
            P2 = P2 + self.coefD[j] * fmat[inJ, 0] * fmat[inI, 1] * fmat[inK, 2]

        PC0 = (
            0.2999
            * ((dROH1**-0.6932) + (dROH2**-0.6932))
            * (1.0099 + costh * -0.1801 + 0.5 * (3 * costh**2 - 1) * 0.0892)
        )
        P1 = self.coefD[0] + P1 * efac + PC0 * 0.529177249
        P2 = self.coefD[0] + P2 * efac + PC0 * 0.529177249

        q3 = torch.stack([-(P1 + P2), P1, P2])  # ,requires_grad=r1.requires_grad)

        return q3

    def calculate_energy_forces(
        self, r1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the potential energy surface (PES) and forces for a water monomer.

        Args:
            r1 (torch.Tensor): Positions of the atoms in the water monomer (in angstrom).

        Returns:
            torch.Tensor: Potential energy.
            torch.Tensor: Forces on the atoms.
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
