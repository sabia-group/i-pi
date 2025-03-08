"""An easy interface to any torch.nn.Module suing ase.Calculator"""

from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
from .ase import ASEDriver
from typing import Dict

__DRIVER_NAME__ = "torch"
__DRIVER_CLASS__ = "TorchDriver"


class TorchDriver(ASEDriver):
    """
    i-pi-py_driver -m torch -o template=start.xyz,model_path=pswater_compiled.model,jit=true -u -a host
    """

    def __init__(self, template, *args, **kwargs):
        self.kwargs = kwargs
        super().__init__(template, *args, **kwargs)

    def check_parameters(self):
        """Check the arguments requuired to run the driver

        This loads the potential and atoms template in MACE
        """

        super().check_parameters()

        self.ase_calculator = EasyTorchCalculator(**self.kwargs)


try:
    import torch
except:
    pass


class EasyTorchCalculator(Calculator):

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        jit: bool = False,
        *argc,
        **kwargs
    ):
        self.device = device
        self.dtype = dtype
        if jit:
            self.model: torch.nn.Module = torch.jit.load(model_path)
        else:
            self.model: torch.nn.Module = torch.load(model_path)
        self.model.to(device=self.device, dtype=self.dtype)
        super().__init__(*argc, **kwargs)

    def calculate(
        self, atoms: Atoms = None, properties=None, system_changes=all_changes
    ) -> None:
        data = {
            "positions": torch.from_numpy(atoms.get_positions()).to(
                device=self.device, dtype=self.dtype
            ),
            # add more stuffs if you need
        }
        self.results: Dict[str, torch.Tensor] = self.model(data)
        for k, value in self.results.items():
            np_value = value.detach().numpy()  # convert torch.float64 to numpy.float64
            self.results[k] = np_value
