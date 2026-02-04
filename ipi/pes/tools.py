import json
import numpy as np
from ipi.utils.units import unit_to_internal, unit_to_user


class Instructions:

    dimensions = {}
    units = {"length": "atomic_unit", "energy": "atomic_unit"}

    def __init__(self, instructions: dict, *argc, **argv):

        super().__init__(*argc, **argv)

        # Read instructions from file or use provided dictionary
        if isinstance(instructions, str):
            with open(instructions, "r") as f:
                instructions = json.load(f)
        elif isinstance(instructions, dict):
            pass
        else:
            raise ValueError("`instructions` can be `str` or `dict` only.")

        # convert parameters to the required units
        to_delete = list()
        for k, dimension in self.dimensions.items():
            variable = f"{k}_unit"
            if variable in instructions:
                to_delete.append(variable)
                if instructions[variable] is not None:
                    factor = convert(
                        1,
                        dimension,
                        _from=instructions[variable],
                        _to=self.units[dimension],
                    )
                    instructions[k] = process_input(instructions[k]) * factor
        for k in to_delete:
            del instructions[k]

        self.instructions = instructions


# ---------------------- #
def convert(
    what: float,
    family: str = None,
    _from: str = "atomic_unit",
    _to: str = "atomic_unit",
) -> float:
    """
    Converts a physical quantity between units of the same type (length, energy, etc.)
    Example:
    value = convert(7.6,'length','angstrom','atomic_unit')
    arr = convert([1,3,4],'energy','atomic_unit','millielectronvolt')
    """
    # from ipi.utils.units import unit_to_internal, unit_to_user
    if family is not None:
        factor = unit_to_internal(family, _from, 1)
        factor *= unit_to_user(family, _to, 1)
        return what * factor
    else:
        return what


# ---------------------- #
def process_input(value):
    """
    Standardizes user input into numerical format (float or np.array).
    """
    if isinstance(value, float):
        return value
    elif isinstance(value, int):
        return value
    elif isinstance(value, list):
        return np.array(value)
    else:
        raise TypeError("Input must be a float or a list.")
