from os.path import join

import numpy as np
import numpy.typing as npt

from .geometry import AimsGeometry


class FrictionTensor:
    """
    TODO.

    ...

    Attributes
    ----------
    friction_tensor
    """

    def __init__(self, directory: str):
        self.geometry = AimsGeometry(join(directory, "geometry.in"))  # noqa: PTH118
        self.friction_tensor_raw = self.read_friction_tensor(
            join(directory, "friction_tensor.out")  # noqa: PTH118
        )

    @property
    def friction_tensor(self) -> npt.NDArray:
        """TODO."""
        return self.friction_tensor_raw

    @friction_tensor.setter
    def friction_tensor(self, friction_tensor_raw: npt.NDArray) -> None:
        self.friction_tensor_raw = friction_tensor_raw

    def read_friction_tensor(self, filename: str) -> npt.NDArray:
        """
        Read the friction tensor when given a calculation directory.

        Saves a full size friction tensor (elements for all atoms) where atom-pairs
        without friction are assigned a friction value of 0.

        Parameters
        ----------
        filename : str
            Path to directry.

        Returns
        -------
        friction_tensor : NDArray
            Friction tensor for all atoms.
        """
        atom_indices = []
        friction_tensor_0 = []

        with open(filename) as f:
            lines = f.readlines()

        for line in lines:
            if "# n_atom" in line:
                spl = line.split()

                line_1 = [i for i in spl if i != ""]

                atom_index = 3 * (int(line_1[2]) - 1) + int(line_1[4]) - 1
                atom_indices.append(atom_index)

            elif "#" not in line:
                friction_tensor_line = [float(i) for i in line.split(" ") if i != ""]

                friction_tensor_0.append(friction_tensor_line)

        friction_tensor_0 = np.array(friction_tensor_0)

        n = len(self.geometry)
        friction_tensor = np.zeros((3 * n, 3 * n))

        for ind_0, atom_index_0 in enumerate(atom_indices):
            for ind_1, atom_index_1 in enumerate(atom_indices):
                friction_tensor[atom_index_0, atom_index_1] = friction_tensor_0[
                    ind_0, ind_1
                ]

        return friction_tensor

    def get_life_time(self, vibration: npt.NDArray) -> npt.NDArray[np.float64]:
        """
        Return life time in ps.

        Parameters
        ----------
        vibration : npt.NDArray
            Vibration vector

        Returns
        -------
        NDArray[float64]
            Life time
        """
        vibration /= np.linalg.norm(vibration)
        force = self.friction_tensor_raw.dot(vibration)
        eta = vibration.dot(force)

        return 1 / eta
