from typing import Union

import numpy as np
import numpy.typing as npt
from ase import Atom, Atoms
from ase.io import cube

from dfttoolkit.base import Parser


class Cube(Parser):
    """
    Perform operations on cube files.
    """

    def __init__(self, **kwargs):
        # Parse file information and perform checks
        super().__init__(self._supported_files.keys(), **kwargs)

        # check that the file is a cube file and in the correct format
        self._check_output_file_extension("cube")
        self._check_binary(False)

        self._volume, self._atoms = cube.read_cube_data(self.path)

    @property
    def _supported_files(self) -> dict:
        return {"cube": ".cube"}

    def __init_subclass__(cls, **kwargs):
        # Revert back to the original __init_subclass__ method to avoid checking for
        # required methods in child class of this class too
        return super(Parser, cls).__init_subclass__(**kwargs)

    @property
    def atoms(self) -> Union[Atom, Atoms]:
        return self._atoms

    @property
    def volume(self) -> npt.NDArray[np.float64]:
        return self._volume  # pyright: ignore
