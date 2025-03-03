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

        # Parse the cube data here rather than in base.File.__post_init__ so we can call
        # ASE's cube.read_cube()
        with open(self.path, "r") as f:
            self._cf = cube.read_cube(f)
            self.lines = f.readlines()
            self.data = b""
            self._binary = False

        self._atoms = self._cf["atoms"]
        self._volume = self._cf["volume"]
        self._origin = self._cf["origin"]

        # Centre the atoms to cube origin
        self._atoms.translate(-self._origin)  # pyright: ignore

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

    @property
    def origin(self) -> npt.NDArray[np.float64]:
        return self._origin  # pyright: ignore
