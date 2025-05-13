import copy
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from ase import Atom, Atoms
from ase.io import cube

from dfttoolkit.geometry import Geometry
from dfttoolkit.utils.periodic_table import PeriodicTable
from dfttoolkit.utils.units import BOHR_IN_ANGSTROM

from .base import Parser
from .utils.math_utils import get_triple_product


class Cube(Parser):
    """
    Read, interpolate, and perform operations on cube files.

    ...

    Attributes
    ----------
    path: str
        path to the cube file
    lines: List[str]
        contents of the cube file
    """

    def __init__(self, **kwargs: str):
        # Parse file information and perform checks
        super().__init__(self._supported_files, **kwargs)

        # Check that the file is a cube file and in the correct format
        self._check_binary(False)

        # Parse the cube data here rather than in base.File.__post_init__ so we can call
        # ASE's cube.read_cube()
        with open(self.path) as f:
            _cf = cube.read_cube(f)
            self.lines = f.readlines()
            self.data = b""
            self._binary = False

        self._atoms = _cf["atoms"]
        self._n_atoms = len(self._atoms)
        self._volume = _cf["volume"]
        self._origin = _cf["origin"]

        # Centre the atoms to cube origin
        self._atoms.translate(-self._origin)

        # Get other cube file parameters
        self._grid_vectors = np.array(
            [float(j) for i in self.lines[2:5] for j in i.split()[1:]]
        )

        self._shape = np.array(
            [int(i.split()[0]) for i in self.lines[3:5]], dtype=np.int64
        )

        self._dV = np.abs(
            get_triple_product(
                self.grid_vectors[0, :],
                self.grid_vectors[1, :],
                self.grid_vectors[2, :],
            )
        )

        # Get atoms
        atom_Z = np.zeros(self._n_atoms, dtype=int)  # noqa: N806
        atom_pos = np.zeros((self._n_atoms, 3))
        for i in range(self._n_atoms):
            spl_atom_line = self.lines[6 + i].split()
            atom_Z[i] = int(spl_atom_line[0])
            atom_pos[i, :] = np.array(spl_atom_line[2:5])

        self._geom = Geometry()
        atom_pos *= BOHR_IN_ANGSTROM
        species = [PeriodicTable.element_symbols(i) for i in atom_Z]
        self._geom.add_atoms(atom_pos, species)

        # Parse the grid data
        self._grid = np.fromiter(
            (float(x) for line in self.lines[7:] for x in line.split()),
            dtype=np.float64,
        )
        self._n_points = len(self._grid)
        self._grid = np.reshape(self._grid, self._shape)

    @property
    def supported_files(self) -> dict[str, str]:
        return {"cube": ".cube"}

    def __init_subclass__(cls, **kwargs: str):
        # Override the parent's __init_subclass__ without calling it
        pass

    @property
    def atoms(self) -> Atom | Atoms:
        """Atoms present in the cube file."""
        return self._atoms

    @property
    def comment(self) -> str:
        """Comment line of the cube file."""
        return " ".join(self.lines[0:1])

    @property
    def dV(self) -> npt.NDArray:  # noqa: N802
        """Volume of the cube file."""
        return self._dV

    @property
    def geometry(self) -> Geometry:
        """Atoms represented in the cube file."""
        return self._geom

    @property
    def grid(self) -> npt.NDArray:
        """Grid data of the cube file."""
        return self._grid

    @property
    def grid_vectors(self) -> npt.NDArray:
        """Grid vectors of the cube file."""
        return self._grid_vectors

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the cube file."""
        return self._n_atoms

    @property
    def n_points(self) -> int:
        """Number of points in the grid data."""
        return self._n_points

    @property
    def origin(self) -> npt.NDArray[np.float64]:
        """Origin of the cube file."""
        return self._origin

    @property
    def shape(self) -> npt.NDArray[np.int64]:
        """Number of dimensions of the grid vectors."""
        return self._shape

    @property
    def volume(self) -> npt.NDArray[np.float64]:
        """Volume of the cube file."""
        return self._volume

    def __add__(self, other: Self):
        new_cube = copy.deepcopy(self)
        new_cube.data += other.data

        return new_cube

    def get_integrated_projection_on_axis(
        self, axis: Literal[0, 1, 2]
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Integrate cube file over the plane perpendicular to the selected axis.

        Returns
        -------
        proj : npt.NDArray
            projected values
        xaxis : npt.
            coordinates of axis (same length as proj)
        """
        axsum = list(range(3))
        axsum.pop(axis)

        dA = np.linalg.norm(  # noqa: N806
            np.cross(self.grid_vectors[axsum[0], :], self.grid_vectors[axsum[1], :])
        )

        # trapeziodal rule: int(f) = sum_i (f_i + f_i+1) * dA / 2
        # but here not div by 2 because no double counting in sum
        proj = np.sum(self.data, axis=tuple(axsum)) * dA
        xstart = self.origin[axis]
        xend = self.origin[axis] + self.grid_vectors[axis, axis] * self.shape[axis]
        xaxis = np.linspace(xstart, xend, self.shape[axis])

        return proj, xaxis

    def get_averaged_projection_on_axis(
        self, axis: Literal[0, 1, 2], divide_by_area: bool = True
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Project cube values onto an axis and normalise by the perpendicular area.

        Returns
        -------
        proj : npt.NDArray
            projected values
        xaxis : npt.NDArray
            coordinates of axis (same length as proj)
        """
        axsum = list(range(3))
        axsum.pop(axis)

        dA = np.linalg.norm(  # noqa: N806
            np.cross(self.grid_vectors[axsum[0], :], self.grid_vectors[axsum[1], :])
        )
        n_datapoints = self.shape[axsum[0]] * self.shape[axsum[1]]
        A = dA * n_datapoints  # noqa: N806

        # this gives sum(data) * dA
        proj, xaxis = self.get_integrated_projection_on_axis(
            axis
        )  # sum_i (f_i +f_i+1) * dA / 2

        # remove dA from integration
        proj = proj / dA

        # average per area or pure mathematical average
        averaged = proj / A if divide_by_area else proj / n_datapoints

        return averaged, xaxis
