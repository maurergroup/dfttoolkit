import copy
from pathlib import Path
from typing import Any, Literal, Self

import ase.io.cube
import numpy as np
import numpy.typing as npt
from ase import Atoms
from scipy.ndimage.interpolation import shift

from .base import Parser
from .geometry import Geometry
from .utils.math_utils import get_triple_product
from .utils.periodic_table import PeriodicTable
from .utils.units import BOHR_IN_ANGSTROM, EPSILON0_AIMS


class Cube(Parser):
    """
    Read, interpolate, and perform operations on cube files.

    NOTE: this class is currently untested - use with caution!

    ...

    Attributes
    ----------
    atoms
    comment
    cube_vectors
    dV
    dv1
    dv2
    dv3
    geometry
    grid
    grid_vectors
    n_atoms
    n_points
    shape
    volume
    path: str
        path to the cube file
    lines: List[str]
        contents of the cube file
    """

    def __init__(self, cube: str):
        # Parse file information and perform checks
        super().__init__(self._supported_files, cube=cube)

        # Check that the file is a cube file and in the correct format
        self._check_binary(False)

        # Parse the cube data here rather than in base.File.__post_init__ so we can call
        # ASE's read_cube()
        with self._path.open() as f:
            _cf = ase.io.cube.read_cube(f)
            self.lines = f.readlines()
            self.data = b""
            self._binary = False

        self._atoms = _cf["atoms"]
        self._n_atoms = len(self._atoms)
        self._origin = _cf["origin"]
        self._volume = _cf["datas"]

        # Centre the atoms to cube origin
        self._atoms.translate(-self._origin)  # pyright: ignore[reportOperatorIssue]

        # Get other cube file parameters
        self._grid_vectors = np.array(
            [float(j) for i in self.lines[2:5] for j in i.split()[1:]]
        )

        self._shape = np.array(
            [int(i.split()[0]) for i in self.lines[3:5]], dtype=np.int64
        )

        self._calculate_cube_vectors()

        # Get atoms
        atom_Z = np.zeros(self._n_atoms, dtype=int)
        atom_pos = np.zeros((self._n_atoms, 3))
        for i in range(self._n_atoms):
            spl_atom_line = self.lines[6 + i].split()
            atom_Z[i] = int(spl_atom_line[0])
            atom_pos[i, :] = np.array(spl_atom_line[2:5])

        self._geom = Geometry()
        atom_pos *= BOHR_IN_ANGSTROM
        species = [PeriodicTable.get_symbol(i) for i in atom_Z]
        self._geom.add_atoms(atom_pos, species)

        # Parse the grid data
        self._grid = np.fromiter(
            (float(x) for line in self.lines[7:] for x in line.split()),
            dtype=np.float64,
        )
        self._n_points = len(self._grid)
        self._grid = np.reshape(self._grid, self._shape)

    @property
    def _supported_files(self) -> dict[str, str]:
        return {"cube": ".cube"}

    def __init_subclass__(cls, **kwargs: str):
        # Override the parent's __init_subclass__ without calling it
        pass

    @property
    def atoms(self) -> Atoms:
        """Atoms present in the cube file."""
        return self._atoms

    @property
    def comment(self) -> str:
        """Comment line of the cube file."""
        return " ".join(self.lines[0:1])

    @property
    def cube_vectors(self) -> npt.NDArray[np.int64]:
        """Cube vectors of the cube file."""
        return self._cube_vectors

    @property
    def dV(self) -> npt.NDArray:  # noqa: N802
        """Volume of the cube file."""
        return self._dV

    @property
    def dv1(self) -> np.floating[Any]:
        """First voxel dimension of the cube file."""
        return self._dv1

    @property
    def dv2(self) -> np.floating[Any]:
        """Second voxel dimension of the cube file."""
        return self._dv2

    @property
    def dv3(self) -> np.floating[Any]:
        """Third voxel dimension of the cube file."""
        return self._dv3

    @property
    def geometry(self) -> Geometry:
        """Atoms represented in the cube file."""
        return self._geom

    @geometry.setter
    def geometry(self, geometry: Geometry) -> None:
        self._geom = geometry
        self._atoms = geometry.get_as_ase()
        self._n_atoms = len(geometry)

    @property
    def grid(self) -> npt.NDArray:
        """Grid data of the cube file."""
        return self._grid

    @grid.setter
    def grid(self, grid: npt.NDArray) -> None:
        self._grid = grid

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
        return self._origin  # pyright: ignore

    @property
    def shape(self) -> npt.NDArray[np.int64]:
        """Number of dimensions of the grid vectors."""
        return self._shape

    @property
    def volume(self) -> npt.NDArray[np.float64]:
        """Volume of the cube file."""
        return self._volume  # pyright: ignore

    def __add__(self, other: Self):
        new_cube = copy.deepcopy(self)
        new_cube.grid += other.grid

        return new_cube

    def __sub__(self, other: Self):
        new_cube = copy.deepcopy(self)
        new_cube.grid -= other.grid

        return new_cube

    def __isub__(self, other: Self):
        self.grid -= other.grid

        return self

    def __mul__(self, other: Self):
        new_cube = copy.deepcopy(self)

        if isinstance(other, float | int):
            new_cube.grid *= other
        else:
            new_cube.grid *= other.data

        return new_cube

    def __imul__(self, other: Self):
        if isinstance(other, float | int):
            self.grid *= other
        else:
            self.grid *= other.data

        return self

    def _calculate_cube_vectors(self) -> None:
        self._cube_vectors = ((self.grid_vectors.T) * self.shape).T

        self._dV = np.abs(
            get_triple_product(
                self.grid_vectors[0, :],
                self.grid_vectors[1, :],
                self.grid_vectors[2, :],
            )
        )

        self._dv1 = np.linalg.norm(self.grid_vectors[0, :])
        self._dv2 = np.linalg.norm(self.grid_vectors[1, :])
        self._dv3 = np.linalg.norm(self.grid_vectors[2, :])

    def get_periodic_replica(self, periodic_replica: tuple) -> Self:
        new_cube = copy.deepcopy(self)

        # add geometry
        geom = copy.deepcopy(self.geometry)
        geom.lattice_vectors = self.cube_vectors
        new_geom = geom.get_periodic_replica(periodic_replica)
        new_cube.geometry = new_geom

        # add data
        new_cube.grid = np.tile(self.grid, periodic_replica)

        # add lattice vectors
        new_shape = self.shape * np.array(periodic_replica)
        new_cube._shape = new_shape
        new_cube._calculate_cube_vectors()
        new_cube._n_points = len(new_cube.grid)

        return new_cube

    def save_to_file(self, name: str | Path) -> None:
        """
        Save cube file.

        Parameters
        ----------
        new_file : str Path
            name of the new cube file
        """
        if isinstance(name, str):
            filename = Path(name)
        elif isinstance(name, Path):
            filename = name

        filename = filename.with_suffix(".cube")

        header = ""

        # comments
        header += self.comment

        # cube file needs exactly 2 comment lines
        n_comment_lines = header.count("\n")
        if n_comment_lines < 2:
            header += "\n" * (2 - n_comment_lines)
        elif n_comment_lines > 2:
            split_head = header.split("\n")
            header = (
                split_head[0]
                + "\n"
                + " ".join(split_head[1:-1])
                + "\n"
                + split_head[-1]
            )

        # n_atoms and origin
        header += (
            f"{self.n_atoms:5d}"
            + "   "
            + "   ".join([f"{x / BOHR_IN_ANGSTROM: 10.6f}" for x in self.origin])
            + "\n"
        )

        # lattice vectors
        for i in range(3):
            header += (
                f"{self.shape[i]:5d}"
                + "   "
                + "   ".join(
                    [
                        f"{self.grid_vectors[i, j] / BOHR_IN_ANGSTROM: 10.6f}"
                        for j in range(3)
                    ]
                )
                + "\n"
            )

        # atoms
        atom_pos = self.geometry.coords
        atom_Z = [PeriodicTable.get_atomic_number(x) for x in self.geometry.species]
        for i in range(self.n_atoms):
            header += (
                f"{atom_Z[i]:5d}"
                + "   "
                + "0.000000"
                + "   "
                + "   ".join(
                    [f"{atom_pos[i, j] / BOHR_IN_ANGSTROM: 10.6f}" for j in range(3)]
                )
                + "\n"
            )

        x_len = self.shape[0]
        y_len = self.shape[1]
        z_len = self.shape[2]
        str_arr_size = int(x_len * y_len)
        string_array = np.empty([str_arr_size, z_len], dtype="<U18")

        # values
        for ix in range(x_len):
            for iy in range(y_len):
                for iz in range(z_len):
                    # for each ix we are consecutively writing all iy elements
                    string_array[ix * y_len + iy, iz] = f" {self.grid[ix, iy, iz]: .8e}"

        # write to file
        with filename.open(mode="w", newline="\n") as f:
            f.write(header)
            for i in range(str_arr_size):
                for j in range(int(np.ceil(z_len / 6))):
                    start_ind = 6 * j
                    end_ind = 6 * (j + 1)
                    end_ind = min(end_ind, z_len)
                    f.write("".join(string_array[i, start_ind:end_ind]) + "\n")

    def get_on_sparser_grid(
        self, reduction_factors: tuple[int, int, int]
    ) -> npt.NDArray:
        """
        TODO.

        Parameters
        ----------
        reduction_factors : tuple[int, int, int]
            factors to reduce the grid by in each dimension

        Returns
        -------
        rho : npt.NDArray
            reduced grid
        """
        rho = self.grid[:: reduction_factors[0], :, :]
        rho = rho[:, :: reduction_factors[1], :]
        return rho[:, :, :: reduction_factors[2]]

    def get_value_list(self) -> npt.NDArray[np.float64]:
        """
        TODO.

        Returns
        -------
        npt.NDArray[np.float64]
            TODO
        """
        return np.reshape(self.grid, [self.n_points])

    def get_point_list(self) -> Any:
        """
        TODO.

        Returns
        -------
        TODO
        also fix type annotations
        """
        ind = np.meshgrid(
            np.arange(self.shape[0]),
            np.arange(self.shape[1]),
            np.arange(self.shape[2]),
            indexing="ij",
        )

        fractional_point_list = np.array([i.ravel() for i in ind]).T
        r = np.dot(fractional_point_list, self.grid_vectors)

        return r + self.origin

    def get_point_coordinates(self) -> Any:
        """
        Create n1 x n2 x n3 x 3 array of coordinates for each data point.

        Returns
        -------
        TODO
        also fix type annotations
        """
        r = self.get_point_list()
        return np.reshape(r, [*self.shape, 3], order="C")

    def get_integrated_projection_on_axis(
        self, axis: Literal[0, 1, 2]
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Integrate cube file over the plane perpendicular to the selected axis.

        Returns
        -------
        proj : TODO
            projected values
        xaxis : TODO
            coordinates of axis (same length as proj)
        """
        axsum = list(range(3))
        axsum.pop(axis)

        dA = np.linalg.norm(
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
        tuple
            proj : npt.NDArray
                projected values
            xaxis : npt.NDArray
                coordinates of axis (same length as proj)
        """
        axsum = list(range(3))
        axsum.pop(axis)

        dA = np.linalg.norm(
            np.cross(self.grid_vectors[axsum[0], :], self.grid_vectors[axsum[1], :])
        )
        n_datapoints = self.shape[axsum[0]] * self.shape[axsum[1]]
        A = dA * n_datapoints

        # this gives sum(data) * dA
        proj, xaxis = self.get_integrated_projection_on_axis(axis)

        # remove dA from integration
        proj = proj / dA

        # average per area or pure mathematical average
        averaged = proj / A if divide_by_area else proj / n_datapoints

        return averaged, xaxis

    def get_charge_field_potential_along_axis(
        self, axis: Literal[0, 1, 2]
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        TODO.

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        axsum = list(range(3))
        axsum.pop(axis)

        dA = np.linalg.norm(
            np.cross(self.grid_vectors[axsum[0], :], self.grid_vectors[axsum[1], :])
        )
        n_datapoints = self.shape[axsum[0]] * self.shape[axsum[1]]
        A = dA * n_datapoints

        charge_density, axis_coords = self.get_integrated_projection_on_axis(2)

        cum_density = np.cumsum(charge_density) * self.dv3

        field = cum_density / EPSILON0_AIMS / A
        potential = -np.cumsum(field) * self.dv3

        return axis_coords, charge_density, cum_density, potential

    def get_voxel_coordinates(
        self,
    ) -> tuple[
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
    ]:
        v1_vec = (
            np.array([self.origin[0] + i * self.dv1 for i in range(self.shape[0])])
            - self.dv1 / 2
        )  # shift by half a grid vector to align voxel to center
        v2_vec = (
            np.array([self.origin[1] + i * self.dv2 for i in range(self.shape[1])])
            - self.dv2 / 2
        )
        v3_vec = (
            np.array([self.origin[2] + i * self.dv3 for i in range(self.shape[2])])
            - self.dv3 / 2
        )

        return v1_vec, v2_vec, v3_vec

    def get_voxel_coordinates_along_lattice(
        self, periodic_replica: tuple
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        TODO.

        unit cell is usually not at 90 degree angle therefore the plot of xy plane
        has to be projected onto the lattice vectors

        Parameters
        ----------
        periodic_replica : tuple
            number of periodic replicas in each direction

        Returns
        -------
        tuple
            v1_plot : NDArray
                x-coordinates of the grid points
            v2_plot : NDArray
                y-coordinates of the grid points
        """
        grid_vec = copy.deepcopy(self.grid_vectors)

        # get lattice vectors
        latt_mat = grid_vec[:-1, :-1]
        latt_mat[0, :] /= np.linalg.norm(latt_mat[0, :])
        latt_mat[1, :] /= np.linalg.norm(latt_mat[1, :])
        R = latt_mat.T

        # get points in cube grid
        v1_vec = (
            np.array([i * self.dv1 for i in range(self.shape[0] * periodic_replica[0])])
            - self.dv1 / 2
        )
        v2_vec = (
            np.array([i * self.dv2 for i in range(self.shape[1] * periodic_replica[1])])
            - self.dv2 / 2
        )
        v1, v2 = np.meshgrid(v1_vec, v2_vec)

        # project points onto lattice
        mult = np.dot(R, np.array([v1.ravel(), v2.ravel()]))
        v1_plot = mult[0, :].reshape(v1.shape) + self.origin[0]
        v2_plot = mult[1, :].reshape(v2.shape) + self.origin[1]

        return v1_plot, v2_plot

    def heights_for_constant_current(self, constant_current: float) -> npt.NDArray:
        """
        Find heights where the STM cube file current is closest to I = constant_current.

        Parameters
        ----------
        constant_current : float
            The current value to find the heights for.

        Returns
        -------
        TODO
        also fix type annotations
        """
        # difference of the current in each point to the constant_current
        delta = np.abs(self.grid - constant_current)

        # get indices of z-dimension of points that were closest to the current
        z_indices = np.argmin(delta, axis=2)

        # get the z-values that correspond to this indices
        _, _, v3_vec = self.get_voxel_coordinates()

        # create an array of the shape of indices with the heights
        # (repeat the v3_vec array to get to the shape of indices)
        heights = np.ones_like(z_indices)[:, :, np.newaxis] * v3_vec

        # cutout those hights that correspond to the indices
        x_indices, y_indices = np.indices(z_indices.shape)
        return heights[(x_indices, y_indices, z_indices)]

    def shift_content_along_vector(
        self,
        vec: npt.NDArray,
        repeat: bool = False,
        integer_only: bool = False,
        return_shift_indices: bool = False,
    ) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
        """
        Shifts values of the Cube along a specific vector.

        All values that are not known are set to zero.
        All values that are now outside the cube are deleted.
        TODO: Extrapolate unknown values
        ---------               ---------
        |xxxxxxx|               |00xxxxx| xx
        |xxxxxxx| shift by vec  |00xxxxx| xx  <-- deleted
        |xxxxxxx|      ---->    |00xxxxx| xx
        |xxxxxxx|               |00xxxxx| xx
        |xxxxxxx|               |00xxxxx| xx
        ---------               ---------

        Parameters
        ----------
        vec : npt.NDArray
            vector to shift the cube along
        repeat : bool, default=False
            if True, the cube is shifted in a periodic way
        integer_only : bool, default=False
            if True, the shift is rounded to the nearest integer
        return_shift_indices : bool, default=False
            if True, the shift indices are returned as well

        Returns
        -------
        Union
            tuple[npt.NDArray, npt.NDArray]
                shifted cube data and  shift indices
            npt.NDArray
                shifted cube data
        """
        # convert vec to indices
        trans_mat = copy.deepcopy(self.grid_vectors).T
        shift_inds = np.dot(np.linalg.inv(trans_mat), vec)

        if integer_only:
            shift_inds = shift_inds.astype(int)

        mode = "wrap" if repeat else "constant"
        data = shift(self.data, shift_inds, mode=mode)

        if return_shift_indices:
            return data, shift_inds

        return data

    def get_value_at_positions(
        self,
        coords: npt.NDArray,
        return_mapped_coords: bool = False,
        xy_periodic: bool = True,
    ) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
        """
        Get the value of the closest data point in cube grid.

        Parameters
        ----------
        coords : npt.NDArray
            List of Cartesian coordinates at which the cubefile values should
            be returned.
        return_mapped_coords : bool, default=False
            Return the Cartesian coordinates, minus the origin of the cubefile,
            of the grid point in the cubefile that is closest to the respective
            position in coords. The default is False.
        xy_periodic : bool, default=True
            If True, the x and y coordinates are treated as periodic. The
            default is True. If False, the x and y coordinates are not
            periodic. This means that if a coordinate is outside the grid, it
            will be set to the closest grid point.

        Returns
        -------
        NDArray
            Vaules at the grid point closest to the respective positions in
            coords
        """
        trans_mat = copy.deepcopy(self.grid_vectors).T
        coords = np.atleast_2d(coords)
        pos_inds = np.round(np.dot(np.linalg.inv(trans_mat), (coords - self.origin).T))
        pos_inds = pos_inds.astype(int)

        n_coords = np.shape(pos_inds)[1]
        if not xy_periodic:
            pos_inds[0, pos_inds[0, :] > self.shape[0]] = self.shape[0] - 1
            pos_inds[0, pos_inds[0, :] < 0] = 0
            pos_inds[1, pos_inds[1, :] > self.shape[1]] = self.shape[1] - 1
            pos_inds[1, pos_inds[1, :] < 0] = 0
        values = np.zeros([n_coords])

        for i in range(n_coords):
            x, y, z = pos_inds[0, i], pos_inds[1, i], pos_inds[2, i]
            if (
                isinstance(x, int)
                and isinstance(y, int)
                and isinstance(z, int)
                and 0 <= x < self.grid.shape[0]
                and 0 <= y < self.grid.shape[1]
                and 0 <= z < self.grid.shape[2]
            ):
                values[i] = self.grid[x, y, z]
            else:
                values[i] = np.nan

        if return_mapped_coords:
            return values, self.origin + np.dot(trans_mat, pos_inds).T

        return values

    def get_interpolated_value_at_positions(  # noqa: PLR0912, PLR0915
        self,
        coords: npt.NDArray,
        return_mapped_coords: bool = False,
        xy_periodic: bool = True,
    ) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
        """
        Get value of closest data point in cube grid.

        Parameters
        ----------
        coords : npt.NDArray
            List of Cartesian coordinates at which the cubefile values should
            be returned.
        return_mapped_coords : bool, default=False
            Return the Cartesian coordinates, minus the origin of the cubefile,
            of the grid point in the cubefile that is closest to the respective
            position in coords.
        xy_periodic : bool, default=True
            If True, the x and y coordinates are treated as periodic.

        Returns
        -------
        Union
            npt.NDArray
                Values at the grid point closest to the respective positions in
                coords
            tuple[npt.NDArray, npt.NDArray]
                Values at the grid point closest to the respective positions in
                coords and the Cartesian coordinates of the grid point in the
                cubefile that is closest to the respective position in coords.
        """
        trans_mat = copy.deepcopy(self.grid_vectors).T
        coords = np.atleast_2d(coords)
        pos_inds_0 = np.dot(np.linalg.inv(trans_mat), (coords - self.origin).T)
        pos_inds = np.round(pos_inds_0).astype(int)

        n_coords = np.shape(pos_inds)[1]
        if not xy_periodic:
            pos_inds[0, pos_inds[0, :] >= self.shape[0]] = self.shape[0] - 1
            pos_inds[0, pos_inds[0, :] < 0] = 0
            pos_inds[1, pos_inds[1, :] >= self.shape[1]] = self.shape[1] - 1
            pos_inds[1, pos_inds[1, :] < 0] = 0
        else:
            pos_inds_0[0, :] = pos_inds_0[0, :] % self.shape[0]
            pos_inds_0[1, :] = pos_inds_0[1, :] % self.shape[1]

            pos_inds[0, :] = pos_inds[0, :] % self.shape[0]
            pos_inds[1, :] = pos_inds[1, :] % self.shape[1]

        pos_inds[2, pos_inds[2, :] >= self.shape[2]] = self.shape[2] - 1
        pos_inds[2, pos_inds[2, :] < 0] = 0

        values = np.zeros([n_coords])

        difference = pos_inds_0 - pos_inds

        if xy_periodic:
            difference[0, difference[0, :] > 1.0] = difference[0, :] - self.shape[0]
            difference[1, difference[1, :] > 1.0] = difference[1, :] - self.shape[1]

        for i in range(n_coords):
            pos_inds_x = pos_inds[:, i] + np.array([np.sign(difference[0])[i], 0, 0])
            pos_inds_y = pos_inds[:, i] + np.array([0, np.sign(difference[1])[i], 0])
            pos_inds_z = pos_inds[:, i] + np.array([0, 0, np.sign(difference[2])[i]])

            # periodic boundary conditions
            if not xy_periodic:
                if pos_inds_x[0] >= self.shape[0]:
                    pos_inds_x[0] = self.shape[0] - 1

                pos_inds_x[0] = max(pos_inds_x[0], 0)

                if pos_inds_y[1] >= self.shape[1]:
                    pos_inds_y[1] = self.shape[1] - 1

                pos_inds_y[1] = max(pos_inds_y[1], 0)

            else:
                if pos_inds_x[0] >= self.shape[0]:
                    pos_inds_x[0] = self.shape[0] - pos_inds_x[0]

                if pos_inds_y[1] >= self.shape[1]:
                    pos_inds_y[1] = self.shape[1] - pos_inds_y[1]

            if pos_inds_z[2] >= self.shape[2]:
                pos_inds_z[2] = self.shape[2] - 1

            pos_inds_z[2] = max(pos_inds_z[2], 0)

            pos_inds_x = pos_inds_x.astype(int)
            pos_inds_y = pos_inds_y.astype(int)
            pos_inds_z = pos_inds_z.astype(int)

            values_0 = self.grid[pos_inds[0, i], pos_inds[1, i], pos_inds[2, i]]
            values_x = self.grid[pos_inds_x[0], pos_inds_x[1], pos_inds_x[2]]
            values_y = self.grid[pos_inds_y[0], pos_inds_y[1], pos_inds_y[2]]
            values_z = self.grid[pos_inds_z[0], pos_inds_z[1], pos_inds_z[2]]

            d_v_x = (values_x - values_0) / np.sign(difference[0])[i]
            d_v_y = (values_y - values_0) / np.sign(difference[1])[i]
            d_v_z = (values_z - values_0) / np.sign(difference[2])[i]

            if np.isnan(d_v_x):
                d_v_x = 0
            if np.isnan(d_v_y):
                d_v_y = 0
            if np.isnan(d_v_z):
                d_v_z = 0

            normal_vector = np.array([-d_v_x, -d_v_y, -d_v_z, 1.0])
            normal_vector /= np.linalg.norm(normal_vector)

            values[i] = (
                normal_vector[3] * values_0
                - normal_vector[0] * difference[0][i]
                - normal_vector[1] * difference[1][i]
                - normal_vector[2] * difference[2][i]
            ) / normal_vector[3]

        if return_mapped_coords:
            return values, self.origin + np.dot(trans_mat, pos_inds).T

        return values

    def calculate_distance_to_local_geometry(
        self, adsorption_geometry: Geometry
    ) -> None:
        """
        Compute distance between cube file molecule and its local adsorption geometry.

        Parameters
        ----------
        adsorption_geometry : Geometry
            Geometry of the local adsorption geometry.
        """
        raise NotImplementedError(
            "This was originally implemented using `Geometry.get_molecules()` but "
            "this is (no longer?) defined as an attribute of Geometry"
        )

        cube_mol = self.geometry.get_molecules()
        cube_geom_center = cube_mol.get_geometric_center(ignore_center_attribute=True)
        ads_geom_center = adsorption_geometry.get_geometric_center(
            ignore_center_attribute=True
        )
        distance_to_adsorption_geometry = ads_geom_center - cube_geom_center
        coord_diff = np.max(
            cube_mol.coords
            + distance_to_adsorption_geometry
            - adsorption_geometry.coords
        )

        if coord_diff < 5e-2:
            msg = (
                "Local Geometry doesnt match cube geometry!, difference is ",
                coord_diff,
            )
            raise ValueError(msg)

        self.corresponding_adsorption_geometry = adsorption_geometry
        self.distance_to_adsorption_geometry = ads_geom_center - cube_geom_center

    def get_values_on_plane(
        self,
        plane_centre: npt.NDArray,
        plane_normal: npt.NDArray,
        plane_extent: float,
        plane_points: int = 100,
    ) -> npt.NDArray:
        """
        Retruns the cubefile values on a given plane.

        Parameters
        ----------
        plane_centre : NDArray
            Centre of the plane.
        plane_normal : NDArray
            Vector normal to the plane, i.e. viewing direction.
        plane_extent : float
            Size of the plane in Angstrom

        Returns
        -------
        values_on_plane : NDArray
        """
        plane_normal /= np.linalg.norm(plane_normal)

        vec_z = np.array([0.0, 0.0, 1.0])

        plane_vec_xy = np.cross(vec_z, plane_normal)
        plane_vec_xy /= np.linalg.norm(plane_vec_xy)
        plane_vec_z = np.cross(plane_normal, plane_vec_xy)
        plane_vec_z /= np.linalg.norm(plane_vec_z)

        extent_vec = np.linspace(-plane_extent, plane_extent, plane_points)

        values_on_plane = np.zeros((len(extent_vec), len(extent_vec)))

        max_dist = (self.dv1 + self.dv2 + self.dv3) / 3

        for ind_1, x in enumerate(extent_vec):
            for ind_2, y in enumerate(extent_vec):
                plane_pos = plane_centre - x * plane_vec_xy + y * plane_vec_z

                value, mapped_coords = self.get_value_at_positions(
                    plane_pos, return_mapped_coords=True
                )

                vec = mapped_coords - plane_pos + self.origin
                mag = np.linalg.norm(vec)

                if mag < max_dist:
                    values_on_plane[ind_1, ind_2] = value

        return values_on_plane

    def calculate_overlap_integral(
        self,
        other: Self,
        print_normalisation_factors: bool = True,
        take_absolute_value: bool = True,
        output_overlap_cube: bool = False,
    ) -> float | tuple[float, Self]:
        """
        Calculate the overlap integral of some quantity with another cubefile.

        NOTE: this is written to work with the standard FHI-aims voxels in Angstrom³
        NOTE: the two orbitals should describe the same exact volume of space!

        Parameters
        ----------
        other: Self
            Cubefile to calculate the overlap with.
        print_normalisation_factors: bool, default=True
            Print the normalisation factors.
        take_absolute_value: bool, default=True
            Take the absolute value of the overlap.

        Returns
        -------
        Union
            float
                Overlap integral of the two cubefiles.
            tuple[float, Self]
                Overlap integral of the two cubefiles and the overlap cubefile.
        """
        # this data is normally provided in angstrom^(-3/2)
        first = copy.deepcopy(self.grid)
        second = copy.deepcopy(other.grid)

        # let's pass to bohr_radius^(-3/2)
        first *= np.sqrt(BOHR_IN_ANGSTROM**3)
        second *= np.sqrt(BOHR_IN_ANGSTROM**3)

        # both arrays get normalised
        first_squared = first * first
        second_squared = second * second
        first_total = first_squared.sum()
        second_total = second_squared.sum()
        first_normalization_factor = np.sqrt(1 / first_total)
        second_normalization_factor = np.sqrt(1 / second_total)

        if print_normalisation_factors:
            print("Normalization factors:")
            print("self: ", first_normalization_factor)
            print("other cubefile: ", second_normalization_factor)

        first *= first_normalization_factor
        second *= second_normalization_factor

        # Now the sum corresponds to the overall overlap (we are integrating in d(Voxel)
        # which is d(bohr_radius^(-3)) ), and we take its absolute value (overlap has
        # no sign)
        product = first * second

        overlap_cube = None

        if output_overlap_cube:
            overlap_cube = copy.deepcopy(self)
            overlap_cube.data = product

        overlap = np.sum(product)

        if take_absolute_value:
            overlap = np.abs(overlap)

        if output_overlap_cube:
            return overlap, overlap_cube  # pyright: ignore[reportReturnType]

        return overlap

    def get_eigenstate_number(self) -> int:
        """
        Get the eigenstate number from the filename.

        Returns
        -------
        int
            The eigenstate number.
        """
        components = self._path.stem.split("_")
        eig_num_i = components.index("eigenstate") + 1

        return int(components[eig_num_i])

    def get_spin_channel(self) -> int:
        """
        Get the spin channel from the filename.

        Returns
        -------
        int
            The spin channel.
        """
        components = self._path.stem.split("_")
        spin_i = components.index("spin") + 1

        return int(components[spin_i])
