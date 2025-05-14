from typing import Any, Union
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from .base import Parser
from .utils.file_utils import MultiDict
from .utils.periodic_table import PeriodicTable


class Parameters(Parser):
    """
    Handle files that control parameters for electronic structure calculations.

    If contributing a new parser, please subclass this class, add the new supported file
    type to _supported_files and match statement in this class' __init__, and call the super().__init__ method, include the new file
    type as a kwarg in the super().__init__ call. Optionally include the self.lines line
    in examples.

    ...

    Attributes
    ----------
    _supported_files : dict
        List of supported file types.
    """

    def __init__(self, **kwargs):
        # Parse file information and perform checks
        super().__init__(self._supported_files, **kwargs)

        self._check_binary(False)

    @property
    def _supported_files(self) -> dict:
        # FHI-aims, ...
        return {"control_in": ".in"}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._format}={self._name})"

    def __init_subclass__(cls, **kwargs):
        # Revert back to the original __init_subclass__ method to avoid checking for
        # required methods in child class of this class too
        return super(Parser, cls).__init_subclass__(**kwargs)


class AimsControl(Parameters):
    """
    FHI-aims control file parser.

    ...

    Attributes
    ----------
    path: str
        path to the aims.out file
    lines: List[str]
        contents of the aims.out file

    Examples
    --------
    >>> ac = AimsControl(control_in="./control.in")
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Use normal methods instead of properties for these methods as we want to specify
    # the setter method using kwargs instead of assigning the value as a dictionary.
    # Then, for consistency, keep get_keywords as a normal function.
    def get_keywords(self) -> MultiDict:
        """
        Get the keywords from the control.in file.

        Returns
        -------
        MultiDict
            Keywords in the control.in file.
        """
        keywords = MultiDict()

        for line in self.lines:
            # Stop at third keyword delimiter if ASE wrote the file
            spl = line.split()
            if len(spl) > 0 and spl[-1] == "(ASE)":
                n_delims = 0
                if line == "#" + ("=" * 79):
                    n_delims += 1
                    if n_delims == 3:
                        # Reached end of keywords
                        break

            elif "#" * 80 in line.strip():
                # Reached the basis set definitions
                break

            if len(spl) > 0 and line[0] != "#":
                keywords[spl[0]] = " ".join(spl[1:])

        return keywords

    def get_species(self) -> list[str]:
        """
        Get the species from a control.in file.

        Returns
        -------
        List[str]
            A list of the species in the control.in file.
        """
        species = []
        for line in self.lines:
            spl = line.split()
            if len(spl) > 0 and spl[0] == "species":
                species.append(line.split()[1])

        return species

    def get_default_basis_funcs(
        self, elements: Union[list[str], None] = None
    ) -> dict[str, str]:
        """
        Get the basis functions.

        Parameters
        ----------
        elements : List[str], optional, default=None
            The elements to parse the basis functions for as chemical symbols.

        Returns
        -------
        Dict[str, str]
            A dictionary of the basis functions for the specified elements.
        """
        # Check that the given elements are valid
        if elements is not None and not set(elements).issubset(
            set(PeriodicTable.element_symbols)
        ):
            raise ValueError("Invalid element(s) given")

        # Warn if the requested elements aren't found in control.in
        if elements is not None and not set(elements).issubset(self.get_species()):
            warn("Could not find all requested elements in control.in", stacklevel=2)

        basis_funcs = {}

        for i, line_1 in enumerate(self.lines):
            spl_1 = line_1.split()
            if "species" in spl_1[0]:
                species = spl_1[1]

                if elements is not None and species not in elements:
                    continue

                for line_2 in self.lines[i + 1 :]:
                    spl = line_2.split()
                    if "species" in spl[0]:
                        break

                    if "#" in spl[0]:
                        continue

                    if "hydro" in line_2:
                        if species in basis_funcs:
                            basis_funcs[species].append(line_2.strip())
                        else:
                            basis_funcs[species] = [line_2.strip()]

        return basis_funcs

    def add_keywords_and_save(self, *args: tuple[str, Any]) -> None:
        """
        Add keywords to the control.in file and write the new control.in to disk.

        Note that files written by ASE or in a format where the keywords are at the top
        of the file followed by the basis sets are the only formats that are supported
        by this function. The keywords need to be added in a Tuple format rather than as
        **kwargs because we need to be able to add multiple of the same keyword.

        Parameters
        ----------
        *args : Tuple[str, Any]
            Keywords to be added to the control.in file.
        """
        # Get the location of the start of the basis sets
        basis_set_start = False

        # if ASE wrote the file, use the 'add' point as the end of keywords delimiter
        # otherwise, use the start of the basis sets as 'add' point
        for i, line_1 in enumerate(self.lines):
            if line_1.strip() == "#" * 80:
                if self.lines[2].split()[-1] == "(ASE)":
                    for j, line_2 in enumerate(reversed(self.lines[:i])):
                        if line_2.strip() == "#" + ("=" * 79):
                            basis_set_start = i - j - 1
                            break
                    break

                # not ASE
                basis_set_start = i
                break

        # Check to make sure basis sets were found
        if not basis_set_start:
            raise IndexError("Could not detect basis sets in control.in")

        # Add the new keywords above the basis sets
        for arg in reversed(args):
            self.lines.insert(basis_set_start, f"{arg[0]:<34} {arg[1]}\n")

        # Write the file
        with open(self.path, "w") as f:
            f.writelines(self.lines)

    def add_cube_cell_and_save(
        self, cell_matrix: NDArray[Any], resolution: int = 100
    ) -> None:
        """
        Add cube output settings to control.in to cover the unit cell specified in
        `cell_matrix` and save to disk.

        Since the default behaviour of FHI-AIMS for generating CUBE files for periodic
        structures with vacuum gives confusing results, this function ensures the cube
        output quantity is calculated for the full unit cell.

        Parameters
        ----------
        cell_matrix : ArrayLike
            2D array defining the unit cell.

        resolution : int
            Number of cube voxels to use for the shortest side of the unit cell.

        """
        if not self.check_periodic():  # Fail for non-periodic structures
            raise TypeError("add_cube_cell doesn't support non-periodic structures")

        shortest_side = min(np.sum(cell_matrix, axis=1))
        resolution = shortest_side / 100.0

        cube_x = (
            2 * int(np.ceil(0.5 * np.linalg.norm(cell_matrix[0, :]) / resolution)) + 1
        )  # Number of cubes along x axis
        x_vector = cell_matrix[0, :] / np.linalg.norm(cell_matrix[0, :]) * resolution
        cube_y = (
            2 * int(np.ceil(0.5 * np.linalg.norm(cell_matrix[1, :]) / resolution)) + 1
        )
        y_vector = cell_matrix[1, :] / np.linalg.norm(cell_matrix[1, :]) * resolution
        cube_z = (
            2 * int(np.ceil(0.5 * np.linalg.norm(cell_matrix[2, :]) / resolution)) + 1
        )
        z_vector = cell_matrix[2, :] / np.linalg.norm(cell_matrix[2, :]) * resolution
        self.add_keywords_and_save(  # Add cube options to control.in
            (
                "cube",
                "origin {} {} {}\n".format(
                    *(np.transpose(cell_matrix @ [0.5, 0.5, 0.5]))
                )
                + "cube edge {} {} {} {}\n".format(cube_x, *x_vector)
                + "cube edge {} {} {} {}\n".format(cube_y, *y_vector)
                + "cube edge {} {} {} {}\n".format(cube_z, *z_vector),
            )
        )
        # print("\tCube voxel resolution is {} Å".format(resolution))

    def remove_keywords_and_save(self, *args: str) -> None:
        """
        Remove keywords from the control.in file and save to disk.

        Note that this will not remove keywords that are commented with a '#'.

        Parameters
        ----------
        *args : str
            Keywords to be removed from the control.in file.
        """
        for keyword in args:
            for i, line in enumerate(self.lines):
                spl = line.split()
                if len(spl) > 0 and spl[0] != "#" and keyword in line:
                    self.lines.pop(i)

        with open(self.path, "w") as f:
            f.writelines(self.lines)

    def check_periodic(self) -> bool:
        """Check if the system is periodic."""
        return "k_grid" in self.get_keywords()
