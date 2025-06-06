from typing import Any
from warnings import warn

import numpy as np
import numpy.typing as npt

from .base import Parser
from .utils.file_utils import MultiDict
from .utils.periodic_table import PeriodicTable


class Parameters(Parser):
    """
    Handle files that control parameters for electronic structure calculations.

    If contributing a new parser, please subclass this class, add the new supported file
    type to _supported_files and match statement in this class' `__init__()`, and call
    the `super().__init__()` method, include the new file type as a kwarg in the
    `super().__init__()`. Optionally include the `self.lines` line
    in examples.

    ...

    Attributes
    ----------
    _supported_files : dict
        List of supported file types.
    """

    def __init__(self, **kwargs: str):
        # Parse file information and perform checks
        super().__init__(self._supported_files, **kwargs)

        self._check_binary(False)

    @property
    def _supported_files(self) -> dict:
        # FHI-aims, ...
        return {"control_in": ".in", "cube": ".cube"}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._format}={self._name})"

    def __init_subclass__(cls, **kwargs: str):
        # Override the parent's __init_subclass__ without calling it
        pass


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

    def __init__(self, control_in: str = "control.in"):
        super().__init__(control_in=control_in)

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
        self, elements: list[str] | None = None
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
            set(PeriodicTable.element_symbols())
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
        self, cell_matrix: npt.NDArray, resolution: int = 100
    ) -> None:
        """
        Add cube output settings to control.in to cover the unit cell specified in
        `cell_matrix` and save to disk.

        Since the default behaviour of FHI-AIMS for generating cube files for periodic
        structures with vacuum gives confusing results, this function ensures the cube
        output quantity is calculated for the full unit cell.

        Parameters
        ----------
        cell_matrix : NDArray
            2D array defining the unit cell.

        resolution : int
            Number of cube voxels to use for the shortest side of the unit cell.

        """  # noqa: D205
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
                if len(spl) > 0 and spl[0] != "#" and keyword == spl[0]:
                    self.lines.pop(i)

        with open(self.path, "w") as f:
            f.writelines(self.lines)

    def check_periodic(self) -> bool:
        """Check if the system is periodic."""
        return "k_grid" in self.get_keywords()


class CubeParameters(Parameters):
    """
    Cube file settings that can be used to generate a control file.

    Attributes
    ----------
    type : str
        type of cube file; all that comes after output cube

    Parameters
    ----------
    cube: str
        path to the cube file
    text: str | None
        text to parse

    Functions
    -------------------
        parse(text): parses textlines

        getText(): returns cubefile specifications-string for ControlFile class
    """

    def __init__(self, cube: str = "cube.cube", text: str | None = None):
        super().__init__(cube=cube)

        self._check_binary(False)

        # Set attrs here rather than `File.__post_init__()` as `Cube.__init__()` uses
        # ASE to parse the data from a cube file, so it's definied in `Cube.__init__()`
        # so `File.__post_init__()` doesn't add these attributes if a cube file
        # extension is detected.
        with open(self.path) as f:
            self.lines = f.readlines()
            self.data = b""
            self._binary = False

        self._type = ""

        # parsers for specific cube keywords:
        # keyword: string_to_number, number_to_string
        self._parsing_functions = {
            "spinstate": [
                lambda x: int(x[0]),
                lambda x: str(x),
            ],
            "kpoint": [lambda x: int(x[0]), lambda x: str(x)],
            "divisor": [lambda x: int(x[0]), lambda x: str(x)],
            "spinmask": [
                lambda x: [int(k) for k in x],
                lambda x: "  ".join([str(k) for k in x]),
            ],
            "origin": [
                lambda x: [float(k) for k in x],
                lambda x: "  ".join([f"{k: 15.10f}" for k in x]),
            ],
            "edge": [
                lambda x: [int(x[0])] + [float(k) for k in x[1:]],
                lambda x: str(int(x[0]))
                + "  "
                + "  ".join([f"{k: 15.10f}" for k in x[1:]]),
            ],
        }

        self._settings = MultiDict()

        if text is not None:
            self.parse(text)

    def __repr__(self):
        text = "CubeSettings object with content:\n"
        text += self.get_text()
        return text

    @property
    def type(self) -> str:
        """Everythin that comes after output cube as a single string."""
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        """Set the type of the cube file."""
        self._type = value

    @property
    def parsing_functions(self) -> dict[str, list[int | str]]:
        """Parsing functions for specific cube keywords."""
        return self._parsing_functions

    @property
    def settings(self) -> MultiDict:
        """Settings for the cube file."""
        return self._settings

    @property
    def origin(self) -> npt.NDArray[np.float64]:
        """Origin of the cube file."""
        raise NotImplementedError(
            "Decide if this property should return the "
            "dictionary value or the first component as a numpy array"
        )

        return self.setting["origin"]
        return np.array(self.settings["origin"][0])

    @origin.setter
    def origin(self, origin: npt.NDArray[np.float64]) -> None:
        self.settings["origin"] = [[origin[0], origin[1], origin[2]]]

    @property
    def edges(self) -> npt.NDArray[np.float64]:
        """Set the edge vectors."""
        return np.array(self.settings["edge"])

    @edges.setter
    def edges(self, edges: tuple[list[int], list[float]]) -> None:
        """
        TODO.

        Parameters
        ----------
        edges : tuple[list[int], list[float]]
            TODO
        """
        raise NotImplementedError("Type annotations need to be fixed")

        self.settings["edge"] = []
        for i, d in enumerate(edges[0]):
            self.settings["edge"].append([d, *list(edges[1][i, :])])

    @property
    def grid_vectors(self) -> float:
        raise NotImplementedError("See edges.setter")

        edges = self.edges
        return edges[:, 1:]

    @property
    def divisions(self) -> float:
        raise NotImplementedError("See edges.setter")

        edges = self.edges
        return edges[:, 0]

    @divisions.setter
    def divisions(self, divs: npt.NDArray[np.float64]) -> None:
        if len(divs) != 3:
            raise ValueError("Requires divisions for all three lattice vectors")

        for i in range(3):
            self.settings["edge"][i][0] = divs[i]

    def parse(self, text: str) -> None:
        """
        TODO.

        Parameters
        ----------
        str
            TODO
        """
        cubelines = []
        for line in text:
            strip = line.strip()
            # parse only lines that start with cube and are not comments
            if not strip.startswith("#"):
                if strip.startswith("cube"):
                    cubelines.append(strip)
                elif strip.startswith("output"):
                    self.type = " ".join(strip.split()[2:])

        # parse cubelines to self.settings
        for line in cubelines:
            nc_lines = line.split("#")[0]  # remove comments
            splitline = nc_lines.split()
            keyword = splitline[1]  # parse keyword
            values = splitline[2:]  # parse all values

            # check if parsing function exists
            if keyword in self.parsing_functions:
                value = self.parsing_functions[keyword]

            # reconvert to single string otherwise
            else:
                value = " ".join(values)

            # save all values as list, append to list if key already exists
            if keyword in self.settings:
                self.settings[keyword].append(value)
            else:
                self.settings[keyword] = [value]

    def has_vertical_unit_cell(self) -> bool:
        conditions = [
            self.settings["edge"][0][3] == 0.0,
            self.settings["edge"][1][3] == 0.0,
            self.settings["edge"][2][1] == 0.0,
            self.settings["edge"][2][1] == 0.0,
        ]
        return False not in conditions

    def set_z_slice(self, z_bottom: float, z_top: float) -> None:
        """
        Crops the cubefile to only include the space between z_bottom and z_top.

        The cubefile could go slightly beyond z_bottom and z_top in order to preserve
        the distance between grid points.

        Parameters
        ----------
        z_bottom: float
            TODO
        z_top: float
            TODO
        """
        if z_top < z_bottom:
            raise ValueError("Ensure that `z_bottom` is smaller than `z_top`")

        if not self.has_vertical_unit_cell():
            raise ValueError(
                "This function is only supported for systems where the "
                "cell is parallel to the z-axis"
            )

        diff = z_top - z_bottom
        average = z_bottom + diff / 2

        # set origin Z
        self.settings["origin"][0][2] = average

        # set edge, approximating for excess
        z_size = self.settings["edge"][2][0] * self.settings["edge"][2][3]
        fraction_of_z_size = z_size / diff
        new_z = self.settings["edge"][2][0] / fraction_of_z_size

        if new_z % 1 != 0:
            new_z = int(new_z) + 1.0

        self.settings["edge"][2][0] = new_z

    def set_grid_by_box_dimensions(
        self,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        z_limits: tuple[float, float],
        spacing: float | tuple[float, float, float],
    ) -> None:
        """
        Set origin and edge as a cuboid box.

        The ranging is within the given limits, with voxel size specified by spacing.

        Parameters
        ----------
        x_limits: tuple[float, float]
            min and max of...TODO
        y_limits: tuple[float, float]
            min and max of...TODO
        z_limits: tuple[float, float]
            min and max of...TODO
        spacing: float | tuple[float, float, float]
            TODO
        """
        raise NotImplementedError("Origin parameter needs to be fixed")

        # TODO: why is this necessary?
        self.origin = [0, 0, 0]
        self.settings["edge"] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        # set one dimension at a time
        for i, lim in enumerate([x_limits, y_limits, z_limits]):
            if lim[0] >= lim[1]:
                raise ValueError("Ensure the minimum is given first")

            diff = lim[1] - lim[0]

            # set origin
            center = lim[0] + (diff / 2)
            self.settings["origin"][0][i] = center

            # set edges
            space = spacing[i] if isinstance(spacing, list) else spacing

            # size of voxel
            self.settings["edge"][i][i + 1] = space

            # number of voxels
            n_voxels = int(diff / space) + 1
            self.settings["edge"][i][0] = n_voxels

    def get_text(self) -> str:
        """
        TODO.

        Returns
        -------
        TODO
        """
        raise NotImplementedError("Fix self.parsing_functions type")

        text = ""
        if len(self.type) > 0:
            text += "output cube " + self.type + "\n"
        else:
            warn("No cube type specified", stacklevel=2)
            text += "output cube" + "CUBETYPE" + "\n"

        for key, values in self.settings.items():
            for v in values:
                text += "cube " + key + " "
                if key in self.parsing_functions:
                    text += self.parsing_functions[key][1](v) + "\n"
                else:
                    print(v)
                    text += v + "\n"

        return text
