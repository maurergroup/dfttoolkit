import collections
import os
import re
from typing import Any
from warnings import warn

import numpy as np
import numpy.typing as npt

from .base import Parser
from .utils.file_utils import MultiDict


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
    >>> cf['k_points'] = '10 10 1'
    >>> newSpecies = SpeciesDefault('path/to/default/species.txt')
    >>> cf.species_definitions['newSpec'] = newSpecies
    >>> cf.saveToFile('path/to/destination/control.in')
    """

    def __init__(self, control_in: str | None):
        super().__init__(control_in=control_in)

        self.control_in = control_in
        self.species_definitions = collections.OrderedDict()
        self.cubes = (
            []
        )  # list of cubefile settings objects which get written into control.in
        self.band_output = {}  # put band segments in a dictionary
        self.dos_output = {}  # dos output
        self.vdw_corrections = {}
        self.additional_output = {}  # all other setting from the 'output' tag
        if control_in is None:
            # reasonable default settings
            self.settings = {
                "xc": "pbe",
                "spin": "none",
                "charge": 0,
                "relativistic": "atomic_zora scalar",
                # 'occupation_type':'gaussian 0.01',
                # 'compensate_multipole_errors':'.true.',
                "sc_accuracy_rho": 1e-5,
                "sc_accuracy_etot": 1e-6,
            }
        else:
            self.settings = {}
            self.band_output["band"] = []
            self.band_output["band_mulliken"] = []
            self.read_from_file(control_in)

    def read_from_file(self, filename):
        with open(filename) as file:
            content = [
                line.strip() for line in file.readlines() if line.strip() != ""
            ]
        self.parse(content)

    def parse(self, content):
        cubelines = []  # prepare for cube file setting if present
        species_lines = []
        species_keywords = [
            "element",
            "nucleus",
            "mass",
            "l_hartree",
            "cut_pot",
            "basis_dep_cutoff",
            "radial_base",
            "radial_multiplier",
            "angular_grids",
            "division",
            "outer_grid",
            "valence",
            "ion_occ",
            "hydro",
            "ionic",
            "include_min_basis",
            "hirshfeld_param",
            "for_aux",
        ]
        mbd_keywords = [
            "vdw_correction_hirshfeld",
            "many_body_dispersion",
            "many_body_dispersion_nl",
        ]

        for ind_line, line in enumerate(content):
            if line.startswith("#"):
                continue
            _line = line.split("#")[0].strip()  # remove trailing comments

            keyword = _line.split()[0]
            if keyword == "output":
                # check for some output types, this is far from complete but the important ones are covered
                output_type = _line.split()[1]
                if re.match(".*dos.*", output_type):
                    self.dos_output[output_type] = " ".join(_line.split()[2:])
                elif output_type in ["band", "band_mulliken"]:
                    self.parseBandSegment(_line, output_type)
                elif output_type == "cube":
                    # reset cube file information if already parsed
                    if cubelines:
                        self.cubes.append(CubeParameters(text=cubelines))
                        cubelines = []
                    cubelines.append(_line)
                # catch-all for other output types
                else:
                    self.additional_output[output_type] = " ".join(
                        _line.split()[2:]
                    )
            elif keyword == "cube":
                cubelines.append(_line)
            # TODO: properly parse mbdnl
            elif keyword == "periodic_hf":
                self.settings[_line.split()[1]] = _line.split()[2]
            elif keyword == "species":
                species_lines.append(ind_line)
            elif keyword in mbd_keywords:
                self.vdw_corrections[keyword] = {}
                components = _line.split()
                if len(components) > 1:
                    for tag in components[1:]:
                        self.vdw_corrections[keyword][tag.split("=")[0]] = (
                            tag.split("=")[1]
                        )
            elif keyword == "vdw_correction_hirshfeld":
                self.vdw_corrections[keyword] = {}
            # skip species keywords
            elif keyword in species_keywords:
                pass
            elif keyword == "k_grid":
                self.settings[keyword] = [
                    int(x) for x in list(line.split())[1:]
                ]
            # catch-all for keywords with simple key-value structure
            else:
                self.settings[keyword] = " ".join(_line.split()[1:])

        # add remaining cube file settings
        if cubelines:
            self.cubes.append(CubeParameters(text=cubelines))
        # extract species definitions (does nothing if specieslines is empty)
        for i, s in enumerate(species_lines):
            if s == species_lines[-1]:
                species_text = "\n".join(content[s : len(content)])
            else:
                species_text = "\n".join(content[s : species_lines[i + 1]])
            species_to_add = SpeciesDefinition(species_text)
            self.species_definitions[species_to_add.species] = species_to_add

        # clean-up
        for band_type in ["band", "band_mulliken"]:
            if (
                band_type in self.band_output
                and not self.band_output[band_type]
            ):
                self.band_output.pop(band_type)

    def parse_band_segment(self, line, band_type):
        segment = line.split()[2:]
        k_start = [float(k_point) for k_point in segment[:3]]
        k_end = [float(k_point) for k_point in segment[3:6]]
        n_points = int(segment[6])
        name_start = segment[7]
        name_end = segment[8]
        self.band_output[band_type].append(
            [k_start, k_end, n_points, name_start, name_end]
        )

    def convert_band_segment_to_string(self, segment):
        k_start = segment[0]
        k_end = segment[1]
        n_points = segment[2]
        name_start = segment[3]
        name_end = segment[4]
        coordinates = f"{k_start[0]} {k_start[1]} {k_start[2]} {k_end[0]} {k_end[1]} {k_end[2]}"
        band_string = coordinates + f" {n_points} {name_start} {name_end}"
        return band_string

    def generate_file_content(self):
        content = []
        content.append("# General settings:")
        settings = self.settings.keys()

        general_settings = [
            "xc",
            "spin",
            "charge",
            "relativistic",
            "spin",
            "charge",
            "relativistic",
            "occupation_type",
            "k_grid",
            "k_offset",
            "k_points_external",
            "generalized_regular_k_grid",
            "hybrid_xc_coeff",
        ]
        present_general_settings = [
            s for s in settings if s in general_settings
        ]
        for s in present_general_settings:
            if s in ["k_grid", "k_offset"]:
                value_string = " ".join(str(val) for val in self.settings[s])
            else:
                value_string = str(self.settings[s])
            content.append(f"{s:24s}{value_string}")
        settings = [s for s in settings if s not in present_general_settings]

        if self.vdw_corrections:
            content.append("")
            content.append("# VdW correction:")
            for s in self.vdw_corrections:
                vdw_settings = []
                for key, value in self.vdw_corrections[s].items():
                    vdw_settings.append(f"{key}={value}")
                if vdw_settings:
                    settings_string = " ".join(vdw_settings)
                    content.append(f"{s:24s}" + settings_string)
                else:
                    content.append(f"{s}")

        sc_acc = re.compile("sc_accuracy.*")
        convergence_settings = sc_acc.findall("\n".join(list(settings)))
        if convergence_settings:
            content.append("")
            content.append("# Convergence criteria:")
            for s in convergence_settings:
                content.append(f"{s:24s}{self.settings[s]}")
        settings = [s for s in settings if s not in convergence_settings]

        # this is not the complete list of mixer settings
        mixer_settings = [
            "adjust_scf",
            "charge_mix_param",
            "mixer",
            "preconditioner",
            "ini_linear_mixing",
            "ini_linear_mix_param",
            "ini_spin_mix_param",
            "n_max_pulay",
            "mixer_threshold",
        ]
        present_mixer_settings = [s for s in settings if s in mixer_settings]
        if present_mixer_settings:
            content.append("")
            content.append("# Mixer settings:")
            for s in present_mixer_settings:
                content.append(f"{s:24s}{self.settings[s]}")
        settings = [s for s in settings if s not in mixer_settings]

        hybrid_settings = ["screening_threshold", "coulomb_threshold"]
        present_hybrid_settings = [s for s in settings if s in hybrid_settings]
        if present_hybrid_settings:
            content.append("")
            content.append("# Hybrid settings:")
            for s in present_hybrid_settings:
                keyword = f"periodic_hf {s}"
                content.append(f"{keyword:32s}{self.settings[s]}")
        settings = [s for s in settings if s not in hybrid_settings]

        if settings:
            content.append("")
            content.append("# Other settings:")
            for s in settings:
                if len(s) >= 24:
                    padding = (len(s) // 4 + 1) * 4
                    line = s.ljust(padding) + self.settings[s]
                    content.append(line)
                else:
                    content.append(f"{s:24s}{self.settings[s]}")

        # deal with output tags
        if (
            self.cubes
            or self.band_output
            or self.dos_output
            or self.additional_output
        ):
            content.append("")
            content.append("# Additional output:")

        if self.additional_output:
            content.append("")
            for key, val in self.additional_output.items():
                if val == "":
                    content.append(f"output {key}")
                else:
                    content.append(f"output {key} {val}")

        if self.dos_output:
            content.append("")
            for key, val in self.dos_output.items():
                content.append(f"output {key} {val}")

        if self.band_output:
            content.append("")
            for band_type in self.band_output:
                for segment in self.band_output[band_type]:
                    band_string = self.convertBandSegmentToString(segment)
                    content.append(f"output {band_type} " + band_string)

        if self.cubes:
            content.append("")
            for cube in self.cubes:
                for line in cube.get_text().split("\n"):
                    content.append(line)

        content.append("")
        for species in self.species_definitions:
            for line in (
                self.species_definitions[species].get_text().split("\n")
            ):
                content.append(line)

        text = "\n".join(content)
        return text

    def save_to_file(self, filename):
        text = self.generate_file_content()

        # Enforce linux file ending, even if running on windows machine by using binary mode
        with open(filename, "w", newline="\n") as f:
            f.write(text)

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

            elif "species" in line.strip():
                # Reached the basis set definitions
                break

            if len(spl) > 0 and line[0] != "#":
                keywords[spl[0]] = " ".join(spl[1:])

        return keywords

    def get_k_grid(self) -> list[int]:
        return self.settings["k_grid"]

    def set_k_grid(self, k_grid):
        self.settings["k_grid"] = k_grid

    def remove_k_grid(self):
        if "k_grid" in self.settings:
            self.settings.pop("k_grid")
        if "k_points_external" in self.settings:
            self.settings.pop("k_points_external")
        if "k_offset" in self.settings:
            self.settings.pop("k_offset")

    def get_species_setting(self, species, param):
        return self.species_definitions[species].settings[param][0]

    def set_species_setting(self, species, param, value):
        self.species_definitions[species].settings[param][0] = value

    def get_species(self) -> list[str]:
        return list(self.species_definitions.keys())

    def add_species(self, species_definition, name=None):
        assert isinstance(
            species_definition, SpeciesDefinition
        ), "Can only add species definition objects"
        if name is None:
            name = species_definition.species
            assert (
                name not in self.species_definitions
            ), "Cannot use a species twice!"
        self.species_definitions[name] = species_definition
        self.species_definitions[name].species = name

    def remove_species(self, species_name):
        if species_name in self.species_definitions:
            del self.species_definitions[species_name]

    def get_basis_functions(
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
        basisfunctions = {}

        if elements is None:
            elements = []
            for e in self.species_definitions:
                elements.append(e)

        for e in elements:
            for b in self.species_definitions[e].basisfunctions:

                if e not in basisfunctions:
                    basisfunctions[e] = []

                if "#" not in b[0]:
                    basisfunctions[e].append(" ".join(b))

        return basisfunctions

    def add_cube_file(self, cube):
        assert isinstance(
            cube, CubeParameters
        ), "Can only add CubeFileSettings object"
        self.cubes.append(cube)

    def check_periodic(self) -> bool:
        """
        Check if the system is periodic.

        """
        return "k_grid" in self.settings.keys()

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
        content = []
        for keyword in args:
            text_new = f"{keyword[0]} {keyword[1]}"
            content.append(text_new)

        self.parse(content)
        self.save_to_file(self.control_in)

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
            raise TypeError(
                "add_cube_cell doesn't support non-periodic structures"
            )

        shortest_side = min(np.sum(cell_matrix, axis=1))
        resolution = shortest_side / 100.0

        cube_x = (
            2
            * int(
                np.ceil(0.5 * np.linalg.norm(cell_matrix[0, :]) / resolution)
            )
            + 1
        )  # Number of cubes along x axis
        x_vector = (
            cell_matrix[0, :] / np.linalg.norm(cell_matrix[0, :]) * resolution
        )
        cube_y = (
            2
            * int(
                np.ceil(0.5 * np.linalg.norm(cell_matrix[1, :]) / resolution)
            )
            + 1
        )
        y_vector = (
            cell_matrix[1, :] / np.linalg.norm(cell_matrix[1, :]) * resolution
        )
        cube_z = (
            2
            * int(
                np.ceil(0.5 * np.linalg.norm(cell_matrix[2, :]) / resolution)
            )
            + 1
        )
        z_vector = (
            cell_matrix[2, :] / np.linalg.norm(cell_matrix[2, :]) * resolution
        )

        print(x_vector)

        # check if previous incomplete cubes have been added
        if len(self.cubes) > 0:
            if not self.cubes[-1].settings:
                self.cubes[-1].origin = np.transpose(
                    cell_matrix @ [0.5, 0.5, 0.5]
                )

                divisions = [cube_x, cube_y, cube_z]
                vectors = np.vstack((x_vector, y_vector, z_vector))
                self.cubes[-1].edges = (divisions, vectors)
        else:
            raise ValueError(
                """Before executing
                             <AimsControl.add_cube_cell_and_save>
                             the cube ouput must be set via
                             <AimsControl.add_keywords_and_save>"""
            )

        self.save_to_file(self.control_in)

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
            if keyword in self.settings:
                self.settings.pop(keyword)

        self.save_to_file(self.control_in)


def _list_to_string(l):
    return " ".join(str(x) for x in l)


class SpeciesDefinition:
    """This class represents a FHIaims species settings (mass, basis set, integration grid, etc)"""

    def __init__(self, text):
        if os.path.isfile(text):
            with open(text) as f:
                text = f.read()
        self.text = text
        self.settings = collections.OrderedDict()

        self.parse(text)

    def parse(self, text):
        text_lines = text.split("\n")
        self.minimal_basis = []
        self.basisfunctions = []
        self.angular_grid_divisions = []

        for ind, l in enumerate(text_lines):
            l = l.strip()
            if len(l) == 0:
                continue  # skip empty lines
            line_is_commented = False

            # Skip all comments, unless they contain ionic, hydro, etc
            if l.startswith("#"):
                l = l[1:].strip()
                if len(l) == 0:
                    continue  # skip lines that contain only '#'
                first_word = l.split()[0]
                if first_word in ["ionic", "hydro", "division", "outer_grid"]:
                    line_is_commented = True
                else:
                    continue

            keyword = l.split()[0]  # obtain keyword
            setting_string = l.split("#")[0]  # remove trailing comments
            settings = setting_string.split()[1:]

            if keyword == "species":
                self.species = settings[0]
            elif keyword in [
                "division",
                "outer_grid",
            ]:  # TODO: check if self.settings['angular_grid'] == 'specified'
                if line_is_commented:
                    self.angular_grid_divisions.append(
                        ["# " + keyword] + settings
                    )
                else:
                    self.angular_grid_divisions.append([keyword] + settings)
            elif keyword in ["ionic", "hydro"]:
                if line_is_commented:
                    self.basisfunctions.append(["# " + keyword] + settings)
                else:
                    self.basisfunctions.append([keyword] + settings)
            elif keyword in ["ion_occ", "valence"]:
                self.minimal_basis.append([keyword] + settings)
            else:
                self.settings[keyword] = settings

    def get_text(self):
        text = "#" * 45 + "\n"
        text += f"species {self.species}\n\n"
        for s in self.settings:
            if s == "angular_grids":
                text += "\n# Angular grid divisions\n"
            text += f"{s} {_list_to_string(self.settings[s])}\n"
            if s == "angular_grids":
                for d in self.angular_grid_divisions:
                    text += _list_to_string(d) + "\n"

        text += "\n# Minimal basis functions\n"
        for b in self.minimal_basis:
            text += _list_to_string(b) + "\n"

        text += "\n# Additional basis functions\n"
        for b in self.basisfunctions:
            text += _list_to_string(b) + "\n"
        return text + "\n"


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

        get_text(): returns cubefile specifications-string for ControlFile class
    """

    def __init__(self, cube: str | None = None, text: str | None = None):
        # super().__init__(cube="cube.cube")

        # Set attrs here rather than `File.__post_init__()` as `Cube.__init__()` uses
        # ASE to parse the data from a cube file, so it's definied in `Cube.__init__()`
        # so `File.__post_init__()` doesn't add these attributes if a cube file
        # extension is detected.
        if cube is not None:
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
                lambda x: "  ".join([f"{k:15.10f}" for k in x]),
            ],
            "edge": [
                lambda x: [int(x[0])] + [float(k) for k in x[1:]],
                lambda x: str(int(x[0]))
                + "  "
                + "  ".join([f"{k:15.10f}" for k in x[1:]]),
            ],
        }

        self._settings = collections.OrderedDict()

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
            raise ValueError(
                "Requires divisions for all three lattice vectors"
            )

        for i in range(3):
            self.settings["edge"][i][0] = divs[i]

    def parse(self, text: str) -> None:
        """
        Parses the text snippet that contains the cube settings.

        Parameters
        ----------
        str
            TODO
        """
        cubelines = []
        for line in text:
            line = line.strip()
            # parse only lines that start with cube and are not comments
            if not line.startswith("#"):
                if line.startswith("cube"):
                    cubelines.append(line)
                elif line.startswith("output"):
                    self.type = " ".join(line.split()[2:])

        # parse cubelines to self.settings
        for line in cubelines:
            line = line.split("#")[0]  # remove comments
            splitline = line.split()
            keyword = splitline[1]  # parse keyword
            values = splitline[2:]  # parse all values
            # check if parsing function exists
            if keyword in self.parsing_functions:
                value = self.parsing_functions[keyword][0](values)
            # reconvert to single string otherwise
            else:
                value = " ".join(values)

            # save all values as list, append to list if key already exists
            if keyword not in self.settings:
                self.settings[keyword] = []

            self.settings[keyword].append(value)

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
        text = ""
        if len(self.type) > 0:
            text += "output cube " + self.type + "\n"
        else:
            warn("No cube type specified", stacklevel=2)
            text += "output cube" + "CUBETYPE" + "\n"

        for key, values in self.settings.items():
            for v in values:
                print(key, v, values)
                text += "cube " + key + " "
                if key in self.parsing_functions:
                    text += self.parsing_functions[key][1](v) + "\n"
                else:
                    text += v + "\n"

        return text
