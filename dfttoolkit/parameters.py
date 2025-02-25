import inspect
from typing import Dict, List, Literal, Union

from dfttoolkit.base import Parser


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
        super().__init__(self._supported_files.keys(), **kwargs)

        match self._format:
            case "control_in":
                self._check_output_file_extension("control_in")
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

    def add_keywords(self, **kwargs: dict) -> None: ...

    # """
    # Add keywords to the control.in file.

    # Parameters
    # ----------
    # **kwargs : dict
    #     Keywords to be added to the control.in file.
    # """

    def remove_keywords(
        self, *args: str, output: Literal["overwrite", "print", "return"] = "return"
    ) -> Union[None, List[str]]:
        """
        Remove keywords from the control.in file.

        Parameters
        ----------
        *args : str
            Keywords to be removed from the control.in file.
        output : Literal["overwrite", "print", "return"], default="overwrite"
            Overwrite the original file, print the modified file to STDOUT, or return
            the modified file as a list of '\\n' separated strings.

        Returns
        -------
        Union[None, List[str]]
            If output is "return", the modified file is returned as a list of '\\n'
            separated strings.
        """

        for keyword in args:
            for i, line in enumerate(self.lines):
                if keyword in line:
                    self.lines.pop(i)

        match output:
            case "overwrite":
                with open(self.path, "w") as f:
                    f.writelines(self.lines)

            case "print":
                print(*self.lines, sep="")

            case "return":
                return self.lines

    def get_keywords(self) -> dict:
        """
        Get the keywords from the control.in file.

        Returns
        -------
        dict
            A dictionary of the keywords in the control.in file.
        """

        keywords = {}

        for line in self.lines:
            spl = line.split()

            if "#" * 80 in line:
                break

            if len(spl) > 0 and spl[0] != "#":
                if len(spl) == 1:
                    keywords[spl[0]] = None
                else:
                    keywords[spl[0]] = " ".join(spl[1:])

        return keywords

    def get_species(self) -> List[str]:
        """
        Get the species from a control.in file.

        Returns
        -------
        List[str]
            A list of the species in the control.in file.
        """

        species = []
        for line in self.lines:
            if "species" in line:
                species.append(line.split()[1])

        return species

    def get_default_basis_funcs(
        self, elements: Union[List[str], None] = None
    ) -> Dict[str, str]:
        """
        Get the basis functions from the control.in file.

        Parameters
        ----------
        elements : List[str], optional, default=None
            The elements to parse the basis functions for.

        Returns
        -------
        Dict[str, str]
            A dictionary of the basis functions for the specified elements.
        """

        basis_funcs = {}

        for i, line_1 in enumerate(self.lines):
            spl_1 = line_1.split()
            if "species" in spl_1[0]:
                species = spl_1[1]

                if elements is not None and species not in elements:
                    continue

                for line_2 in self.lines[i:]:
                    spl = line_2.split()
                    if "species" in spl[0]:
                        break

                    if "#" in spl[0]:
                        continue

                    if "hydro" in line_2:
                        basis_funcs[species] = line_2.strip()

        return basis_funcs

    def check_periodic(self): ...
