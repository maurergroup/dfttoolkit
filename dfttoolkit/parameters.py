from typing import Dict, List, Union
from warnings import warn

from dfttoolkit.base import Parser
from dfttoolkit.utils.periodic_table import PeriodicTable


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
    def get_keywords(self) -> Dict[str, str]:
        """
        Get the keywords from the control.in file.

        Returns
        -------
        dict
            A dictionary of the keywords in the control.in file.
        """

        keywords = {}

        for line in self.lines:
            if "#" * 80 in line:
                break

            spl = line.split()

            if len(spl) > 0 and spl[0] != "#":
                keywords[spl[0]] = " ".join(spl[1:])

        return keywords

    def add_keywords(self, **kwargs: str) -> None:
        """
        Add keywords to the control.in file.

        Note that files written by ASE or in a format where the keywords are at the top
        of the file followed by the basis sets are the only formats that are supported
        by this function.

        Parameters
        ----------
        **kwargs : dict
            Keywords to be added to the control.in file.
        """

        aims_keywords = {}
        pop_eyes = []

        # Get current keywords
        for i, line in enumerate(self.lines):
            if "#" * 80 in line:
                # Reached the basis set definitions
                break

            spl = line.split()

            if len(spl) > 0 and spl[0] != "#":
                aims_keywords[spl[0]] = " ".join(spl[1:])
                pop_eyes.append(i)

        # Check to make sure basis sets were found
        if i + 1 == len(self.lines):
            raise IndexError(f"Unable to find species defaults in {self.path}")

        # Remove the lines with the keywords
        n_pops = 0
        for i in pop_eyes:
            self.lines.pop(i - n_pops)
            n_pops += 1

        # Update with new keywords
        aims_keywords.update(kwargs)

        # Write the new keywords
        for i, line in enumerate(self.lines):
            if "#" * 80 in line:
                # Reached basis set definitions
                # Add new keywords here
                for key, val in reversed(aims_keywords.items()):
                    self.lines.insert(i, f"{key:<34} {val}\n")

                break

        # Write the file
        with open(self.path, "w") as f:
            f.writelines(self.lines)

    def remove_keywords(self, *args: str) -> None:
        """
        Remove keywords from the control.in file.

        Parameters
        ----------
        *args : str
            Keywords to be removed from the control.in file.
        output : Literal["overwrite", "print", "return"], default="overwrite"
            Overwrite the original file, print the modified file to STDOUT, or return
            the modified file as a list of '\\n' separated strings.
        """

        for keyword in args:
            for i, line in enumerate(self.lines):
                if keyword in line:
                    self.lines.pop(i)

        with open(self.path, "w") as f:
            f.writelines(self.lines)

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
            if "species" == line.split()[0]:
                species.append(line.split()[1])

        return species

    def get_default_basis_funcs(
        self, elements: Union[List[str], None] = None
    ) -> Dict[str, str]:
        """
        Get the basis functions

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
            warn("Could not find all requested elements in control.in")

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

    def add_initial_charge(self, target_atom: str, charge: float = 0.1) -> None:
        """
        Add an initial charge to an atom
        """

        for i, line in enumerate(self.lines):
            # Add initial charge in the aims format
            if len(line) > 0 and line.split()[-1] == f"{target_atom}1":
                self.lines.insert(i + 1, f"    initial_charge         {charge}\n")

        with open(self.path, "w") as f:
            f.writelines(self.lines)

    def check_periodic(self): ...
