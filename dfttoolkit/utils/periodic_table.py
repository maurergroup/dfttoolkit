from pathlib import Path
from typing import Any, ClassVar

import yaml

from .file_utils import classproperty


class Element:
    """
    Hold and access data for individual elements.

    ...

    Attributes
    ----------
    name: str
        Elemental name
    symbol: str
        Chemical symbol
    atomic_number:
        Atomic number of the element
    atomic_mass:
        Average atomic mass of an element
    """

    def __init__(self, name: str, **properties: Any):
        # Ensure these attributes are set for all elements
        self.name = name
        self.symbol = properties["symbol"]
        self.atomic_number = properties["number"]
        self.atomic_mass = properties["atomic_mass"]

        # Dynamically assign other attributes based on what is available
        for key, val in properties.items():
            setattr(self, key, val)

    def __repr__(self):
        return f"Element({vars(self)})"

    def __getitem__(self, key: str):
        return getattr(self, key)


class PeriodicTable:
    """
    Hold and access data for elements in the periodic table.

    ...

    Attributes
    ----------
    elements: dict
        Data on physical properties of elements
    order: List[str]
        Element names ordered by their atomic number
    """

    # Parse the yaml file
    with open(Path(__file__).parent / "periodic_table.yaml") as pt:
        _raw_table = yaml.safe_load(pt)

    # Save
    _order = _raw_table["order"]
    _elements: ClassVar = {}
    for name in _order:
        del _raw_table[name]["name"]
        _elements[_raw_table[name]["symbol"]] = Element(name, **_raw_table[name])

    def __new__(cls):
        """Prevent instantiation of this class."""
        raise TypeError("This class cannot be instantiated.")

    @classproperty
    def elements(cls) -> dict[str, Element]:  # noqa: N805
        """
        Elements as a dictionary of their symbols mapped to element objects.

        Returns
        -------
        dict[str, Element]
            Dictionary of elements.
        """
        return cls._elements

    @classproperty
    def element_names(cls) -> list[str]:  # noqa: N805
        """
        Element names ordered by atomic number.

        Returns
        -------
        list[str]
            List of element names.
        """
        return [e.name for e in cls._elements.values()]

    @classproperty
    def element_symbols(cls) -> list[str]:  # noqa: N805
        """
        Element symbols ordered by atomic number.

        Returns
        -------
        list[str]
            List of element symbols.
        """
        return list(cls._elements.keys())

    @classmethod
    def get_element(cls, symbol: str) -> Element:
        """
        Retrieve an element as an instance of Element.

        Parameters
        ----------
        symbol : str
            Chemical symbol of the element.

        Returns
        -------
        Element
            Instance of Element.
        """
        return cls._elements[symbol]

    @classmethod
    def get_symbol(cls, atomic_number: int) -> str:
        """
        Get the chemical symbol of an element.

        Parameters
        ----------
        atomic_number : int
            Atomic number of the element.

        Returns
        -------
        str
            Chemical symbol.
        """
        return cls.element_symbols[atomic_number - 1]

    @classmethod
    def get_name(cls, symbol: str) -> str:
        """
        Get the full name of an element.

        Parameters
        ----------
        symbol : str
            Chemical symbol of the element.

        Returns
        -------
        str
            Full name.
        """
        return cls.get_element(symbol).name

    @classmethod
    def get_atomic_number(cls, symbol: str) -> int:
        """
        Get the atomic number of an element.

        Parameters
        ----------
        symbol : str
            Chemical symbol of the element

        Returns
        -------
        int
            Atomic number.
        """
        return cls.get_element(symbol).atomic_number

    @classmethod
    def get_atomic_mass(cls, symbol: str) -> float:
        """
        Get the atomic mass of an element.

        Parameters
        ----------
        symbol : str
            chemical symbol of the element

        Returns
        -------
        float
            Atomic mass.
        """
        return cls.get_element(symbol).atomic_mass

    @classmethod
    def get_covalent_radius(cls, symbol: str) -> float:
        """
        Get the covalent radius of an element.

        Parameters
        ----------
        symbol : str
            chemical symbol of the element

        Returns
        -------
        float
            Covalent radius in atomic units.
        """
        with open(Path(__file__).parent / "covalent_radii.yaml") as cr:
            data = yaml.safe_load(cr)

        return data[symbol]

    @classmethod
    def get_species_colours(cls, symbol: str) -> tuple[float, float, float]:
        """
        Get the JMol colour of an element.

        Parameters
        ----------
        symbol : str
            chemical symbol of the element

        Returns
        -------
        Tuple[float, float, float]
            Covalent radius in atomic units.
        """
        with open(Path(__file__).parent / "elemental_colourmaps.yaml") as ec:
            data = yaml.safe_load(ec)

        return data[symbol]
