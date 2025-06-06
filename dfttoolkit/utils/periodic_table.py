from pathlib import Path
import yaml
from typing import Union, Any, List
from .file_utils import classproperty


class Element:
    """
    Hold and access data for individual elements

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

    def __getitem__(self, key):
        return getattr(self, key)


class PeriodicTable:
    """
    Create a periodic table object

    Returns
    -------
    dict
        a dictionary representing the periodic table
    """

    with open(Path(__file__).parent / "periodic_table.yaml", "r") as pt:
        _periodic_table = yaml.safe_load(pt)

    def __new__(cls):
        raise TypeError("This class cannot be instantiated.")

    @classproperty
    def elements(cls) -> dict:
        # Save
        _order = cls._periodic_table["order"]
        cls._elements = {}
        for name in _order:
            del cls._periodic_table[name]["name"]
            # _elements[s] = Element(name, **_raw_table[name])
            cls._elements[cls._periodic_table[name]["symbol"]] = Element(
                name, **cls._periodic_table[name]
            )

        return cls._elements

    @classproperty
    def element_names(cls) -> List[str]:
        return [e.name for e in cls.elements.values()]

    @classproperty
    def element_symbols(cls) -> List[str]:
        return list(cls.elements.keys())

    @classmethod
    def get_element(cls, symbol: str) -> Element:
        """
        Retrieve an element as an instance of Element

        Parameters
        ----------
        symbol : str
            Chemical symbol of the element.

        Returns
        -------
        Element
            Instance of Element.
        """

        return cls.elements[symbol]

    @classmethod
    def get_element_dict(cls, element: Union[str, int]) -> dict:

        element_dict = None

        if element in cls._periodic_table:
            element_dict = cls._periodic_table[element]
        else:
            for key in cls._periodic_table["order"]:
                element_0 = cls._periodic_table[key]

                if (
                    element == element_0["name"]
                    or element == element_0["number"]
                    or element == element_0["symbol"]
                ):
                    element_dict = element_0
                    break

        if element_dict is None:
            raise ValueError(
                f'Could not find element "{element}" in periodic table!'
            )

        return element_dict

    @classmethod
    def get_name(cls, element: Union[str, int]) -> str:
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

        return cls.get_element_dict(element)["name"]

    @classmethod
    def get_atomic_number(cls, element: Union[str, int]) -> int:
        """
        Returns the atomic number if given the species as a string.

        Parameters
        ----------
        species : str or int
            Name or chemical sysmbol of the atomic species.

        Returns
        -------
        int
            atomic number.

        """
        return cls.get_element_dict(element)["number"]

    @classmethod
    def get_atomic_mass(cls, element: Union[str, int]) -> float:
        """
        Returns the atomic mass if given the species as a string.

        Parameters
        ----------
        species : str or int
            Name or chemical sysmbol of the atomic species.

        Returns
        -------
        float
            atomic mass in atomic units.

        """
        return cls.get_element_dict(element)["atomic_mass"]

    @classmethod
    def get_chemical_symbol(cls, element: Union[str, int]) -> float:
        """
        Returns the chemical symbol if given the species as an atomic number.

        Parameters
        ----------
        species : str or int
            Name or chemical sysmbol of the atomic species.

        Returns
        -------
        float
            atomic mass in atomic units.

        """
        return cls.get_element_dict(element)["symbol"]

    @classmethod
    def get_covalent_radius(cls, element: Union[str, int]) -> float:
        """
        Returns the chemical symbol if given the species as an atomic number.

        Parameters
        ----------
        species : str or int
            Name or chemical sysmbol of the atomic species.

        Returns
        -------
        float
            Covalent radius in atomic units.

        """
        with open(Path(__file__).parent / "covalent_radii.yaml", "r") as cr:
            data = yaml.safe_load(cr)

        return data[cls.get_element_dict(element)["symbol"]]

    @classmethod
    def get_species_colors(cls, element: Union[str, int]) -> float:
        """
        Returns the chemical symbol if given the species as an atomic number.

        Parameters
        ----------
        species : str or int
            Name or chemical sysmbol of the atomic species.

        Returns
        -------
        float
            Covalent radius in atomic units.

        """
        with open(
            Path(__file__).parent / "elemental_colourmaps.yaml", "r"
        ) as ec:
            data = yaml.safe_load(ec)

        return data[cls.get_element_dict(element)["symbol"]]
