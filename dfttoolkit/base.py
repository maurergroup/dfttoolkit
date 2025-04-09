import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dfttoolkit.utils.exceptions import UnsupportedFileError


@dataclass
class File:
    """
    Hold information about a file

    ...

    Attributes
    ----------
    path : str
        the path to the file
    format : str
        the type of file it is (eg. aims.out, control.in, etc.)
    name : str
        the name of the file
    extension : str
        the extension of the file
    lines : Union[List[str], bytes]
        the contents of the file stored as either a list of strings for text files or
        bytes for binary files
    binary : bool
        whether the file is stored as binary or not
    """

    path: str
    _format: str
    _name: str = field(init=False)
    _extension: str = field(init=False)
    _binary: bool = field(init=False)
    lines: List[str] = field(init=False)
    data: bytes = field(init=False)

    def __post_init__(self):
        # Do not run init code for DummyParser in test_base.TestParser
        if "arbitrary_format" in self._format:
            return

        self._path = Path(self.path)
        if not self._path.is_file():
            raise FileNotFoundError(f"{self.path} path not found.")

        self._name = self._path.name
        self._extension = self._path.suffix

        self.__dataclass_fields__["_extension"].metadata = {"type": self._format}

        if self._extension == ".csc":
            with open(self.path, "rb") as f:
                self.data = f.read()
                self.lines = []
                self._binary = True

        else:
            with open(self.path, "r") as f:
                self.lines = f.readlines()
                self.data = b""
                self._binary = False

    def __str__(self) -> str:
        if len(self.lines) == 0:
            raise OSError(f"{self._name} is a binary file")
        else:
            return "".join(self.lines)


class Parser(File, ABC):
    """Handle all file parsing"""

    def __init__(self, supported_files, **kwargs):
        # Check that only one supported file was provided
        if not kwargs:
            raise TypeError(
                f"Ensure one of {list(supported_files.keys())} is specified as a kwarg."
            )

        provided_keys = set(kwargs.keys())

        if len(provided_keys) != 1:
            raise TypeError(
                f"Ensure only one of {list(supported_files.keys())} is specified."
            )

        # Check if the provided file is a supported type
        key = next(iter(provided_keys))

        if key not in supported_files:
            raise UnsupportedFileError(provided_keys, list(supported_files.keys()))

        # Check if the provided file is a valid file type
        if supported_files.get(key) not in kwargs[key]:
            raise KeyError(f"{kwargs[key]} is not a valid {key} file")

        super().__init__(*reversed(next(iter(kwargs.items()))))

    @property
    @abstractmethod
    def _supported_files(self) -> dict:
        """Currently supported output file types and extensions"""
        ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Get the class __init__
        cls_init = cls.__dict__.get("__init__")

        if cls_init is None:
            raise TypeError(f"{cls.__name__} must implement __init__")

        src = inspect.getsource(cls_init)

        # Check if the required methods are implemented
        # required_methods = ["_check_output_file_extension, _check_binary"]
        # if not all(method in src for method in required_methods):
        #     raise TypeError(
        #         f"{cls.__name__} must implement {*required_methods,} methods"
        #     )

        if "_check_binary" not in src:
            raise TypeError(f"{cls.__name__} must implement _check_binary method")

    def _check_binary(self, binary: bool) -> None:
        """
        Check if the file is supposed to be a binary of text format

        Parameters
        ----------
        binary : bool
            Whether the file is expected to be a binary or not
        """

        if self._binary is not binary:
            expected_str = "binary" if binary else "text"
            raise ValueError(f"{self._name} should be {expected_str} format")
