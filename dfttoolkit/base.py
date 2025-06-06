import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import FunctionType, MethodType

from .utils.exceptions import UnsupportedFileError


@dataclass
class File:
    """
    Hold information about a file.

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
    lines : list[str] | bytes
        the contents of the file stored as either a list of strings for text files or
        bytes for binary files
    binary : bool
        whether the file is stored as binary or not
    """

    path: str
    _path: Path = field(init=False)
    _format: str
    _name: str = field(init=False)
    _extension: str = field(init=False)
    _binary: bool = field(init=False)
    lines: list[str] = field(init=False)
    data: bytes = field(init=False)

    def __post_init__(self):
        self._path = Path(self.path)

        if not self._path.is_file():
            raise FileNotFoundError("Path not found.")

        # Do not run init code for DummyParser in test_base.TestParser
        if "arbitrary_format" in self._format:
            return

        self._name = self._path.name
        self._extension = self._path.suffix

        self.__dataclass_fields__["_extension"].metadata = {"type": self._format}

        if self._extension == ".csc":
            with open(self.path, "rb") as f:
                self.data = f.read()
                self.lines = []
                self._binary = True

        elif self._extension != ".cube":
            with open(self.path) as f:
                self.lines = f.readlines()
                self.data = b""
                self._binary = False

    def __str__(self) -> str:
        if len(self.lines) == 0:
            raise OSError("Is a binary file")

        return "".join(self.lines)


class Parser(File, ABC):
    """Handle all file parsing."""

    def __init__(self, supported_files: dict[str, str], **kwargs: str):
        # Check that only one supported file was provided
        if not kwargs:
            msg = (
                f"Ensure one of {list(supported_files.keys())} is specified as a kwarg."
            )
            raise TypeError(msg)

        provided_keys = set(kwargs.keys())

        if len(provided_keys) != 1:
            msg = f"Ensure only one of {list(supported_files.keys())} is specified."
            raise TypeError(msg)

        # Check if the provided file is a supported type
        key = next(iter(provided_keys))

        if key not in supported_files:
            raise UnsupportedFileError(key, supported_files.keys())

        # Check if the provided file is a valid file type
        if supported_files.get(key) != Path(kwargs[key]).suffix:
            msg = f"{kwargs[key]} is not a valid {key} file"
            raise KeyError(msg)

        super().__init__(*reversed(next(iter(kwargs.items()))))

    @property
    @abstractmethod
    def _supported_files(self) -> dict[str, str]:
        """Currently supported output file types and extensions."""
        ...

    def __init_subclass__(cls, **kwargs: str):
        super().__init_subclass__(**kwargs)

        # Get the class __init__
        init_obj = cls.__dict__.get("__init__")

        if init_obj is None:
            msg = f"{cls.__name__} must implement __init__"
            raise TypeError(msg)

        if not isinstance(init_obj, FunctionType | MethodType):
            msg = f"{cls.__name__}.__init__ is not a function or method"
            raise TypeError(msg)

        src = inspect.getsource(init_obj)

        if "_check_binary" not in src:
            msg = f"{cls.__name__} must implement _check_binary method"
            raise TypeError(msg)

    def _check_binary(self, binary: bool) -> None:
        """
        Check if the file is supposed to be a binary of text format.

        Parameters
        ----------
        binary : bool
            Whether the file is expected to be a binary or not
        """
        if self._binary is not binary:
            expected_str = "binary" if binary else "text"
            msg = f"{self._name} should be {expected_str} format"
            raise ValueError(msg)
