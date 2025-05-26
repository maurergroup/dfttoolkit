from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from click import edit


class MultiDict(MutableMapping):
    """
    Dictionary that can assign 'multiple values' to a single key.

    Primitive implementation that works by having each value as a list, and appending
    new values to the list
    """

    def __init__(self, *args: tuple[str, Any]):
        self._dict = {}

        for key, val in args:
            if key in self._dict:
                self._dict[key].append(val)

            else:
                self._dict[key] = val

    def __setitem__(self, key: Any, val: Any):
        if key in self._dict:
            self._dict[key].append(val)
        else:
            self._dict[key] = [val]

    def __repr__(self):
        return f"{self.__class__.__name__}({self._dict})"

    def __str__(self):
        return str(self._dict)

    def __getitem__(self, key: Any):
        return self._dict[key]

    def __delitem__(self, key: Any):
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict.keys())

    def reversed_items(self) -> Iterator[tuple[str, Any]]:
        """Yield (key, value) pairs in reverse key order and reversed values."""
        for key in reversed(list(self._dict.keys())):
            for val in reversed(self._dict[key]):
                yield key, val


Param = ParamSpec("Param")
RetType = TypeVar("RetType")
T = TypeVar("T")
C = TypeVar("C", bound=type)


# class _ClassPropertyDescriptor(Generic[T]):
class _ClassPropertyDescriptor:
    """
    Descriptor for creating class-level properties.

    This class is not intended to be used directly. It is returned by the
    `classproperty` decorator to enable properties that behave like `@property`, but
    can be accessed directly on the class rather than an instance.

    Parameters
    ----------
    func : TODO
        A function that takes the class as its only argument and returns the value of
        the property.
    """

    # def __init__(self, func:Callable[[type[C]], T]):
    def __init__(self, func):  # noqa: ANN001
        self.func = func

    # def __get__(self, obj: Optional[Any], cls:type[C]) :
    def __get__(self, obj, cls):  # noqa: ANN001
        return self.func(cls)


# def classproperty(func: Callable[[type[C], T]]) -> _ClassPropertyDescriptor[T]:
def classproperty(func):  # noqa: ANN001, ANN201
    """
    Use as a decorator to define a class-level property.

    This works like the built-in `@property` decorator, but allows the
    property to be accessed directly on the class without requiring an
    instance. Similar to `@classmethod`, but used to expose computed
    attributes as properties.

    Parameters
    ----------
    func : TODO
        A method that takes the class as its only argument and returns the value
        of the property.

    Returns
    -------
    TODO
        A descriptor that implements the class-level property.
    """
    return _ClassPropertyDescriptor(func)


def aims_bin_path_prompt(change_bin: bool | str, save_dir: Path) -> str:
    """
    Prompt the user to enter the path to the FHI-aims binary.

    If it is found in .aims_bin_loc.txt, the path will be read from there, unless
    change_bin is True, in which case the user will be prompted to enter the path again.

    Parameters
    ----------
    change_bin : Union[bool, str]
        whether the user wants to change the binary path. If str == "change_bin", the
        user will be prompted to enter the path to the binary again.
    save_dir : str
        the directory to save or look for the .aims_bin_loc.txt file

    Returns
    -------
    binary : str
        path to the location of the FHI-aims binary
    """
    marker = (
        "\n# Enter the path to the FHI-aims binary above this line\n"
        "# Ensure that the full absolute path is provided"
    )

    def write_bin() -> str:
        binary = edit(marker)
        binary = str(binary).split()[0]

        if binary is not None:
            if Path(binary).is_file():
                with open(f"{save_dir}/.aims_bin_loc.txt", "w+") as f:
                    f.write(binary)

            else:
                raise FileNotFoundError(
                    "the path to the FHI-aims binary does not exist"
                )

        else:
            raise FileNotFoundError(
                "the path to the FHI-aims binary could not be found"
            )

        return binary

    if (
        not Path(f"{save_dir}/.aims_bin_loc.txt").is_file()
        or change_bin == "change_bin"
    ):
        binary = write_bin()

    else:
        # Parse the binary path from .aims_bin_loc.txt
        with open(f"{save_dir}/.aims_bin_loc.txt") as f:
            binary = f.readlines()[0]

        # Check if the binary path exists and is a file
        if not Path(binary).is_file():
            binary = write_bin()

    return binary
