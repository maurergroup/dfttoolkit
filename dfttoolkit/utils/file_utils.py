from collections.abc import Callable, Iterator, MutableMapping
from pathlib import Path
from types import FunctionType
from typing import Any, Optional

from click import edit


class MultiDict(MutableMapping):
    """
    Dictionary that can assign 'multiple values' to a single key.

    Very basic implementation that works by having each value as a list, and appending
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
