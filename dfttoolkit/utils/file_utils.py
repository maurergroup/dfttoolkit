from pathlib import Path
from typing import Union

from click import edit


class ClassPropertyDescriptor(object):
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)

        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)

        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func

        return self


def classproperty(func):
    """Combine class methods and properties to make a classproperty"""
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


def aims_bin_path_prompt(change_bin: Union[bool, str], save_dir) -> str:
    """
    Prompt the user to enter the path to the FHI-aims binary, if not already found in
    .aims_bin_loc.txt

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

    def write_bin():
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
        with open(f"{save_dir}/.aims_bin_loc.txt", "r") as f:
            binary = f.readlines()[0]

        # Check if the binary path exists and is a file
        if not Path(binary).is_file():
            binary = write_bin()

    return binary
