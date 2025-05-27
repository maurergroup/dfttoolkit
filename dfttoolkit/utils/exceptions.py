from collections.abc import Collection
from typing import Any


class ItemNotFoundError(Exception):
    """Exception raised when an item is not found in a dictionary."""

    def __init__(self, item: Any):
        super().__init__(f"'{item[0]}': '{item[1]}' item not found in dictionary.")


class UnsupportedFileError(Exception):
    """
    Exception raised when a file format is not supported.

    Parameters
    ----------
    file_type : str
        The type of file that is not supported.
    supported_files : Collection[str]
        A collection of supported file formats.
    """

    def __init__(self, file_type: str, supported_files: Collection[str]):
        super().__init__(
            f"{file_type} is not a supported file format. Ensure the file is one "
            f"of {supported_files}."
        )
