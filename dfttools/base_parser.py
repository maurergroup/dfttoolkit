import os.path


class BaseParser:
    """
    Generic routines for all parsers.

    ...

    Attributes
    ----------
    file_paths : dict
        the paths to the parsed files
    file_contents : dict
        the contents of parsed files
    """

    def __init__(self, supported_files, **kwargs):
        self.file_paths = {}
        self.file_contents = {}

        for kwarg in kwargs:
            # Check if the file type is supported
            if kwarg not in supported_files:
                raise ValueError(f"{kwarg} is not a supported file.")

            # Check if the file path exists
            if not os.path.isfile(kwargs[kwarg]):
                raise ValueError(f"{kwargs[kwarg]} does not exist.")

            # Store the file paths
            self.file_paths[kwarg] = kwargs[kwarg]

            # Get the contents of the files
            with open(kwargs[kwarg], "r") as f:
                self.file_contents[kwarg] = f.readlines()