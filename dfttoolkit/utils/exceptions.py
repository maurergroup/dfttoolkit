class ItemNotFoundError(Exception):
    def __init__(self, item):
        super().__init__(f"'{item[0]}': '{item[1]}' item not found in dictionary.")


class UnsupportedFileError(Exception):
    def __init__(self, file_type, supported_files):
        super().__init__(
            f"{file_type} is not a supported file format. Ensure the file is one "
            f"of {supported_files}."
        )
