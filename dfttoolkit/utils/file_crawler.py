import os
from os.path import join


def find_all_aims_output_files(
    directory: str,
    include_restart: bool = True,
    allow_all_out_files: bool = False,
    allow_multiple_files: bool = False,
) -> list[str]:
    """
    Recursively searches for AIMS output files and returns their full filenames.

    Parameters
    ----------
    directory : str
        TODO
    include_restart : bool, default=True
        TODO
    allow_all_out_files : bool, default=False
        TODO
    allow_multiple_files : bool, default=False
        TODO

    Returns
    -------
    list[str]
        TODO
    """
    aims_fnames = []

    for root, _directories, _files in os.walk(directory):
        fname_list = find_aims_output_file(
            root, allow_all_out_files, allow_multiple_files
        )

        if len(fname_list) > 0:
            for fname in fname_list:
                if include_restart:
                    aims_fnames.append(join(root, fname))  # noqa: PTH118
                else:
                    root_name = os.path.basename(os.path.normpath(root))  # noqa: PTH119
                    is_restart_folder = len(root_name) == len(
                        "restartXX"
                    ) and root_name.startswith("restart")
                    if not is_restart_folder:
                        aims_fnames.append(join(root, fname))  # noqa: PTH118

    return aims_fnames


def find_aims_output_file(
    calc_dir: str, allow_all_out_files: bool = False, allow_multiple_files: bool = False
) -> list[str]:
    """
    Search a directory for output files.

    Parameters
    ----------
    calc_dir : str
        Directory to search for output files.
    allow_all_out_files : bool, default=False
        TODO
    allow_multiple_files : bool, default=False
        TODO

    Returns
    -------
    list[str]
        TODO
    """
    return find_file(
        calc_dir,
        allow_all_out_files=allow_all_out_files,
        allow_multiple_files=allow_multiple_files,
        list_of_filenames=[
            "aims.out",
            "out.aims",
            "output",
            "output.aims",
            "aims.output",
        ],
    )


def find_vasp_output_file(calc_dir: str) -> list:
    """
    Search a directory for VASP output files.

    Parameters
    ----------
    calc_dir : str
        Directory to search for output files

    Returns
    -------
    list
        List of found output files
    """
    return find_file(calc_dir, allow_all_out_files=False, list_of_filenames=["outcar"])


def find_file(
    calc_dir: str,
    allow_all_out_files: bool = False,
    allow_multiple_files: bool = False,
    list_of_filenames: list[str] | None = None,
) -> list[str]:
    """
    Search a directory for output files.

    Parameters
    ----------
    calc_dir : str
        Directory to search for output files.
    allow_all_out_files : bool, default=False
        TODO
    allow_multiple_files : bool, default=False
        TODO
    list_of_filenames : list[str] | None, default=None
        TODO

    Returns
    -------
    list[str]
        TODO
    """
    if list_of_filenames is None:
        list_of_filenames = []

    allfiles = [f for f in os.listdir(calc_dir) if os.path.isfile(join(calc_dir, f))]  # noqa: PTH113, PTH118, PTH208
    filename = [f for f in allfiles if f.lower() in list_of_filenames]

    if allow_all_out_files and len(filename) == 0:
        filename = [f for f in allfiles if f.endswith(".out")]

    if len(filename) > 1 and not allow_multiple_files:
        msg = f"Multiple output files found: {calc_dir}, {filename}"
        raise ValueError(msg)

    return filename
