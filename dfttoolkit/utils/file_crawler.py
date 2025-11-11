import os
from pathlib import Path

from dfttoolkit.output import AimsOutput


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
            root_path = Path(root)
            for fname in fname_list:
                if include_restart:
                    aims_fnames.append(root / fname)
                else:
                    root_name = root_path.name
                    is_restart_folder = len(root_name) == len(
                        "restartXX"
                    ) and root_name.startswith("restart")
                    if not is_restart_folder:
                        aims_fnames.append(root / fname)

    return aims_fnames


def find_aims_output_file(
    calc_dir: str,
    allow_all_out_files: bool = False,
    allow_multiple_files: bool = False,
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

    calc_dir_path = Path(calc_dir)
    allfiles = [f.name for f in calc_dir_path.iterdir() if f.is_file()]
    filename = [f for f in allfiles if f.lower() in list_of_filenames]

    if allow_all_out_files and len(filename) == 0:
        filename = [f for f in allfiles if f.endswith(".out")]

    if len(filename) > 1 and not allow_multiple_files:
        msg = f"Multiple output files found: {calc_dir}, {filename}"
        raise ValueError(msg)

    return filename


def find_all_aims_calculations_and_status(startpath: str) -> dict[str, dict]:
    """Find all AIMS calculations and their calculation status."""
    calculations_results = {}

    for root, _, files in os.walk(startpath):
        if root == startpath:
            continue

        root_path = Path(root)
        path_from_start = str(root_path.parent)
        control_in_folder = "control.in" in files
        geometry_in_folder = "geometry.in" in files

        output_file_name = find_aims_output_file(root)
        output_in_folder = bool(output_file_name)
        folder_name = root_path.name

        if output_in_folder:
            values_dict = {}
            # catch if found file is not an actual AIMS output file?
            aims_out = AimsOutput(str(root / output_file_name[0]))
            values_dict["started"] = True
            values_dict["finished"] = aims_out.check_exit_normal()
            values_dict["path"] = path_from_start
            calculations_results[folder_name] = values_dict

        if (not output_in_folder) and geometry_in_folder and control_in_folder:
            calculations_results[folder_name] = {
                "started": False,
                "finished": False,
                "idle_time": None,
                "path": path_from_start,
            }

    return calculations_results
