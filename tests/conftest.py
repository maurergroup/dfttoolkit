import os
import subprocess

import pytest
import yaml
from dfttoolkit.utils.file_utils import MultiDict, aims_bin_path_prompt


def pytest_addoption(parser):
    """Add custom command line options to the pytest command."""

    parser.addoption(
        "--run-aims",
        nargs="?",
        const=True,
        default=False,
        choices=[None, "change_bin"],
        help="Optionally re-calculate the FHI-aims output files with a binary specified"
        " by the user. The first time this is run, the user will be prompted to enter"
        " the path to the FHI-aims binary. If the user wants to change the path in"
        " subsequent runs, they can use the 'change_bin' option, which will"
        " automatically call the binary path prompt again.",
    )


def multidict_constructor(loader, node):
    """PyYaml constructor to read MultiDict objects"""
    return MultiDict(*loader.construct_sequence(node))


# Enable PyYaml to read MultiDict objects
yaml.FullLoader.add_constructor("!MultiDict", multidict_constructor)


@pytest.fixture(scope="session")
def run_aims(request):
    yield request.config.getoption("--run-aims")


@pytest.fixture(scope="session")
def aims_calc_dir(run_aims):
    """
    Run FHI-aims calculations using a custom binary if specified by --run-aims.

    If the calculation has already been run (ie. if the directory
    `custom_bin_aims_calcs` exists), the calculations will not be run again, unless the
    user specifies `change_bin` as an option to --run-aims.
    """

    # Check if the directory already exists
    if os.path.isdir("custom_bin_aims_calcs") and run_aims != "change_bin":
        return "custom_bin_aims_calcs"
    elif run_aims is not False:
        cwd = os.path.dirname(os.path.realpath(__file__))
        binary = aims_bin_path_prompt(run_aims, cwd)
        subprocess.run(["bash", f"{cwd}/run_aims.sh", binary, str(run_aims)])
        yield "custom_bin_aims_calcs"
    else:
        yield "default_aims_calcs"


@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory):
    """Temporary directory for all tests to write files to"""

    d = tmp_path_factory.mktemp("tmp")
    yield d


@pytest.fixture(scope="session")
def ref_data():
    cwd = os.path.dirname(os.path.realpath(__file__))

    with open(f"{cwd}/test_references.yaml", "r") as references:
        # FullLoader necessary for parsing tuples
        yield yaml.load(references, Loader=yaml.FullLoader)


@pytest.fixture(scope="session")
def cwd():
    yield os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="session")
def default_calc_dir(cwd):
    yield f"{cwd}/fixtures/default_aims_calcs/1/"
