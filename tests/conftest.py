import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml
from _pytest.fixtures import FixtureFunctionMarker

from dfttoolkit.utils.file_utils import MultiDict, aims_bin_path_prompt


def pytest_addoption(parser) -> None:  # noqa: ANN001
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


def multidict_constructor(loader: yaml.Loader, node: yaml.SequenceNode) -> MultiDict:
    """PyYaml constructor to read MultiDict objects."""
    return MultiDict(*loader.construct_sequence(node))


# Enable PyYaml to read MultiDict objects
yaml.FullLoader.add_constructor("!MultiDict", multidict_constructor)


@pytest.fixture(scope="session")
def run_aims(request) -> bool | str:  # noqa: ANN001
    """Return the value of the --run-aims command line option."""
    return request.config.getoption("--run-aims")


@pytest.fixture(scope="session")
def cwd() -> Path:
    """Return the current working directory."""
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def aims_calc_dir(run_aims, cwd) -> Generator[str, None, None] | str:  # noqa: ANN001
    """
    Run FHI-aims calculations using a custom binary if specified by --run-aims.

    If the calculation has already been run (ie. if the directory
    `custom_bin_aims_calcs` exists), the calculations will not be run again, unless the
    user specifies `change_bin` as an option to --run-aims.
    """
    # Check if the directory already exists
    if Path("custom_bin_aims_calcs").is_dir() and run_aims != "change_bin":
        return "custom_bin_aims_calcs"
    elif run_aims is not False:
        binary = aims_bin_path_prompt(run_aims, cwd)
        subprocess.run(  # noqa: S603
            ["bash", str(cwd / "run_aims.sh"), str(binary), str(run_aims)],  # noqa: S607
            check=False,
        )
        yield "custom_bin_aims_calcs"
    else:
        yield "default_aims_calcs"


@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory) -> Path:  # noqa: ANN001
    """Temporary directory for all tests to write files to."""
    return tmp_path_factory.mktemp("tmp")


@pytest.fixture(scope="session")
def ref_data(cwd: FixtureFunctionMarker) -> Generator[dict, None, None]:
    """Load the appropriate test references too big for individual test functions."""
    with open(f"{cwd}/test_references.yaml") as references:
        # FullLoader necessary for parsing tuples
        # This is ok since references.yaml is hardcoded and not from user input
        yield yaml.load(references, Loader=yaml.FullLoader)  # noqa: S506


@pytest.fixture(scope="session")
def default_calc_dir(cwd: FixtureFunctionMarker) -> str:
    """Use `default_aims_calcs/1` as the default calculation directory."""
    return f"{cwd}/fixtures/default_aims_calcs/1/"
