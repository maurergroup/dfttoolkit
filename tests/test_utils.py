import pytest
import os
from dfttoolkit.utils import utils


def test_specified_dir_not_found():
    @utils.no_repeat(calc_dir="bogus")
    def to_be_decorated():
        return True

    with pytest.raises(ValueError) as excinfo:
        to_be_decorated()

    assert "bogus is not a directory." == str(excinfo.value)


@pytest.mark.parametrize(
    ("default_calc_dir", "force", "expected"),
    [
        ("./", False, True),
        (
            f"{os.path.dirname(os.path.realpath(__file__))}/fixtures/default_aims_calcs/1",
            True,
            True,
        ),
    ],
)
def test_default_dir_no_args(default_calc_dir, force, expected):
    @utils.no_repeat(calc_dir=default_calc_dir, force=force)
    def to_be_decorated():
        return True

    assert to_be_decorated() == expected


def test_no_repeat(capfd, default_calc_dir):
    @utils.no_repeat(calc_dir=default_calc_dir)
    def to_be_decorated():
        return True

    to_be_decorated()

    out, err = capfd.readouterr()
    assert (
        out == f"aims.out already exists in {default_calc_dir}. Skipping calculation.\n"
    )
    assert err == ""


class TestPeriodicTable: ...
