import shutil
from collections.abc import Generator

import numpy as np
import pytest

from dfttoolkit.parameters import AimsControl


class TestAimsControl:
    """Test class for AimsControl."""

    @property
    def aims_fixture_no(self) -> int:
        return int(self.ac.path.split("/")[-2])

    @pytest.fixture(params=range(1, 14), autouse=True)
    def aims_control(self, cwd, request, aims_calc_dir) -> None:
        self.ac = AimsControl(
            control_in=f"{cwd}/fixtures/{aims_calc_dir}/{request.param!s}/control.in"
        )

    @pytest.fixture
    def added_keywords_ref_files(self, cwd) -> Generator[list[str], None, None]:
        with open(
            f"{cwd}/fixtures/manipulated_aims_files/add_keywords/"
            f"{self.aims_fixture_no}/control.in",
        ) as f:
            yield f.readlines()

    @pytest.fixture
    def removed_keywords_ref_files(self, cwd) -> Generator[list[str], None, None]:
        with open(
            f"{cwd}/fixtures/manipulated_aims_files/remove_keywords/"
            f"{self.aims_fixture_no}/control.in",
        ) as f:
            yield f.readlines()

    @pytest.fixture
    def cube_cell_ref_files(self, cwd) -> Generator[list[str] | None, None, None]:
        if self.aims_fixture_no != 13:
            with open(
                f"{cwd}/fixtures/manipulated_aims_files/cube_cell/"
                f"{self.aims_fixture_no}/control.in",
            ) as f:
                yield f.readlines()

        else:
            yield None

    def test_get_keywords(self, ref_data, cwd) -> None:
        assert self.ac.get_keywords() == ref_data["keywords"][self.aims_fixture_no - 1]

        # Check it works for the cube files as these have multiple keywords that are the
        # same with different values
        if self.aims_fixture_no != 13:
            cube_ac = AimsControl(
                control_in=f"{cwd}/fixtures/manipulated_aims_files/cube_cell/"
                f"{self.aims_fixture_no}/control.in"
            )
            # if self.aims_fixture_no != 13:
            assert (
                cube_ac.get_keywords()
                == ref_data["cube_keywords"][self.aims_fixture_no - 1]
            )

    def test_get_species(self) -> None:
        cluster_species = ["O", "H"]
        periodic_species = ["Si"]

        if self.aims_fixture_no in [1, 2, 3, 5, 7, 9]:
            assert self.ac.get_species() == cluster_species
        elif self.aims_fixture_no == 13:
            assert self.ac.get_species() == ["Cu", "O", "C"]
        else:
            assert self.ac.get_species() == periodic_species

    def test_get_default_basis_funcs(self) -> None:
        basis_funcs = [
            {
                "O": ["hydro 2 p 1.8", "hydro 3 d 7.6", "hydro 3 s 6.4"],
                "H": ["hydro 2 s 2.1", "hydro 2 p 3.5"],
            },
            {"Si": ["hydro 3 d 4.2", "hydro 2 p 1.4", "hydro 4 f 6.2"]},
        ]

        if self.aims_fixture_no == 1:
            assert self.ac.get_default_basis_funcs() == basis_funcs[0]
        if self.aims_fixture_no == 4:
            assert self.ac.get_default_basis_funcs() == basis_funcs[1]

    def test_add_cube_cell_and_save(self, tmp_dir, cube_cell_ref_files) -> None:
        if self.aims_fixture_no != 13:
            control_path = tmp_dir / "control.in"
            shutil.copy(self.ac.path, control_path)
            ac = AimsControl(control_in=str(control_path))
            try:
                ac.add_keywords_and_save(("output", "cube total_density"))
                ac.add_cube_cell_and_save(np.eye(3, 3) * [3, 4, 5], resolution=100)
            except TypeError:
                assert not ac.check_periodic()
            else:
                assert (
                    "".join(cube_cell_ref_files) == control_path.read_text()
                )  # Check correct for periodic

    def test_add_keywords_and_save(self, tmp_dir, added_keywords_ref_files) -> None:
        control_path = tmp_dir / "control.in"
        shutil.copy(self.ac.path, control_path)
        ac = AimsControl(control_in=str(control_path))
        ac.add_keywords_and_save(
            ("xc", "dfauto scan"),
            ("output", "cube spin_density"),
            ("output", "mulliken"),
        )

        assert "".join(added_keywords_ref_files) == control_path.read_text()

    def test_remove_keywords_and_save(
        self, tmp_dir, removed_keywords_ref_files
    ) -> None:
        control_path = tmp_dir / "control.in"
        shutil.copy(self.ac.path, control_path)
        ac = AimsControl(control_in=str(control_path))
        ac.remove_keywords_and_save("xc", "relax_geometry", "k_grid")

        assert "".join(removed_keywords_ref_files) == control_path.read_text()

    def test_check_periodic(self) -> None:
        if self.aims_fixture_no in [4, 6, 8, 10, 11, 12]:
            assert self.ac.check_periodic() is True
        else:
            assert self.ac.check_periodic() is False


# ruff: noqa: ANN001, S101, ERA001
