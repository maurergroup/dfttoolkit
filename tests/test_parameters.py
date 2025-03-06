import shutil

import pytest
import numpy as np
from dfttoolkit.parameters import AimsControl


class TestAimsControl:
    @property
    def aims_fixture_no(self) -> int:
        return int(self.ac.path.split("/")[-2])

    @pytest.fixture(params=range(1, 13), autouse=True)
    def aims_control(self, cwd, request, aims_calc_dir):
        self.ac = AimsControl(
            control_in=f"{cwd}/fixtures/{aims_calc_dir}/{str(request.param)}/control.in"
        )

    @pytest.fixture
    def added_keywords_ref_files(self, cwd):
        with open(
            f"{cwd}/fixtures/manipulated_aims_files/add_keywords/"
            f"{self.aims_fixture_no}/control.in",
            "r",
        ) as f:
            yield f.readlines()

    @pytest.fixture
    def removed_keywords_ref_files(self, cwd):
        with open(
            f"{cwd}/fixtures/manipulated_aims_files/remove_keywords/"
            f"{self.aims_fixture_no}/control.in",
            "r",
        ) as f:
            yield f.readlines()

    @pytest.fixture
    def cube_cell_ref_files(self, cwd):
        with open(
            f"{cwd}/fixtures/manipulated_aims_files/remove_keywords/"
            f"{self.aims_fixture_no}/control.in",
            "r",
        ) as f:
            yield f.readlines()

    def test_get_keywords(self, ref_data):
        keywords = self.ac.get_keywords()
        assert keywords == ref_data["keywords"][self.aims_fixture_no - 1]

    def test_add_keywords(self, tmp_dir, added_keywords_ref_files):
        control_path = tmp_dir / "control.in"
        shutil.copy(self.ac.path, control_path)
        ac = AimsControl(control_in=str(control_path))
        ac.add_keywords(xc="dfauto scan", output="cube spin_density")

        assert "".join(added_keywords_ref_files) == control_path.read_text()

    def test_add_cube_cell(self, tmp_dir, cube_cell_ref_files):
        control_path = tmp_dir / "control.in"
        shutil.copy(self.ac.path, control_path)
        ac = AimsControl(control_in=str(control_path))
        try:
            ac.add_keywords(output="cube total_density")
            ac.add_cube_cell(np.eye(3, 3) * [3, 4, 5], resolution=100)
        except TypeError:
            assert ac.check_periodic() == False
        else:
            assert (
                "".join(cube_cell_ref_files) == control_path.read_text()
            )  # Check correct for periodic

    def test_remove_keywords(self, tmp_dir, removed_keywords_ref_files):
        control_path = tmp_dir / "control.in"
        shutil.copy(self.ac.path, control_path)
        ac = AimsControl(control_in=str(control_path))
        ac.remove_keywords("xc", "relax_geometry", "k_grid")

        assert "".join(removed_keywords_ref_files) == control_path.read_text()

    def test_get_species(self):
        cluster_species = ["O", "H"]
        periodic_species = ["Si"]

        if self.aims_fixture_no in [1, 2, 3, 5, 7, 9]:
            assert self.ac.get_species() == cluster_species
        else:
            assert self.ac.get_species() == periodic_species

    def test_get_default_basis_funcs(self):
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

    def test_check_periodic(self):
        if self.aims_fixture_no in [4, 6, 8, 10, 11, 12]:
            assert self.ac.check_periodic() is True
        else:
            assert self.ac.check_periodic() is False
