import numpy as np
import pytest

from dfttoolkit.geometry import AimsGeometry


class TestAimsGeometry:
    """Test the AimsGeometry class."""

    @property
    def _aims_fixture_no(self) -> int:
        return int(self.aims_geometries.path.split("/")[-2])

    @pytest.fixture(params=range(10, 11), autouse=True)
    def aims_out(self, cwd, request, aims_calc_dir) -> None:
        self.ap_geom_load = AimsGeometry(
            f"{cwd}/fixtures/{aims_calc_dir}/{request.param}/geometry.in"
        )

    @pytest.fixture(autouse=True)
    def geometry(self) -> None:
        self.geom = AimsGeometry()
        self.geom.add_atoms(
            cartesian_coords=[
                [0, 0, 0],
                [1, 0, 0],
            ],
            species=["H", "H"],
            constrain_relax=[
                np.array([False, False, False]),
                np.array([True, True, True]),
            ],
        )

    @pytest.fixture(autouse=True)
    def geometry_periodic(self) -> None:
        self.ap_geom = AimsGeometry()

        self.ap_geom.lattice_vectors = np.array(
            [
                [1.27349850, 2.20576400, 0.00000000],
                [-1.27349850, 2.20576400, 0.00000000],
                [0.00000000, 0.00000000, 68.33693789],
            ]
        )

        self.ap_geom.add_atoms(
            cartesian_coords=[
                [-0.00000002, 1.47049920, 0.00000000],
                [0.00000000, 0.00000000, 2.07961400],
                [0.00000000, 2.94102000, 4.15922800],
                [-0.00000002, 1.47049920, 6.23160806],
                [0.00000002, -0.00000809, 8.30498122],
            ],
            species=[
                "Cu",
                "Cu",
                "Cu",
                "Cu",
                "Cu",
            ],
            constrain_relax=[
                np.array([True, True, True]),
                np.array([True, True, True]),
                np.array([True, True, True]),
                np.array([False, False, False]),
                np.array([False, False, False]),
            ],
        )

    def test_save_and_read_file(self, tmp_path) -> None:
        # pytest method of creating a temporary directory
        d = tmp_path / "tmp"
        d.mkdir()

        geom_path = str(d / "geometry.in")
        self.geom.save_to_file(geom_path)

        geometry_read = AimsGeometry(geom_path)

        assert self.geom == geometry_read

    def test_get_displaced_atoms(self) -> None:
        new_geom = self.geom.get_displaced_atoms(1)
        assert not np.allclose(self.geom.coords[0], new_geom.coords[0])
        assert np.allclose(self.geom.coords[1], new_geom.coords[1])

    def test_get_symmetries(self) -> None:
        symmetries = self.ap_geom.get_symmetries(symmetry_precision=1e-03)
        assert len(symmetries[0]) == 6
        assert len(symmetries[1]) == 6

    def test_get_number_of_electrons(self) -> None:
        n_electrons = self.ap_geom.get_number_of_electrons()
        assert n_electrons == 145

    def test_get_slab(self) -> None:
        slab = self.ap_geom_load.get_slab(4, surface=(1, 1, 1))

        test_lattice_vectors = np.array(
            [
                [3.87256419, 0.00000000, 0.00000000],
                [-1.93628209, 3.35373897, 0.00000000],
                [0.00000000, 0.00000000, 9.48580626],
            ]
        )

        assert np.allclose(slab.lattice_vectors, test_lattice_vectors)


# ruff: noqa: ANN001, S101, ERA001
