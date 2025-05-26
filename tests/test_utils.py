import os

import numpy as np
import numpy.typing as npt
import pytest

import dfttoolkit.utils.math_utils as mu
from dfttoolkit.utils.run_utils import no_repeat


class TestRunUtils:
    """Tests and fixtures for run_utils.py."""

    @pytest.fixture(scope="class")
    def default_calc_dir(self) -> str:
        return (
            f"{os.path.dirname(os.path.realpath(__file__))}/fixtures/"  # noqa: PTH120
            "default_aims_calcs/1"
        )

    @pytest.fixture(scope="class")
    def warn_msg(self) -> str:
        return (
            "Found keywords used in the `no_repeat` wrapper in `to_be_decorated`, "
            "which will override the values for the wrapper arguments. If this is the "
            "intended behaviour, this warning can be suppressed by specifying "
            "`suppress_warn=True` in the wrapper arguments."
        )

    @pytest.mark.parametrize(
        ("output_file", "default_calc_dir", "force"),
        [
            ("not_a_file", "aims.out", "aims.out"),
            (default_calc_dir, "./", default_calc_dir),
            (False, False, True),
        ],
        indirect=["default_calc_dir"],
    )
    def test_no_repeat_no_skip(self, output_file, default_calc_dir, force) -> None:
        @no_repeat(output_file=output_file, calc_dir=default_calc_dir, force=force)
        def to_be_decorated() -> bool:
            return True

        assert to_be_decorated()

    def test_no_repeat_skip(self, capfd, default_calc_dir) -> None:
        @no_repeat(calc_dir=default_calc_dir)
        def to_be_decorated() -> bool:
            return True

        to_be_decorated()

        out, err = capfd.readouterr()
        assert out == "Skipping `to_be_decorated`: `aims.out` already exists.\n"
        assert err == ""

    def test_no_repeat_override_kwargs_def(self, warn_msg, default_calc_dir) -> None:
        @no_repeat
        def to_be_decorated(
            output_file="aims.out", calc_dir=default_calc_dir, force=False
        ) -> bool:
            assert output_file == "aims.out"
            assert calc_dir == default_calc_dir
            assert force is False
            return True

        with pytest.warns(UserWarning, match=warn_msg):
            to_be_decorated()

    def test_no_repeat_override_kwargs_call(self, warn_msg, default_calc_dir) -> None:
        @no_repeat
        def to_be_decorated() -> bool:
            return True

        with pytest.warns(UserWarning, match=warn_msg):
            to_be_decorated(
                output_file="aims.out", calc_dir=default_calc_dir, force=False
            )

    def test_no_repeat_no_warn(self, capfd, default_calc_dir) -> None:
        @no_repeat
        def to_be_decorated() -> bool:
            return True

        to_be_decorated(
            output_file="aims.out",
            calc_dir=default_calc_dir,
            suppress_warn=True,
        )

        out, err = capfd.readouterr()
        assert out == "Skipping `to_be_decorated`: `aims.out` already exists.\n"
        assert err == ""

    def test_no_repeat_specified_dir_not_found(self) -> None:
        @no_repeat(calc_dir="bogus")
        def to_be_decorated() -> bool:
            return True

        with pytest.raises(NotADirectoryError) as excinfo:
            to_be_decorated()

        assert str(excinfo.value) == "Provided `calc_dir` is not a directory."


class TestMathUtils:
    """
    Tests and fixtures for math_utils.

    Fixtures
    --------
    x_1_arr : NDArray[np.int64]
        [1, 0, 0] as a numpy array
    y_1_arr : NDArray[np.int64]
        [0, 1, 0] as a numpy array
    z_1_arr : NDArray[np.int64]
        [0, 0, 1] as a numpy array
    xyz_1_arr : NDArray[np.int64]
        [1, 1, 1] as a numpy array
    arange_arr : NDArray[np.int64]
        [1, 2, 3, 4, 5] as a numpy array
    """

    # TODO: check if these can be autoused in each test
    @pytest.fixture
    def x_1_arr(self) -> npt.NDArray[np.int64]:
        return np.array([1, 0, 0])

    @pytest.fixture
    def y_1_arr(self) -> npt.NDArray[np.int64]:
        return np.array([0, 1, 0])

    @pytest.fixture
    def z_1_arr(self) -> npt.NDArray[np.int64]:
        return np.array([0, 0, 1])

    @pytest.fixture
    def xyz_1_arr(self) -> npt.NDArray[np.int64]:
        return np.array([1, 1, 1])

    @pytest.fixture
    def arange_arr(self) -> npt.NDArray[np.int64]:
        return np.arange(1, 6)

    def test_get_rotation_matrix(self, x_1_arr, y_1_arr) -> None:
        R = mu.get_rotation_matrix(x_1_arr, y_1_arr)
        rotated_vec = R @ x_1_arr
        assert np.allclose(rotated_vec, y_1_arr)

    def test_get_rotation_matrix_around_axis(self, x_1_arr, y_1_arr, z_1_arr) -> None:
        phi = np.pi / 2
        R = mu.get_rotation_matrix_around_axis(z_1_arr, phi)
        rotated_vec = R @ x_1_arr
        assert np.allclose(rotated_vec, -y_1_arr)

    def test_get_mirror_matrix(self, x_1_arr, arange_arr) -> None:
        M = mu.get_mirror_matrix(x_1_arr)
        mirrored_vec = M @ arange_arr[0:3]
        assert np.allclose(mirrored_vec, np.array([-1, 2, 3]))

    def test_get_angle_between_vectors(self, x_1_arr, y_1_arr) -> None:
        angle = mu.get_angle_between_vectors(x_1_arr, y_1_arr)
        assert np.isclose(angle, 0)

    def test_get_fractional_coords(self, xyz_1_arr) -> None:
        frac_coords = mu.get_fractional_coords(xyz_1_arr, np.eye(3))
        assert np.allclose(frac_coords, xyz_1_arr)

    def test_get_cartesian_coords(self, xyz_1_arr) -> None:
        cart_coords = mu.get_cartesian_coords(xyz_1_arr, np.eye(3))
        assert np.allclose(cart_coords, xyz_1_arr)

    def test_get_triple_product(self, x_1_arr, y_1_arr, z_1_arr) -> None:
        result = mu.get_triple_product(x_1_arr, y_1_arr, z_1_arr)
        assert np.isclose(result, 1)

    def test_smooth_function(self, arange_arr) -> None:
        smoothed = mu.smooth_function(arange_arr, 3)
        assert len(smoothed) == len(arange_arr)

    def test_get_cross_correlation_function(self) -> None:
        signal = np.cos(np.linspace(0, 1000, 10000))
        corr = mu.get_cross_correlation_function(signal, signal)
        x = 2 * corr[:500] - signal[:500]
        assert np.all(np.abs(x) < 1e-3)

    def test_get_fourier_transform(self, arange_arr) -> None:
        freqs, ft = mu.get_fourier_transform(arange_arr[0:4], 1.0)
        assert len(freqs) == len(ft)

    def test_lorentzian(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = mu.lorentzian(x, 1.0, 1.0, 1.0)
        assert isinstance(y, np.ndarray)
        assert x.shape == y.shape

    def test_gaussian_window(self) -> None:
        window = mu.gaussian_window(10)
        assert len(window) == 10

    def test_apply_gaussian_window(self) -> None:
        data = np.ones(10)
        windowed_data = mu.apply_gaussian_window(data)
        assert len(windowed_data) == len(data)

    def test_norm_matrix_by_dagonal(self) -> None:
        matrix = np.eye(3) * 2
        normed_matrix = mu.norm_matrix_by_dagonal(matrix)
        assert np.allclose(np.diag(normed_matrix), np.ones(3))

    def test_mae(self) -> None:
        delta = np.array([1.0, -1.0, 2.0])
        assert np.isclose(mu.mae(delta), 4 / 3)

    def test_rel_mae(self, xyz_1_arr) -> None:
        delta = np.array([1.0, -1.0, 2.0])
        assert np.isclose(mu.rel_mae(delta, xyz_1_arr * 2), 2 / 3)


class TestPeriodicTable: ...  # noqa: D101



# ruff: noqa: ANN001, S101
