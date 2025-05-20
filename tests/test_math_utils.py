import numpy as np
import numpy.typing as npt
import pytest

import dfttoolkit.utils.math_utils as mu


class TestMathUtils:
    """
    Tests and fixtures for math_utils.

    Fixtures
    --------
    x_1_arr : NDArray[np.int64]
        [1, 0, 0] as a numpy array
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
        frac_coords = mu.get_fractional_coords(xyz_1_arr, np.eye(1))
        assert np.allclose(frac_coords, xyz_1_arr)

    def test_get_cartesian_coords(self, xyz_1_arr) -> None:
        cart_coords = mu.get_cartesian_coords(xyz_1_arr, np.eye(1))
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


# ruff: noqa: ANN001, S101
