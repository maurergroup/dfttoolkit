import numpy as np

from dfttoolkit.utils.math_utils import *


class TestMathUtils:
    def test_get_rotation_matrix(self) -> None:
        vec_start = np.array([1, 0, 0])
        vec_end = np.array([0, 1, 0])
        R = get_rotation_matrix(vec_start, vec_end)
        rotated_vec = R @ vec_start
        assert np.allclose(rotated_vec, vec_end)

    def test_get_rotation_matrix_around_axis(self) -> None:
        axis = np.array([0, 0, 1])
        phi = np.pi / 2
        R = get_rotation_matrix_around_axis(axis, phi)
        vec = np.array([1, 0, 0])
        rotated_vec = R @ vec
        assert np.allclose(rotated_vec, np.array([0, -1, 0]))

    def test_get_mirror_matrix(self) -> None:
        normal = np.array([1, 0, 0])
        M = get_mirror_matrix(normal)
        vec = np.array([1, 2, 3])
        mirrored_vec = M @ vec
        assert np.allclose(mirrored_vec, np.array([-1, 2, 3]))

    def test_get_angle_between_vectors(self) -> None:
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        angle = get_angle_between_vectors(v1, v2)
        assert np.isclose(angle, 0)

    def test_get_fractional_coords(self) -> None:
        lattice = np.eye(3)
        cart_coords = np.array([[1, 1, 1]])
        frac_coords = get_fractional_coords(cart_coords, lattice)
        assert np.allclose(frac_coords, cart_coords)

    def test_get_cartesian_coords(self) -> None:
        lattice = np.eye(3)
        frac_coords = np.array([[1, 1, 1]])
        cart_coords = get_cartesian_coords(frac_coords, lattice)
        assert np.allclose(cart_coords, frac_coords)

    def test_get_triple_product(self) -> None:
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        c = np.array([0, 0, 1])
        result = get_triple_product(a, b, c)
        assert np.isclose(result, 1)

    def test_smooth_function(self) -> None:
        y = np.array([1, 2, 3, 4, 5])
        smoothed = smooth_function(y, 3)
        assert len(smoothed) == len(y)

    def test_get_cross_correlation_function(self) -> None:
        signal = np.cos(np.linspace(0, 1000, 10000))
        corr = get_cross_correlation_function(signal, signal)
        x = 2 * corr[:500] - signal[:500]
        assert np.all(np.abs(x) < 1e-3)

    def test_get_fourier_transform(self) -> None:
        signal = np.array([1, 2, 3, 4])
        freqs, ft = get_fourier_transform(signal, 1.0)
        assert len(freqs) == len(ft)

    def test_lorentzian(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = lorentzian(x, 1.0, 1.0, 1.0)
        assert y.shape == x.shape

    def test_gaussian_window(self) -> None:
        window = gaussian_window(10)
        assert len(window) == 10

    def test_apply_gaussian_window(self) -> None:
        data = np.ones(10)
        windowed_data = apply_gaussian_window(data)
        assert len(windowed_data) == len(data)

    def test_norm_matrix_by_dagonal(self) -> None:
        matrix = np.eye(3) * 2
        normed_matrix = norm_matrix_by_dagonal(matrix)
        assert np.allclose(np.diag(normed_matrix), np.ones(3))

    def test_mae(self) -> None:
        delta = np.array([1.0, -1.0, 2.0])
        assert np.isclose(mae(delta), 4 / 3)

    def test_rel_mae(self) -> None:
        delta = np.array([1.0, -1.0, 2.0])
        target = np.array([2.0, 2.0, 2.0])
        assert np.isclose(rel_mae(delta, target), 2 / 3)
