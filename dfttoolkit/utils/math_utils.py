from copy import deepcopy
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy


def get_rotation_matrix(
    vec_start: npt.NDArray[np.float64], vec_end: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Calculate the rotation matrix to align two unit vectors.

    Given a two (unit) vectors, vec_start and vec_end, this function calculates the
    rotation matrix U, so that U * vec_start = vec_end.

    U the is rotation matrix that rotates vec_start to point in the direction
    of vec_end.

    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677

    Parameters
    ----------
    vec_start, vec_end : NDArray[float64]
        Two vectors that should be aligned. Both vectors must have a l2-norm of 1.

    Returns
    -------
    NDArray[float64]
        The rotation matrix U as npt.NDArray with shape (3,3)
    """
    if not np.isclose(np.linalg.norm(vec_start), 1) and not np.isclose(
        np.linalg.norm(vec_end), 1
    ):
        raise ValueError("`vec_start` and `vec_end` args must be unit vectors")

    v = np.cross(vec_start, vec_end)
    c = np.dot(vec_start, vec_end)
    v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + v_x + v_x.dot(v_x) / (1 + c)


def get_rotation_matrix_around_axis(axis: npt.NDArray, phi: float) -> npt.NDArray:
    """
    Generate a rotation matrix around a given vector.

    Parameters
    ----------
    axis : NDArray
        Axis around which the rotation is done.
    phi : float
        Angle of rotation around axis in radiants.

    Returns
    -------
    NDArray
        Rotation matrix
    """
    axis_vec = np.array(axis, dtype=np.float64)
    axis_vec /= np.linalg.norm(axis_vec)

    eye = np.eye(3, dtype=np.float64)
    ddt = np.outer(axis_vec, axis_vec)
    skew = np.array(
        [
            [0, axis_vec[2], -axis_vec[1]],
            [-axis_vec[2], 0, axis_vec[0]],
            [axis_vec[1], -axis_vec[0], 0],
        ],
        dtype=np.float64,
    )

    return ddt + np.cos(phi) * (eye - ddt) + np.sin(phi) * skew


def get_rotation_matrix_around_z_axis(phi: float) -> npt.NDArray:
    """
    Generate a rotation matrix around the z axis.

    Parameters
    ----------
    phi : float
        Angle of rotation around axis in radiants.

    Returns
    -------
    NDArray
        Rotation matrix
    """
    return get_rotation_matrix_around_axis(np.array([0.0, 0.0, 1.0]), phi)


def get_mirror_matrix(normal_vector: npt.NDArray) -> npt.NDArray:
    """
    Generate a transformation matrix for mirroring through plane given by the normal.

    Parameters
    ----------
    normal_vector : NDArray
        Normal vector of the mirror plane.

    Returns
    -------
    NDArray
        Mirror matrix
    """
    n_vec = normal_vector / np.linalg.norm(normal_vector)
    eps = np.finfo(np.float64).eps
    a = n_vec[0]
    b = n_vec[1]
    c = n_vec[2]
    M = np.array(
        [
            [1 - 2 * a**2, -2 * a * b, -2 * a * c],
            [-2 * a * b, 1 - 2 * b**2, -2 * b * c],
            [-2 * a * c, -2 * b * c, 1 - 2 * c**2],
        ]
    )
    M[np.abs(M) < eps * 10] = 0
    return M


def get_angle_between_vectors(
    vector_1: npt.NDArray, vector_2: npt.NDArray
) -> npt.NDArray:
    """
    Determine angle between two vectors.

    Parameters
    ----------
    vector_1 : NDArray
    vector_2 : NDArray

    Returns
    -------
    float
        Angle in radiants.
    """
    return (
        np.dot(vector_1, vector_2) / np.linalg.norm(vector_1) / np.linalg.norm(vector_2)
    )


def get_fractional_coords(
    cartesian_coords: npt.NDArray, lattice_vectors: npt.NDArray
) -> npt.NDArray:
    """
    Transform cartesian coordinates into fractional coordinates.

    Parameters
    ----------
    cartesian_coords: NDArray
        Cartesian coordinates of atoms (can be Nx2 or Nx3)
    lattice_vectors: [N_dim x N_dim] numpy array:
        Matrix of lattice vectors: Each ROW corresponds to one lattice vector!

    Returns
    -------
    fractional_coords: [N x N_dim] numpy array
        Fractional coordinates of atoms
    """
    fractional_coords = np.linalg.solve(lattice_vectors.T, cartesian_coords.T)
    return fractional_coords.T


def get_cartesian_coords(
    frac_coords: npt.NDArray, lattice_vectors: npt.NDArray
) -> npt.NDArray:
    """
    Transform fractional coordinates into cartesian coordinates.

    Parameters
    ----------
    frac_coords: NDArray
        Fractional coordinates of atoms (can be Nx2 or Nx3)
    lattice_vectors: NDArray
        Matrix of lattice vectors: Each ROW corresponds to one lattice vector!

    Returns
    -------
    NDArray
        Cartesian coordinates of atoms

    """
    return np.dot(frac_coords, lattice_vectors)


def get_triple_product(a: npt.NDArray, b: npt.NDArray, c: npt.NDArray) -> npt.NDArray:
    """
    Get the triple product (DE: Spatprodukt): a*(bxc).

    Parameters
    ----------
    a: NDArray
        TODO
    b: NDArray
        TODO
    c: NDArray
        TODO

    Returns
    -------
    NDarray
        Triple product of each input vector
    """
    if len(a) != 3 or len(b) != 3 or len(c) != 3:
        raise ValueError("Each vector must be of length 3")

    return np.dot(np.cross(a, b), c)


def smooth_function(y: npt.NDArray, box_pts: int) -> npt.NDArray[np.floating[Any]]:
    """
    Smooths a function using convolution.

    Parameters
    ----------
    y : NDArray
        TODO
    box_pts : int
        TODO

    Returns
    -------
    y_smooth : NDArray[floating[Any]]
        TODO
    """
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode="same")


def get_cross_correlation_function(
    signal_0: npt.NDArray,
    signal_1: npt.NDArray,
    detrend: bool = False,
) -> npt.NDArray:
    """
    Calculate the autocorrelation function for a given signal.

    Parameters
    ----------
    signal_0 : NDArray
        First siganl for which the correlation function should be calculated.
    signal_1 : NDArray
        Second siganl for which the correlation function should be calculated.

    Returns
    -------
    NDArray
        Autocorrelation function from 0 to max_lag.
    """
    if detrend:
        signal_0 = scipy.signal.detrend(signal_0)
        signal_1 = scipy.signal.detrend(signal_1)

    cross_correlation = np.correlate(signal_0, signal_1, mode="full")
    cross_correlation = cross_correlation[cross_correlation.size // 2 :]

    # normalize by number of overlapping data points
    cross_correlation /= np.arange(cross_correlation.size, 0, -1)
    cutoff = int(cross_correlation.size * 0.75)
    return cross_correlation[:cutoff]


def get_autocorrelation_function_manual_lag(
    signal: npt.NDArray, max_lag: int
) -> npt.NDArray:
    """
    Alternative method for autocorrelation of a signal using numpy.corrcoef.

    Allows the lag to be set manually.

    Parameters
    ----------
    signal : NDArray
        Siganl for which the autocorrelation function should be calculated.
    max_lag : int | None
        Autocorrelation will be calculated for a range of 0 to max_lag,
        where max_lag is the largest lag for the calculation of the
        autocorrelation function

    Returns
    -------
    autocorrelation : npt.NDArray
        Autocorrelation function from 0 to max_lag.
    """
    lag = npt.NDArray(range(max_lag))
    autocorrelation = np.array([np.nan] * max_lag)

    for i in lag:
        corr = 1.0 if i == 0 else np.corrcoef(signal[i], signal[:-i])[0][1]
        autocorrelation[i] = corr

    return autocorrelation


def get_fourier_transform(
    signal: npt.NDArray, time_step: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Calculate the fourier transform of a given siganl.

    Parameters
    ----------
    signal : NDArray
        Siganl for which the autocorrelation function should be calculated.
    time_step : float
        Time step of the signal in seconds.

    Returns
    -------
    tuple[NDArray, NDArray]
        Frequencs and absolute values of the fourier transform.

    """
    f = scipy.fft.fftfreq(signal.size, d=time_step)
    y = scipy.fft.fft(signal)

    L = f >= 0

    return f[L], y[L]


def lorentzian(
    x: float | npt.NDArray[np.float64], a: float, b: float, c: float
) -> float | npt.NDArray[np.float64]:
    """
    Return a Lorentzian function.

    Parameters
    ----------
    x : float | NDArray[float64]
        Argument x of f(x) --> y.
    a : float
        Maximum of Lorentzian.
    b : float
        Width of Lorentzian.
    c : float
        Magnitude of Lorentzian.

    Returns
    -------
    f : float | NDArray[float64]
        Outupt of a Lorentzian function.
    """
    return c / (1.0 + ((x - a) / (b / 2.0)) ** 2)


def exponential(
    x: float | npt.NDArray[np.float64], a: float, b: float
) -> float | npt.NDArray[np.float64]:
    """
    Return an exponential function.

    Parameters
    ----------
    x : float | NDArray[float64]
        Argument x of f(x) --> y
    a : float
        TODO
    b : float
        TODO
    """
    return a * np.exp(x * b)


def double_exponential(
    x: float | npt.NDArray, a: float, b: float, c: float
) -> float | npt.NDArray:
    """TODO."""
    return a * (np.exp(x * b) + np.exp(x * c))


def gaussian_window(N: int, std: float = 0.4) -> npt.NDArray[np.float64]:
    """
    Generate a Gaussian window.

    Parameters
    ----------
    N : int
        Number of points in the window.
    std : float
        Standard deviation of the Gaussian window, normalized
        such that the maximum value occurs at the center of the window.

    Returns
    -------
    window : NDarray[float64]
        Gaussian window of length N.

    """
    n = np.linspace(-1, 1, N)
    return np.exp(-0.5 * (n / std) ** 2)


def apply_gaussian_window(
    data: npt.NDArray[np.float64], std: float = 0.4
) -> npt.NDArray[np.float64]:
    """
    Apply a Gaussian window to an array.

    Parameters
    ----------
    data : NDarray[float64]
        Input data array to be windowed.
    std : float
        Standard deviation of the Gaussian window.

    Returns
    -------
    windowed_data : NDArray[float64]
        Windowed data array.
    """
    N = len(data)
    window = gaussian_window(N, std)
    return data * window


def hann_window(N: int) -> npt.NDArray[np.float64]:
    """
    Generate a Hann window.

    Parameters
    ----------
    N : int
        Number of points in the window.

    Returns
    -------
    NDArray
        Hann window of length N.
    """
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))


def apply_hann_window(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Apply a Hann window to an array.

    Parameters
    ----------
    data : NDarray[float64]
        Input data array to be windowed.

    Returns
    -------
    NDarray[float64]
        Windowed data array.
    """
    N = len(data)
    window = hann_window(N)
    return data * window


def norm_matrix_by_dagonal(matrix: npt.NDArray) -> npt.NDArray:
    """
    Norms a matrix such that the diagonal becomes 1.

    | a_11 a_12 a_13 |       |   1   a'_12 a'_13 |
    | a_21 a_22 a_23 |  -->  | a'_21   1   a'_23 |
    | a_31 a_32 a_33 |       | a'_31 a'_32   1   |

    Parameters
    ----------
    matrix : NDArray
        Matrix that should be normed.

    Returns
    -------
    matrix : NDArray
        Normed matrix.
    """
    diagonal = np.array(np.diagonal(matrix))
    L = diagonal == 0.0
    diagonal[L] = 1.0

    new_matrix = deepcopy(matrix)
    new_matrix /= np.sqrt(
        np.tile(diagonal, (matrix.shape[1], 1)).T
        * np.tile(diagonal, (matrix.shape[0], 1))
    )

    return new_matrix


def mae(delta: npt.NDArray) -> np.floating:
    """
    Calculate the mean absolute error from a list of value differnces.

    Parameters
    ----------
    delta : NDArray
        Array containing differences

    Returns
    -------
    float
        mean absolute error
    """
    return np.mean(np.abs(delta))


def rel_mae(
    delta: npt.NDArray, target_val: npt.NDArray
) -> np.floating | np.float64 | float:
    """
    Compute relative MAE from value differences and corresponding target values.

    Parameters
    ----------
    delta : NDArray
        Array containing differences
    target_val : NDArray
        Array of target values against which the difference should be compared

    Returns
    -------
    float
        relative mean absolute error
    """
    target_norm = np.mean(np.abs(target_val))
    return np.mean(np.abs(delta)).item() / (target_norm + 1e-9)


def rmse(delta: npt.NDArray) -> float:
    """
    Calculate the root mean sqare error from a list of value differnces.

    Parameters
    ----------
    delta : NDArray
        Array containing differences

    Returns
    -------
    float
        root mean square error
    """
    return np.sqrt(np.mean(np.square(delta)))


def rel_rmse(delta: npt.NDArray, target_val: npt.NDArray) -> float:
    """
    Calculate the relative root mean sqare error from a list of value differences.

    Parameters
    ----------
    delta : NDArray
        Array containing differences
    target_val : NDArray
        Array of target values against which the difference should be compared

    Returns
    -------
    float
        relative root mean sqare error

    """
    target_norm = np.sqrt(np.mean(np.square(target_val)))
    return np.sqrt(np.mean(np.square(delta))) / (target_norm + 1e-9)


def get_moving_average(
    signal: npt.NDArray[np.float64], window_size: int
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Cacluated the moving average and the variance around the moving average.

    Parameters
    ----------
    signal : NDArray[float64]
        Signal for which the moving average should be calculated.
    window_size : int
        Window size for the mocing average.

    Returns
    -------
    npt.NDArray[floating]
        Moving average.
    npt.NDArray[floating]
        Variance around the moving average.
    """
    moving_avg = np.convolve(signal, np.ones(window_size) / window_size, mode="valid")
    variance = np.array(
        [
            np.var(signal[i : i + window_size])
            for i in range(len(signal) - window_size + 1)
        ]
    )

    return moving_avg, variance


def get_maxima_in_moving_interval(
    function_values: npt.NDArray,
    interval_size: int,
    step_size: int,
    filter_value: int,
) -> npt.NDArray:
    """
    Slide an interval along the function, filtering out points below a threshold.

    Points smaller than filter_value times the maximum within the interval are removed.

    Parameters
    ----------
    function_values : NDArray
        1D array of function values.
    interval_size : int
        Size of the interval (number of points).
    step_size : int
        Step size for moving the interval.

    Returns
    -------
    NDArray
        Filtered array where points outside the threshold are set to NaN.
    """
    # Convert input to a NumPy array
    function_values = np.asarray(function_values)

    # Initialize an array to store the result
    filtered_indices = []

    # Slide the interval along the function
    for start in range(0, len(function_values) - interval_size + 1, step_size):
        # Define the end of the interval
        end = start + interval_size

        # Extract the interval
        interval = function_values[start:end]

        # Find the maximum value in the interval
        max_value = np.max(interval)

        # Apply the filter
        threshold = filter_value * max_value

        indices = np.where(interval >= threshold)
        filtered_indices += list(start + indices[0])

    return np.array(list(set(filtered_indices)))


def get_pearson_correlation_coefficient(x: npt.NDArray, y: npt.NDArray) -> np.floating:
    """
    TODO.

    Parameters
    ----------
    x : npt.NDArray
        First array of data points.
    y : npt.NDArray
        Second array of data points.

    Returns
    -------
    floating
        Pearson correlation coefficient between x and y.
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)

    return np.mean((x - mean_x) * (y - mean_y)) / std_x / std_y


def get_t_test(x: npt.NDArray, y: npt.NDArray) -> np.floating:
    """
    Calculate the t-test statistic for two sets of data.

    Parameters
    ----------
    x : NDArray
        First array of data points.
    y : NDArray
        Second array of data points.

    Returns
    -------
    floating
        t-test statistic for the two sets of data.
    """
    r = get_pearson_correlation_coefficient(x, y)
    n = len(x)

    return np.abs(r) * np.sqrt((n - 2) / (1 - r**2))


def probability_density(t: float, n: int) -> float:
    """
    Probability density function for the t-distribution.

    Parameters
    ----------
    t : float
        The t-value for which the probability density is calculated.
    n : int
        The number of data points in the sample.

    Returns
    -------
    float
        The probability density at the given t-value.
    """
    degrees_of_freedom = n - 2

    return (
        scipy.special.gamma((degrees_of_freedom + 1.0) / 2.0)
        / (
            np.sqrt(np.pi * degrees_of_freedom)
            * scipy.special.gamma(degrees_of_freedom / 2.0)
        )
        * (1 + t**2 / degrees_of_freedom) ** (-(degrees_of_freedom + 1.0) / 2.0)
    )


def get_significance(x: npt.NDArray, t: float) -> float:
    """
    Calculate the significance of a t-test statistic.

    Parameters
    ----------
    x : npt.NDArray
        Array of data points.
    t : float
        t-test statistic value.

    Returns
    -------
    float
        The significance level of the t-test statistic.
    """
    n = len(x)

    return scipy.integrate.quad(probability_density, -np.inf, t, args=(n))[0]


def squared_exponential_kernel(
    x1_vec: npt.NDArray, x2_vec: npt.NDArray, tau: float
) -> npt.NDArray[np.float64]:
    """
    Return a simple squared exponential kernel to determine a similarity measure.

    Parameters
    ----------
    x1_vec : NDArray
        Descriptor for first set of data of shape
        [data points, descriptor dimensions].
    x2_vec : NDArray
        Descriptor for second set of data of shape
        [data points, descriptor dimensions].
    tau : float
        Correlation length for the descriptor.

    Returns
    -------
    NDArray[float64]
        Matrix contianing pairwise kernel values.
    """
    # Ensure inputs are at least 2D (n_samples, n_features)
    x1 = np.atleast_2d(x1_vec)
    x2 = np.atleast_2d(x2_vec)

    # If they are 1D row-vectors, convert to column vectors
    if x1.shape[0] == 1 or x1.shape[1] == 1:
        x1 = x1.reshape(-1, 1)
    if x2.shape[0] == 1 or x2.shape[1] == 1:
        x2 = x2.reshape(-1, 1)

    # Compute squared distances (broadcasting works for both 1D and 2D now)
    diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
    sq_dist = np.sum(diff**2, axis=2)

    # Apply the RBF formula
    return np.exp(-0.5 * sq_dist / tau**2)


class GPR:
    """
    A simple Gaussian Process Regression (GPR) model for interpolation and smoothing.

    Parameters
    ----------
    x : NDArray
        Descriptor of shape [data points, descriptor dimensions].
    y : NDArray
        Data to be learned.
    tau : float
        Correlation length for the descriptor.
    sigma : float
        Uncertainty of the input data.
    """

    def __init__(self, x: npt.NDArray, y: npt.NDArray, tau: float, sigma: float):
        K1 = squared_exponential_kernel(x, x, tau)

        self.K1_inv = np.linalg.inv(K1 + np.eye(len(x)) * sigma)
        self.x = x
        self.y = y
        self.tau = tau
        self.sigma = sigma

    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        return self.predict(x)

    def predict(self, x: npt.NDArray) -> npt.NDArray:
        """
        TODO.

        Parameters
        ----------
        x : NDArray
            TODO

        Returns
        -------
        NDArray
            TODO
        """
        K2 = squared_exponential_kernel(x, self.x, self.tau)
        y_test0 = K2.dot(self.K1_inv)

        return y_test0.dot(self.y)
