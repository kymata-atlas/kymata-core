import numpy as np
from numpy.typing import NDArray


def normalize(x: NDArray, inplace: bool = False) -> NDArray:
    """
    Remove the mean and divide by the Euclidean magnitude.

    raises: ZeroDivisionError

    Args:
        x (DNArray): Data array of shape [..., t], where t represents the dimension over which to normalise.
        inplace (bool): If inplace is True, the array will be modified in place. If false, a normalized copy will be
            returned.

    Returns:
        NDArray: The normalized array

    Raises:
        ZeroDivisionError: if the magnitude is sufficiently close to zero
    """
    if not inplace:
        x = np.copy(x)

    x -= np.mean(x, axis=-1, keepdims=True)

    # In case the values of x are very small, sometimes _magnitude can return 0, which would cause a divide by zero
    # error. Having already centred x, we can upscale it before downscaling it to avoid this issue.
    # If the _magnitude should actually be 0 (i.e. an error), this won't make a difference to that.
    if (_normalize_magnitude(x, axis=-1) == 0).any():
        x *= 1_000_000
    # If we STILL have a magnitude-0 vector, we will have a problem, so should raise the error immediately.
    with np.errstate(divide="raise"):
        x /= _normalize_magnitude(x, axis=-1)

    return x


def _normalize_magnitude(x: NDArray, axis: int) -> NDArray:
    """Reusable magnitude function for use in `normalize`."""
    return np.sqrt(np.sum(x**2, axis=axis, keepdims=True))


def window_stds_unnorm(x: NDArray, n: int):
    """
    Get the unnormalised standard deviation (std × sqrt(n)) on each position of a sliding window of width n.
    This can be used to correct for normalisation difference between each set of n samples.

    Args:
        x (NDArray): Input data. Expects size (channels, splits, timepoints)
        n (int): Width of the sliding window

    Returns:

    """
    n_chann, n_splits, _n_timepoints = x.shape

    # We'll use cumulative sums to make range sums easy (summing from a to b will be cumsum(b)-cumsum(a))
    y = np.concatenate((np.zeros((n_chann, n_splits, 1)), np.cumsum(x**2, axis=-1)), axis=-1)
    z = np.concatenate((np.zeros((n_chann, n_splits, 1)), np.cumsum(x, axis=-1)), axis=-1)

    # These subtractions give all distance-n cumsum differences (i.e. sums over all width-n sliding windows)
    y = y[:, :, -n - 1 : -1] - y[:, :, :n]
    z = z[:, :, -n - 1 : -1] - z[:, :, :n]

    stds = (y - ((z**2) / n)) ** 0.5
    return stds
