import numpy as np
from numpy.typing import NDArray


def normalize(x: NDArray, inplace: bool = False, eps: float = 1e-7) -> NDArray:
    """
    Remove the mean and divide by the Euclidean magnitude.
    If inplace is True, the array will be modified in place. If false, a normalized copy will be returned.
    raises: ZeroDivisionError
    """
    if not inplace:
        x = np.copy(x)

    x -= np.mean(x, axis=-1, keepdims=True)

    # In case the values of x are very small, sometimes _magnitude can return 0, which would cause a divide by zero
    # error. Having already centred x, we can upscale it before downscaling it to avoid this issue.
    # If the _magnitude should actually be 0 (i.e. an error), this won't make a difference to that.
    if (_normalize_magnitude(x) == 0).any():
        x *= 1_000_000
    # If we STILL have a magnitude-0 vector, we will have a problem, so should raise the error immediately.
    with np.errstate(divide="raise"):
        x /= _normalize_magnitude(x) + eps

    return x


def _normalize_magnitude(x: NDArray) -> NDArray:
    """Reusable magnitude function for use in `normalize`."""
    return np.sqrt(np.sum(x**2, axis=-1, keepdims=True))


def get_stds(x: NDArray, n: int):
    """
    Get the stds (times sqrt(n)) to correct for normalisation difference between each set of n_samples
    """
    d0, d1, d2 = x.shape
    y = np.concatenate((np.zeros((d0, d1, 1)), np.cumsum(x**2, axis=-1)), axis=-1)
    z = np.concatenate((np.zeros((d0, d1, 1)), np.cumsum(x, axis=-1)), axis=-1)
    y = y[:, :, -n - 1 : -1] - y[:, :, :n]
    z = z[:, :, -n - 1 : -1] - z[:, :, :n]
    return (y - ((z**2) / n)) ** 0.5
