import numpy as np
from numpy.typing import NDArray


def normalize(x: NDArray, inplace: bool = False) -> NDArray:
    """
    Remove the mean and divide by the Euclidean magnitude.
    If inplace is True, the array will be modified in place. If false, a normalized copy will be returned.
    """
    if not inplace:
        x = np.copy(x)

    x -= np.mean(x, axis=-1, keepdims=True)

    # In case the values of x are very small, sometimes _magnitude can return 0, which would cause a divide by zero
    # error. Having already centred x, we can upscale it before downscaling it to avoid this issue. In case the
    # _magnitude should actually be 0, this won't make a difference to that.
    if (_magnitude(x) == 0).all():
        x *= 1_000_000
    x /= _magnitude(x)

    return x


def _magnitude(x: NDArray) -> NDArray:
    return np.sqrt(np.sum(x**2, axis=-1, keepdims=True))


def get_stds(x: NDArray, n: int):
    """
    Get the stds (times sqrt(n)) to correct for normalisation difference between each set of n_samples
    """
    d0, d1, d2 = x.shape
    y = np.concatenate((np.zeros((d0, d1, 1)), np.cumsum(x**2, axis=-1)), axis=-1)
    z = np.concatenate((np.zeros((d0, d1, 1)), np.cumsum(x, axis=-1)), axis=-1)
    y = y[:,:,-n-1:-1] - y[:,:,:n]
    z = z[:,:,-n-1:-1] - z[:,:,:n]
    return (y - ((z**2) / n)) ** 0.5
