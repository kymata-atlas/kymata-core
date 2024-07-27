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
    x /= np.sqrt(np.sum(x**2, axis=-1, keepdims=True))
    return x


def get_stds(x, n):
    """
    Get the stds (times sqrt(n)) to correct for normalisation difference between each set of n_samples
    """
    d0, d1, d2 = x.shape
    y = np.concatenate((np.zeros((d0, d1, 1)), np.cumsum(x**2, axis=-1)), axis=-1)
    z = np.concatenate((np.zeros((d0, d1, 1)), np.cumsum(x, axis=-1)), axis=-1)
    y = y[:,:,-n-1:-1] - y[:,:,:n]
    z = z[:,:,-n-1:-1] - z[:,:,:n]
    return (y - ((z**2) / n)) ** 0.5
