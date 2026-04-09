import numpy as np
from numpy.typing import NDArray


def normalize(x: NDArray, eps: float = 1e-7) -> NDArray:
    """
    Remove the mean and divide by the Euclidean magnitude.
    """
    x -= np.mean(x, axis=-1, keepdims=True)
    x /= np.sqrt(np.sum(x**2, axis=-1, keepdims=True)) + eps
    return x


def get_stds(x, n, eps: float = 1e-5):
    """
    Get the stds (times sqrt(n)) to correct for normalisation difference between each set of n_samples
    """
    d0, d1, d2 = x.shape
    y = np.concatenate((np.zeros((d0, d1, 1)), np.cumsum(x**2, axis=-1)), axis=-1)
    z = np.concatenate((np.zeros((d0, d1, 1)), np.cumsum(x, axis=-1)), axis=-1)
    y = y[:, :, -n - 1 : -1] - y[:, :, :n]
    z = z[:, :, -n - 1 : -1] - z[:, :, :n]
    return (y - ((z**2) / n) + eps) ** 0.5


def index_in(v1: NDArray, v2: NDArray) -> NDArray:
    """
    For each element of v1, returns its index in v2.
    Raises a ValueError if an element of v1 is not found in v2.

    (Thanks https://stackoverflow.com/a/8251757)
    """
    index_v2 = np.argsort(v2)
    sorted_v2 = v2[index_v2]
    sorted_index = np.searchsorted(sorted_v2, v1)

    v1_index = np.take(index_v2, sorted_index, mode="clip")
    mask = v2[v1_index] != v1

    result = v1_index.copy()
    result[mask] = -1
    return result
