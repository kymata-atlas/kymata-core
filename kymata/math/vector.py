import numpy as np
from numpy.typing import NDArray


def normalize(x: NDArray, eps: float = 1e-7) -> NDArray:
    """
    Remove the mean and divide by the Euclidean magnitude.
    """
    x -= np.mean(x, axis=-1, keepdims=True)
    x /= np.sqrt(np.sum(x**2, axis=-1, keepdims=True)) + eps
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
