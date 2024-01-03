from numpy import mean, sum, sqrt
from numpy.typing import NDArray


def normalize(x: NDArray) -> NDArray:
    """
    Remove the mean and divide by the Euclidean magnitude.
    """
    x -= mean(x, axis=-1, keepdims=True)
    x /= sqrt(sum(x**2, axis=-1, keepdims=True))
    return x
