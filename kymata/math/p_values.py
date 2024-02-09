from numpy import log10
from numpy.typing import ArrayLike


# The canonical log base
log_base = 10


def p_to_logp(arraylike: ArrayLike) -> ArrayLike:
    """The one-stop-shop for converting from p-values to log p-values."""
    return log10(arraylike)


def logp_to_p(arraylike: ArrayLike) -> ArrayLike:
    """The one-stop-shop for converting from log p-values to p-values."""
    return float(10) ** arraylike
