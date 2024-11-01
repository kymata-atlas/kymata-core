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


def p_to_surprisal(arraylike: ArrayLike) -> ArrayLike:
    """Converts p-values to surprisal values."""
    return logp_to_surprisal(p_to_logp(arraylike))


def logp_to_surprisal(arraylike: ArrayLike) -> ArrayLike:
    """Converts logp-values to surprisal values."""
    return -1 * arraylike


def surprisal_to_logp(arraylike: ArrayLike) -> ArrayLike:
    """Converts surprisal values to logp values."""
    return -1 * arraylike


def surprisal_to_p(arraylike: ArrayLike) -> ArrayLike:
    """Converts surprisal values to p-values."""
    return logp_to_p(surprisal_to_logp(arraylike))
