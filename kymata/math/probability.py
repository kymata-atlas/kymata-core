from __future__ import annotations

from statistics import NormalDist

from numpy import log10, longdouble
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


def p_threshold_for_sigmas(sigmas: float) -> float:
    """Threshold for "n-sigma" tests."""
    assert sigmas > 0
    return 1 - NormalDist(mu=0, sigma=1).cdf(sigmas)


def bonferroni_correct(alpha: float, n_comparisons: int) -> float:
    """Applies Bonferroni correction to an alpha threshold."""
    return 1 - ((1 - alpha) ** n_comparisons)


def sidak_correct(alpha: float, n_comparisons: int) -> float:
    """Applies Šidák correction to an alpha threshold."""
    return 1 - (
        (1 - alpha)
        ** longdouble(1 / (2 * n_comparisons))
    )
