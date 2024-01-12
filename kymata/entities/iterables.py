"""
Functions for dealing with iterables.
"""

from itertools import groupby
from typing import Sequence

from numpy import ndarray
from numpy.typing import NDArray


def all_equal(sequence: Sequence) -> bool:
    """All entries in Iterable are the same."""
    if len(sequence) == 0:
        # universal quantification over empty set is always true
        return True
    elif len(sequence) == 1:
        # One item is equal to itself
        return True
    elif isinstance(sequence[0], ndarray):
        # numpy arrays deal with equality weirdly

        # Check first two items are equal, and equal to the rest
        first: NDArray = sequence[0]
        if not isinstance(sequence[1], ndarray):
            return False
        second: NDArray = sequence[1]
        try:
            # noinspection PyUnresolvedReferences
            return (first == second).all() and all_equal(sequence[1:])
        except AttributeError:
            return False
    else:
        g = groupby(sequence)
        return next(g, True) and not next(g, False)
