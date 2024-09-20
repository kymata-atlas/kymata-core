"""
Functions for dealing with iterables.
"""

from itertools import groupby
from typing import Collection

from numpy import ndarray
from numpy.typing import NDArray


def all_equal(sequence: Collection) -> bool:
    """
    Check if all elements in the sequence are the same.

    This function tests whether all elements in the provided sequence are equal. It handles special cases for
    sequences of length 0 or 1, and includes specific handling for numpy arrays due to their unique behavior
    with equality checks.

    Args:
        sequence (Collection): A sequence of elements to be checked for equality. This can be any iterable
                             including lists, tuples, or numpy arrays.

    Returns:
        bool: True if all elements in the sequence are equal, False otherwise.

    Notes:
        - An empty sequence returns True as there are no elements to compare (universal quantification over
          an empty set is true).
        - A sequence with one element returns True as the single element is trivially equal to itself.
        - For numpy arrays, the function ensures element-wise comparison using the `.all()` method to handle
          array equality correctly.
    """
    sequence = list(sequence)

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
