"""
Functions for dealing with iterables.
"""

from typing import Iterable


def all_equal(iterable: Iterable) -> bool:
    """All entries in Iterable are the same."""
    # Use `<= 1` here instead of `== 1` because we want the predicate to evaluate to true on an empty iterable.
    return len(set(iterable)) <= 1
