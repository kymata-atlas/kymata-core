"""
Functions for dealing with iterables.
"""

from itertools import groupby
from typing import Iterable


def all_equal(iterable: Iterable) -> bool:
    """All entries in Iterable are the same."""
    # Thanks https://stackoverflow.com/a/3844832/2883198
    g = groupby(iterable)
    return next(g, True) and not next(g, False)
