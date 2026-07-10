from typing import NamedTuple, TypeVar, Hashable, Any, Callable, Mapping


class Point2d(NamedTuple):
    """
    A point in 2d space with an x ordinate and a y ordinate
    """
    x: float
    y: float


T_coerce = TypeVar("T_coerce")
T_default = TypeVar("T_default")


def get_coerce(
        d: Mapping[Hashable, Any],
        key: Hashable,
        coerce: Callable[[Any], T_coerce],
        default: T_default,
) -> T_default | T_coerce:
    """
    Return `coerce(d[key])` if `key` exists, otherwise `default`.
    """
    if key in d:
        return coerce(d[key])
    else:
        return default
