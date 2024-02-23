import math


def round_down(val: float, interval: int):
    """
    Rounds down to the nearest multiple of `multiple`.
    E.g. round_down(13, 10) = 10
    """
    return int(math.floor(val / interval)) * interval


def round_up(val: float, interval: int):
    """
    Rounds up to the nearest multiple of `multiple`.
    E.g. round_up(13, 10) = 20
    """
    return int(math.ceil(val / interval)) * interval
