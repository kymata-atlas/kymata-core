import numpy as np
from numpy._typing import NDArray
from scipy.interpolate import splev


_XY = tuple[float, float]


def make_bspline_paths(spike_coordinate_pairs: list[tuple[_XY, _XY]]) -> list[list[NDArray]]:
    """
    Given a list of spike positions pairs, return a list of
    b-splines. First, find the control points, and second
    create the b-splines from these control points.

    Args:
        spike_coordinate_pairs (List[List[Tuple[float, float]]]): Each list contains the x-axis values and y-axis values
            for the start and end of a BSpline, e.g., [(0, 1), (1, 0)].

    Returns:
        List[List[np.array]]: A list of a list of np arrays. Each list contains two np.arrays. The first np.array contains
            the x-axis points and the second one contains the y-axis points. Together, they define a BSpline. Thus, it is
            a list of BSplines.
    """
    bspline_path_array = []
    for pair in spike_coordinate_pairs:

        ctr_pts = _make_bspline_ctr_points(pair)
        bspline_path_array.append(_make_bspline_path(ctr_pts))

    return bspline_path_array


def _make_bspline_ctr_points(start_and_end_node_coordinates: tuple[_XY, _XY]) -> NDArray:
    """
    Given the position of a start spike and an end spike, create
    a set of 6 control points needed for a b-spline.

    The first one and last one is the position of a start spike
    and an end spikes themselves, and the intermediate four are
    worked out using some simple rules.

    Args:
        start_and_end_node_coordinates (List[Tuple[float, float]]): List containing the start and end coordinates for one edge.
            First tuple is start, second is end. First element in tuple is x coord, second is y coord.

    Returns:
        np.array: A list of tuples of coordinates. Each coordinate pair represents a control point.
    """

    start_X, start_Y = start_and_end_node_coordinates[0]
    end_X, end_Y = start_and_end_node_coordinates[1]

    # allow for hexel radius offset for arrowhead.
    end_X = end_X - 0.003
    start_X = start_X - 0.003

    if end_X < start_X:
        # reverse BSpline
        start_X, end_X = end_X, start_X
        start_Y, end_Y = end_Y, start_Y

    # Offset points: chosen for aesthetics, but with a squish down to evenly-spaced when nodes are too small
    x_diff = end_X - start_X
    offsets = [
        min(0.005, 1 * x_diff / 5),
        min(0.010, 2 * x_diff / 5),
        min(0.020, 3 * x_diff / 5),
        min(0.030, 4 * x_diff / 5),
    ]

    ctr_points = np.array(
        [
            # start
            (start_X, start_Y),
            # first 2
            (start_X + offsets[0], start_Y),
            (start_X + offsets[1], start_Y),
            # second 2
            (start_X + offsets[2], end_Y),
            (start_X + offsets[3], end_Y),
            # end
            (end_X, end_Y),
        ]
    )

    return ctr_points


def _make_bspline_path(ctr_points: NDArray) -> list[NDArray]:
    """
    With an input of six control points, return an interpolated
    b-spline path which corresponds to a curved edge from one node to another.

    Args:
        ctr_points (NDArray): 2d NDArray containing the coordinates of the center points.

    Returns:
        List[NDArray]: A list of NDArrays that represent one BSpline path. The first list is a list of x-axis coordinates
            the second is a list of y-axis coordinates.
    """
    x = ctr_points[:, 0]
    y = ctr_points[:, 1]

    length = len(x)
    t = np.linspace(0, 1, length - 2, endpoint=True)
    t = np.append([0, 0, 0], t)
    t = np.append(t, [1, 1, 1])

    tck = [t, [x, y], 3]
    u3 = np.linspace(0, 1, (max(length * 2, 70)), endpoint=True)
    # Don't know why this is raising a warning
    # noinspection PyTypeChecker
    bspline_path: list[NDArray] = splev(u3, tck)

    return bspline_path
