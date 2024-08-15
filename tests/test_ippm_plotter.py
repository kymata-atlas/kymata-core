from unittest.mock import patch

from kymata.ippm.ippm_plotter import IPPMPlotter
from tests.test_ippm_data_tools import Node

import numpy as np

"""
    Testing Philosophy for IPPMPlotter
    ==================================
    It is a pain to test matplotlib plotting works correctly. Therefore, we will omit testing
    that code since it will be tested by actively using the IPPM module.

    On the other hand, rather than have no tests at all, we can still test the numerical computations
    even though they are private functions.
"""

test_graph = {
    "input": Node(100, (0, 0.2), []),
    "func1-0": Node(280, (10, 0.4), ["input"]),
    "func1-1": Node(790, (25, 0.4), ["func1-0"]),
    "func2-0": Node(610, (50, 0.6), ["input", "func1-1"]),
    "func3-0": Node(920, (60, 0.8), ["func2-0"]),
    "func3-1": Node(120, (65, 0.8), ["func3-0"]),
    "func4-0": Node(420, (70, 1), ["func3-1"]),
}
test_colors = {
    "func1": "#023eff",
    "func2": "#0ff7c00",
    "func3": "#1ac938",
    "func4": "#e8000b",
    "input": "#a201e9",
}


@patch("kymata.ippm.ippm_plotter.splev")
def test_IPPMPlotter_MakeBSplinePaths_Successfully(mock_splev):
    expected_b_spline_paths = [
        np.array(range(65, 100, 10)),
        np.array(np.linspace(0.8, 1, 10)),
    ]
    mock_splev.return_value = expected_b_spline_paths
    ctr_points = np.array([[65, 0.8], [85, 0.8], [90, 0.8], [95, 1], [100, 1]])
    plotter = IPPMPlotter()
    actual_bspline_paths = plotter._make_b_spline_path(ctr_points)
    assert actual_bspline_paths == expected_b_spline_paths


def test_IPPMPlotter_MakeBSplineCtrPoints_Successfully():
    coords = [[65, 0.8], [70, 1]]
    plotter = IPPMPlotter()
    actual_ctr_points = plotter._make_b_spline_ctr_points(coords)
    expected_ctr_points = [[65, 0.8], [85, 0.8], [90, 0.8], [95, 1], [100, 1], [70, 1]]

    for expected_point, actual_point in zip(expected_ctr_points, actual_ctr_points):
        assert expected_point[0] == actual_point[0]
        assert expected_point[1] == actual_point[1]


@patch("kymata.ippm.ippm_plotter.splev")
def test_IPPMPlotter_MakeBSplinePath_Successfully(mock_splev):
    mock_splev.return_value = [
        np.array(range(65, 100, 10)),
        np.array(np.linspace(0.8, 1, 10)),
    ]
    pairs = [[(65, 0.8), (70, 1)]]
    plotter = IPPMPlotter()
    actual_b_splines = plotter._make_b_spline_paths(pairs)
    expected_b_spline_paths = [
        np.array(range(65, 100, 10)),
        np.array(np.linspace(0.8, 1, 10)),
    ]

    for actual_path_x_coord, expected_path_x_coord in zip(
        actual_b_splines[0][0], expected_b_spline_paths[0]
    ):
        assert actual_path_x_coord == expected_path_x_coord
    for actual_path_y_coord, expected_path_y_coord in zip(
        actual_b_splines[0][1], expected_b_spline_paths[1]
    ):
        assert actual_path_y_coord == expected_path_y_coord
