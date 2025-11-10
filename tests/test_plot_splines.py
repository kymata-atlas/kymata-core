"""
Testing Philosophy for plot.ippmting functionality
==================================
It is a pain to test matplotlib plotting works correctly. Therefore, we will omit testing
that code since it will be tested by actively using the IPPM module.

On the other hand, rather than have no tests at all, we can still test the numerical computations
even though they are private functions.
"""
from math import isclose
from unittest.mock import patch

import numpy as np

from kymata.plot.splines import make_bspline_paths, _make_bspline_ctr_points, _make_bspline_path


@patch("kymata.plot.ippm.splev")
def test_IPPMPlotter_MakeBSplinePaths_Successfully(mock_splev):
    expected_b_spline_paths = [
        np.array(range(65, 100, 10)),
        np.array(np.linspace(0.8, 1, 10)),
    ]
    mock_splev.return_value = expected_b_spline_paths
    ctr_points = np.array(
        [
            (65, 0.8),
            (70, 0.8),
            (80, 0.8),
            (85, 1),
            (95, 1),
        ]
    )
    actual_bspline_paths = _make_bspline_path(ctr_points)
    assert actual_bspline_paths == expected_b_spline_paths


def test_IPPMPlotter_MakeBSplineCtrPoints_Successfully():
    coords = [(0.05, 0.8), (0.09, 1)]
    actual_ctr_points = _make_bspline_ctr_points(coords)
    expected_ctr_points = [(0.05, 0.8), (0.055, 0.8), (0.06, 0.8), (0.07, 1), (0.08, 1), (0.09, 1)]

    for expected_point, actual_point in zip(expected_ctr_points, actual_ctr_points):
        assert isclose(expected_point[0], actual_point[0])
        assert isclose(expected_point[1], actual_point[1])


@patch("kymata.plot.ippm.splev")
def test_IPPMPlotter_MakeBSplinePath_Successfully(mock_splev):
    mock_splev.return_value = [
        np.array(range(65, 100, 10)),
        np.array(np.linspace(0.8, 1, 10)),
    ]
    pairs = [[(65, 0.8), (70, 1)]]
    actual_b_splines = make_bspline_paths(pairs)
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
