from copy import deepcopy

import numpy as np

from kymata.entities.constants import HEMI_RIGHT
from kymata.entities.expression import ExpressionPoint
from kymata.ippm.build import IPPMBuilder

test_hexels = {
    "func1": [ExpressionPoint("c", 10, "func1", -28), ("c", 25, "func1", -79)],
    "func2": [ExpressionPoint("c", 50, "func2", -61)],
    "func3": [ExpressionPoint("c", 60, "func3", -92), ExpressionPoint("c", 65, "func3", -12)],
    "func4": [ExpressionPoint("c", 70, "func4", -42)],
}
test_hierarchy = {
    "input": [],
    "func1": ["input"],
    "func2": ["input", "func1"],
    "func3": ["func2"],
    "func4": ["func3"],
}
test_inputs = ["input"]
test_hemi = HEMI_RIGHT


def map_mag_to_size(x):
    return -10 * np.log10(x)


def test_IPPMBuilder_BuildGraph_Successfully():
    builder = IPPMBuilder(test_hexels, test_inputs, test_hierarchy, test_hemi)
    expected_graph = {
        "input": (100, "abc", (0, 0.2), []),
        "func1-0": (map_mag_to_size(1e-28), "#023eff", (10, 0.4), ["input"]),
        "func1-1": (map_mag_to_size(1e-79), "#023eff", (25, 0.4), ["func1-0"]),
        "func2-0": (
            map_mag_to_size(1e-61),
            "#0ff7c00",
            (50, 0.6),
            ["input", "func1-1"],
        ),
        "func3-0": (map_mag_to_size(1e-92), "#1ac938", (60, 0.8), ["func2-0"]),
        "func3-1": (map_mag_to_size(1e-12), "#1ac938", (65, 0.8), ["func3-0"]),
        "func4-0": (map_mag_to_size(1e-42), "#e8000b", (70, 1), ["func3-1"]),
    }
    actual_graph = builder.graph

    assert set(actual_graph.keys()) == set(expected_graph.keys())
    for node, val in actual_graph.items():
        assert val.magnitude == expected_graph[node][0]
        assert (val.position[0], round(val.position[1], 1)) == expected_graph[node][2]
        assert val.inc_edges == expected_graph[node][3]


def test_IPPMBuilder_BuildGraph_EmptyHexels_Successfully():
    empty_hexels = {}
    builder = IPPMBuilder(empty_hexels, test_inputs, test_hierarchy, test_hemi)
    expected_graph = {"input": (100, "abc", (0, 0.2), [])}
    actual_graph = builder.graph

    assert set(actual_graph.keys()) == set(expected_graph.keys())
    for node, val in actual_graph.items():
        assert val.magnitude == expected_graph[node][0]
        assert (val.position[0], round(val.position[1], 1)) == expected_graph[node][2]
        assert val.inc_edges == expected_graph[node][3]


def test_IPPMBuilder_BuildGraph_EmptyHierarchy_Successfully():
    empty_hierarchy = {}
    builder = IPPMBuilder(test_hexels, test_inputs, empty_hierarchy, test_hemi)
    expected_graph = {}
    actual_graph = builder.graph

    assert actual_graph == expected_graph


def test_IPPMBuilder_BuildGraph_EmptyInputs_Successfully():
    empty_inputs = []
    builder = IPPMBuilder(test_hexels, empty_inputs, test_hierarchy, test_hemi)
    expected_graph = {
        "func1-0": (map_mag_to_size(1e-28), "#023eff", (10, 0.4), []),
        "func1-1": (map_mag_to_size(1e-79), "#023eff", (25, 0.4), ["func1-0"]),
        "func2-0": (map_mag_to_size(1e-61), "#0ff7c00", (50, 0.6), ["func1-1"]),
        "func3-0": (map_mag_to_size(1e-92), "#1ac938", (60, 0.8), ["func2-0"]),
        "func3-1": (map_mag_to_size(1e-12), "#1ac938", (65, 0.8), ["func3-0"]),
        "func4-0": (map_mag_to_size(1e-42), "#e8000b", (70, 1), ["func3-1"]),
    }
    actual_graph = builder.graph

    assert set(actual_graph.keys()) == set(expected_graph.keys())
    for node, val in actual_graph.items():
        assert val.magnitude == expected_graph[node][0]
        assert (val.position[0], round(val.position[1], 1)) == expected_graph[node][2]
        assert val.inc_edges == expected_graph[node][3]


def test_IPPMBuilder_BuildGraph_MissingFunctionsInHexels_Successfully():
    mismatched_hexels = deepcopy(test_hexels)
    mismatched_hexels.pop("func2")
    builder = IPPMBuilder(mismatched_hexels, test_inputs, test_hierarchy, test_hemi)
    expected_graph = {
        "input": (100, "abc", (0, 0.2), []),
        "func1-0": (map_mag_to_size(1e-28), "#023eff", (10, 0.4), ["input"]),
        "func1-1": (map_mag_to_size(1e-79), "#023eff", (25, 0.4), ["func1-0"]),
        "func3-0": (map_mag_to_size(1e-92), "#1ac938", (60, 0.8), []),
        "func3-1": (map_mag_to_size(1e-12), "#1ac938", (65, 0.8), ["func3-0"]),
        "func4-0": (map_mag_to_size(1e-42), "#e8000b", (70, 1), ["func3-1"]),
    }
    actual_graph = builder.graph

    assert set(actual_graph.keys()) == set(expected_graph.keys())
    for node, val in actual_graph.items():
        assert val.magnitude == expected_graph[node][0]
        assert (val.position[0], round(val.position[1], 1)) == expected_graph[node][2]
        assert val.inc_edges == expected_graph[node][3]
