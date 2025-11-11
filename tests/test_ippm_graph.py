from copy import deepcopy, copy

import pytest

from kymata.entities.expression import ExpressionPoint, BLOCK_SCALP
from kymata.ippm.graph import IPPMGraph, _node_id_from_point
from kymata.ippm.hierarchy import TransformHierarchy, CandidateTransformList


@pytest.fixture
def sample_hierarchy() -> TransformHierarchy:
    """A sample hierarchy to match with `sample_points`."""
    return {
        "input": [],
        "func1": ["input"],
        "func2": ["input", "func1"],
        "func3": ["func2"],
        "func4": ["func3"],
    }


@pytest.fixture
def empty_hierarchy() -> TransformHierarchy:
    """An empty hierarchy with no transforms."""
    return dict()


@pytest.fixture
def sample_points() -> list[ExpressionPoint]:
    """A sample list of points to match with `sample_hierarchy`."""
    return [
        ExpressionPoint("c", 10, "func1", -28),
        ExpressionPoint("c", 25, "func1", -79),
        ExpressionPoint("c", 50, "func2", -61),
        ExpressionPoint("c", 60, "func3", -92),
        ExpressionPoint("c", 65, "func3", -12),
        ExpressionPoint("c", 70, "func4", -42),
    ]


@pytest.fixture
def empty_points() -> list[ExpressionPoint]:
    """Empty list of points."""
    return []


def test_ippmgraph_build_successfully(sample_hierarchy, sample_points):
    graph = IPPMGraph(sample_hierarchy, points_by_block={"scalp": sample_points})

    # _points_by_transform stores the *original ExpressionPoint objects* grouped by block and transform
    assert graph._points_by_transform["scalp"] == {
        "func1": [ExpressionPoint("c", 10, "func1", -28),
                  ExpressionPoint("c", 25, "func1", -79)],
        "func2": [ExpressionPoint("c", 50, "func2", -61)],
        "func3": [ExpressionPoint("c", 60, "func3", -92),
                  ExpressionPoint("c", 65, "func3", -12)],
        "func4": [ExpressionPoint("c", 70, "func4", -42)],
    }
    assert graph.transforms == {"input", "func1", "func2", "func3", "func4"}
    assert graph.inputs == {"input"}
    assert graph.terminals == {"func4"}

    # Get the actual input node created by the IPPMGraph constructor
    input_nodes_in_graph = [n for n in graph.graph_full.nodes if n.is_input and n.transform == "input"]
    assert len(input_nodes_in_graph) == 1
    pseudo_input_node = input_nodes_in_graph[0]

    # Helper to get IPPMNode from ExpressionPoint for assertions involving graph edges
    def get_ippm_node_from_point(point: ExpressionPoint, block: str):
        # We need to find the exact IPPMNode created by the graph for this ExpressionPoint.
        # This involves matching transform, latency, and hemisphere.
        return next(n for n in graph.graph_full.nodes
                    if n.transform == point.transform and n.latency == point.latency and n.hemisphere == block)

    # Convert ExpressionPoints in assertions to the actual IPPMNodes in the graph
    node_func1_10 = get_ippm_node_from_point(ExpressionPoint("c", 10, "func1", -28), BLOCK_SCALP)
    node_func1_25 = get_ippm_node_from_point(ExpressionPoint("c", 25, "func1", -79), BLOCK_SCALP)
    node_func2_50 = get_ippm_node_from_point(ExpressionPoint("c", 50, "func2", -61), BLOCK_SCALP)

    assert (
            set(graph.graph_full.successors(pseudo_input_node))
            == {node_func1_10, node_func1_25, node_func2_50}
    )

    assert graph.serial_sequence == (
        {"input"},
        {"func1"},
        {"func2"},
        {"func3"},
        {"func4"},
    )


def test_ippmgraph_empty_points_builds_successfully(sample_hierarchy, empty_points):
    graph = IPPMGraph(sample_hierarchy, points_by_block=dict(scalp=empty_points))

    assert graph.transforms == {"input"}
    assert graph.inputs == {"input"}
    assert graph.terminals == {"input"}
    assert len(graph._points_by_transform["scalp"]) == 0


def test_ippmgraph_empty_hierarchy_builds_successfully(empty_hierarchy, empty_points):
    graph = IPPMGraph(empty_hierarchy, points_by_block=dict(scalp=empty_points))

    assert graph.transforms == set()
    assert graph.inputs == set()
    assert graph.terminals == set()
    assert len(graph._points_by_transform["scalp"]) == 0


def test_ippmgraph_copy():
    ctl = CandidateTransformList({
        "in": [],
        "A": ["in"],
        "B": ["A"],
        "C": ["A"],
    })
    points = [
        ExpressionPoint("c1", 1, "A", -50),
        ExpressionPoint("c2", 2, "A", -50),
        ExpressionPoint("c3", 3, "A", -50),
        ExpressionPoint("c4", 4, "B", -50),
        ExpressionPoint("c5", 5, "C", -50),
    ]
    graph_1 = IPPMGraph(deepcopy(ctl), points_by_block={"scalp": deepcopy(points)})
    graph_2 = copy(graph_1) # Use copy() directly from the object

    assert graph_1 == graph_2 # This now uses the fixed __eq__ and __copy__


def test_ippmgraph_missing_points(sample_hierarchy, sample_points):
    # Filter points to simulate missing 'func1' data
    filtered_points = [p for p in sample_points if p.transform != "func1"]
    graph = IPPMGraph(sample_hierarchy, points_by_block={"scalp": filtered_points})

    # _points_by_transform stores the *original ExpressionPoint objects* grouped by block and transform
    assert graph._points_by_transform["scalp"] == {
        "func2": [ExpressionPoint("c", 50, "func2", -61)],
        "func3": [ExpressionPoint("c", 60, "func3", -92),
                  ExpressionPoint("c", 65, "func3", -12)],
        "func4": [ExpressionPoint("c", 70, "func4", -42)],
    }
    assert graph.transforms == {"input", "func2", "func3", "func4"}
    assert graph.inputs == {"input"}
    assert graph.terminals == {"func4"}

    # Get the actual input node created by the IPPMGraph constructor
    input_nodes_in_graph = [n for n in graph.graph_full.nodes if n.is_input and n.transform == "input"]
    assert len(input_nodes_in_graph) == 1
    pseudo_input_node = input_nodes_in_graph[0]

    # Helper to get IPPMNode from ExpressionPoint for assertions involving graph edges
    def get_ippm_node_from_point(point: ExpressionPoint, block: str):
        return next(n for n in graph.graph_full.nodes
                    if n.transform == point.transform and n.latency == point.latency and n.hemisphere == block)

    # Convert ExpressionPoints in assertions to the actual IPPMNodes in the graph
    node_func2_50 = get_ippm_node_from_point(ExpressionPoint("c", 50, "func2", -61), BLOCK_SCALP)

    assert (
            set(graph.graph_full.successors(pseudo_input_node))
            == {node_func2_50} # Ensure we are comparing IPPMNode objects
    )

    assert graph.serial_sequence == (
        {"input"},
        {"func2"},
        {"func3"},
        {"func4"},
    )


def test_ippmgraph_last_to_first():
    ctl = CandidateTransformList({
        "in": [],
        "A": ["in"],
        "B": ["A"],
        "C": ["A"],
    })
    points = [
        ExpressionPoint("c1", 1, "A", -50),
        ExpressionPoint("c2", 2, "A", -50),
        ExpressionPoint("c3", 3, "A", -50),
        ExpressionPoint("c4", 4, "B", -50),
        ExpressionPoint("c5", 5, "C", -50),
    ]
    graph = IPPMGraph(ctl=ctl, points_by_block=dict(scalp=points))
    ftl = graph.graph_last_to_first
    inputs = graph.inputs
    for p in points:
        # Check if the IPPMNode corresponding to the ExpressionPoint is in ftl.nodes
        # This requires reconstructing the IPPMNode as the node_id now includes block information
        node_id_for_point = _node_id_from_point(p, BLOCK_SCALP, None)
        assert any(n.node_id == node_id_for_point for n in ftl.nodes)

    for inp in inputs:
        assert inp in {n.transform for n in ftl.nodes}

    # Get the actual input node from the graph
    input_nodes_in_graph = [n for n in graph.graph_full.nodes if n.is_input and n.transform == "in"]
    assert len(input_nodes_in_graph) == 1
    pseudo_input_node = input_nodes_in_graph[0]

    # Helper to get IPPMNode from ExpressionPoint for assertions
    def get_ippm_node(point: ExpressionPoint, block: str, graph_obj: IPPMGraph):
        return next(n for n in graph_obj.graph_full.nodes if n.transform == point.transform and n.latency == point.latency and n.hemisphere == block)

    # Convert ExpressionPoints in assertions to the actual IPPMNodes in the graph_full
    # Then use these nodes to check successors in ftl
    node_c1_A = get_ippm_node(ExpressionPoint("c1", 1, "A", -50), BLOCK_SCALP, graph)
    node_c2_A = get_ippm_node(ExpressionPoint("c2", 2, "A", -50), BLOCK_SCALP, graph)
    node_c3_A = get_ippm_node(ExpressionPoint("c3", 3, "A", -50), BLOCK_SCALP, graph)
    node_c4_B = get_ippm_node(ExpressionPoint("c4", 4, "B", -50), BLOCK_SCALP, graph)
    node_c5_C = get_ippm_node(ExpressionPoint("c5", 5, "C", -50), BLOCK_SCALP, graph)


    assert set(ftl.successors(pseudo_input_node)) == {node_c1_A}
    assert set(ftl.successors(node_c1_A)) == {node_c2_A}
    assert set(ftl.successors(node_c2_A)) == {node_c3_A}
    assert set(ftl.successors(node_c3_A)) == {node_c4_B, node_c5_C}
    assert set(ftl.successors(node_c4_B)) == set()
    assert set(ftl.successors(node_c5_C)) == set()