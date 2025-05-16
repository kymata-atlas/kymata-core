from copy import deepcopy, copy

import pytest

from kymata.entities.expression import ExpressionPoint
from kymata.ippm.graph import IPPMGraph, input_stream_pseudo_expression_point, IPPMConnectionStyle
from kymata.ippm.hierarchy import TransformHierarchy, CandidateTransformList


@pytest.fixture
def sample_hierarchy() -> TransformHierarchy:
    """
    A sample hierarchy to match with `sample_points`.

             func1
          ↗        ↘
    input —————————→ func2 → func3 → func4
    """
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
    graph = IPPMGraph(sample_hierarchy, sample_points)

    assert graph.transforms == {"input", "func1", "func2", "func3", "func4"}
    assert graph._points_by_transform == {
        "func1": [ExpressionPoint("c", 10, "func1", -28),
                  ExpressionPoint("c", 25, "func1", -79)],
        "func2": [ExpressionPoint("c", 50, "func2", -61)],
        "func3": [ExpressionPoint("c", 60, "func3", -92),
                  ExpressionPoint("c", 65, "func3", -12)],
        "func4": [ExpressionPoint("c", 70, "func4", -42)],
    }
    assert graph.inputs == {"input"}
    assert graph.terminals == {"func4"}

    # `graph.points` already tested as correct
    assert (
            set(graph.graph_full.successors(input_stream_pseudo_expression_point("input")))
            == set(graph._points_by_transform["func1"] + graph._points_by_transform["func2"])
    )

    assert graph.serial_sequence == [
        ["input"],
        ["func1"],
        ["func2"],
        ["func3"],
        ["func4"],
    ]


def test_ippmgraph_empty_points_builds_successfully(sample_hierarchy, empty_points):
    graph = IPPMGraph(sample_hierarchy, empty_points)

    assert graph.transforms == {"input"}
    assert graph.inputs == {"input"}
    assert graph.terminals == {"input"}
    assert len(graph._points_by_transform) == 0


def test_ippmgraph_empty_hierarchy_builds_successfully(empty_hierarchy, empty_points):
    graph = IPPMGraph(empty_hierarchy, empty_points)

    assert graph.transforms == set()
    assert graph.inputs == set()
    assert graph.terminals == set()
    assert len(graph._points_by_transform) == 0


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
    graph_1 = IPPMGraph(deepcopy(ctl), deepcopy(points))
    graph_2 = IPPMGraph(deepcopy(ctl), deepcopy(points))
    assert copy(graph_1) == graph_2


def test_ippmgraph_missing_points(sample_hierarchy, sample_points):
    graph = IPPMGraph(sample_hierarchy,
                      # Delete func1
                      [p for p in sample_points if p.transform != "func1"])

    assert graph.transforms == {"input", "func2", "func3", "func4"}
    assert graph._points_by_transform == {
        "func2": [ExpressionPoint("c", 50, "func2", -61)],
        "func3": [ExpressionPoint("c", 60, "func3", -92),
                  ExpressionPoint("c", 65, "func3", -12)],
        "func4": [ExpressionPoint("c", 70, "func4", -42)],
    }
    assert graph.inputs == {"input"}
    assert graph.terminals == {"func4"}

    # `graph.points` already tested as correct
    assert (
            set(graph.graph_full.successors(input_stream_pseudo_expression_point("input")))
            == set(graph._points_by_transform["func2"])
    )

    assert graph.serial_sequence == [
        ["input"],
        ["func2"],
        ["func3"],
        ["func4"],
    ]


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
    graph = IPPMGraph(ctl=ctl, points=points)
    ftl = graph.graph_last_to_first
    inputs = graph.inputs
    for p in points:
        assert p in ftl.nodes
    for inp in inputs:
        assert inp in {n.transform for n in ftl.nodes}
    assert set(ftl.successors(input_stream_pseudo_expression_point("in"))) == {ExpressionPoint("c1", 1, "A", -50)}
    assert set(ftl.successors(ExpressionPoint("c1", 1, "A", -50))) == {ExpressionPoint("c2", 2, "A", -50)}
    assert set(ftl.successors(ExpressionPoint("c2", 2, "A", -50))) == {ExpressionPoint("c3", 3, "A", -50)}
    assert set(ftl.successors(ExpressionPoint("c3", 3, "A", -50))) == {ExpressionPoint("c4", 4, "B", -50),
                                                                       ExpressionPoint("c5", 5, "C", -50)}
    assert set(ftl.successors(ExpressionPoint("c4", 4, "B", -50))) == set()
    assert set(ftl.successors(ExpressionPoint("c5", 5, "C", -50))) == set()


def test_ippmgraph_points_for_transform(sample_hierarchy, sample_points):
    graph = IPPMGraph(sample_hierarchy, sample_points)

    assert graph.points_for_transform("func1") == [
        ExpressionPoint("c", 10, "func1", -28),
        ExpressionPoint("c", 25, "func1", -79),
    ]
    assert graph.points_for_transform("func2") == [
        ExpressionPoint("c", 50, "func2", -61),
    ]
    assert graph.points_for_transform("func3") == [
        ExpressionPoint("c", 60, "func3", -92),
        ExpressionPoint("c", 65, "func3", -12),
    ]
    assert graph.points_for_transform("func4") == [
        ExpressionPoint("c", 70, "func4", -42),
    ]


def test_ippmgraph_edges_for_transform(sample_hierarchy, sample_points):
    graph = IPPMGraph(sample_hierarchy, sample_points)

    connection_style = IPPMConnectionStyle.first_to_first
    assert graph.edges_between_transforms("func1", "func2", connection_style) == [
        (
            ExpressionPoint("c", 10, "func1", -28),
            ExpressionPoint("c", 50, "func2", -61),
        ),
    ]
    assert graph.edges_between_transforms("func1", "func3", connection_style) == []
    assert graph.edges_between_transforms("func1", "func4", connection_style) == []
    assert graph.edges_between_transforms("func2", "func3", connection_style) == [
        (
            ExpressionPoint("c", 50, "func2", -61),
            ExpressionPoint("c", 60, "func3", -92),
        ),
    ]
    assert graph.edges_between_transforms("func2", "func4", connection_style) == []
    assert graph.edges_between_transforms("func3", "func4", connection_style) == [
        (
            ExpressionPoint("c", 60, "func3", -92),
            ExpressionPoint("c", 70, "func4", -42),
        ),
    ]
