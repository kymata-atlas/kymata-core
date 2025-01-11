import pytest

from kymata.entities.expression import ExpressionPoint
from kymata.ippm.graph import IPPMGraph, input_stream_pseudo_expression_point
from kymata.ippm.hierarchy import TransformHierarchy


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
    graph = IPPMGraph(sample_hierarchy, sample_points)

    assert graph.transforms == {"input", "func1", "func2", "func3", "func4"}
    assert graph.points == {
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
            == set(graph.points["func1"] + graph.points["func2"])
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
    assert len(graph.points) == 0


def test_ippmgraph_empty_hierarchy_builds_successfully(empty_hierarchy, empty_points):
    graph = IPPMGraph(empty_hierarchy, empty_points)

    assert graph.transforms == set()
    assert graph.inputs == set()
    assert graph.terminals == set()
    assert len(graph.points) == 0


def test_ippmgraph_missing_points(sample_hierarchy, sample_points):
    graph = IPPMGraph(sample_hierarchy,
                      # Delete func1
                      [p for p in sample_points if p.transform != "func1"])

    assert graph.transforms == {"input", "func2", "func3", "func4"}
    assert graph.points == {
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
            == set(graph.points["func2"])
    )

    assert graph.serial_sequence == [
        ["input"],
        ["func2"],
        ["func3"],
        ["func4"],
    ]
