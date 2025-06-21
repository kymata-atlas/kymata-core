from math import inf

import pytest

from kymata.entities.expression import ExpressionPoint
from kymata.io.json import serialise_graph
from kymata.ippm.graph import IPPMGraph
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
def sample_points() -> list[ExpressionPoint]:
    """A sample list of points to match with `sample_hierarchy`."""
    return [
        ExpressionPoint("c1", 10, "func1", -28),
        ExpressionPoint("c2", 25, "func1", -79),
        ExpressionPoint("c3", 50, "func2", -61),
        ExpressionPoint("c4", 60, "func3", -92),
        ExpressionPoint("c5", 65, "func3", -12),
        ExpressionPoint("c6", 70, "func4", -42),
    ]


def test_graph_serialises_valid_input(sample_hierarchy, sample_points):
    graph = IPPMGraph(sample_hierarchy, dict(scalp=sample_points))

    result = serialise_graph(graph.graph_last_to_first)

    expected = {
        'directed': True,
        'multigraph': False,
        'graph': {},
        'nodes': [
            {
                'node_id': 'sc1',
                'hemisphere': 'scalp',
                'is_input_node': False,
                'channel': 'c1',
                'latency': 10,
                'transform': 'func1',
                'logp_value': -28
            },
            {
                'node_id': 'sc2',
                'hemisphere': 'scalp',
                'is_input_node': False,
                'channel': 'c2',
                'latency': 25,
                'transform': 'func1',
                'logp_value': -79
            },
            {
                'node_id': 'sc3',
                'hemisphere': 'scalp',
                'is_input_node': False,
                'channel': 'c3',
                'latency': 50,
                'transform': 'func2',
                'logp_value': -61
            },
            {
                'node_id': 'sc4',
                'hemisphere': 'scalp',
                'is_input_node': False,
                'channel': 'c4',
                'latency': 60,
                'transform': 'func3',
                'logp_value': -92
            },
            {
                'node_id': 'sc5',
                'hemisphere': 'scalp',
                'is_input_node': False,
                'channel': 'c5',
                'latency': 65,
                'transform': 'func3',
                'logp_value': -12
            },
            {
                'node_id': 'sc6',
                'hemisphere': 'scalp',
                'is_input_node': False,
                'channel': 'c6',
                'latency': 70,
                'transform': 'func4',
                'logp_value': -42
            },
            {
                'node_id': 'i1',
                'hemisphere': 'scalp',
                'is_input_node': True,
                'channel': 1,
                'latency': 0,
                'transform': 'input',
                'logp_value': -inf
            }
        ],
        'edges': [
            {'source': 'sc1', 'target': 'sc2'},
            {'source': 'sc2', 'target': 'sc3'},
            {'source': 'sc3', 'target': 'sc4'},
            {'source': 'sc4', 'target': 'sc5'},
            {'source': 'sc5', 'target': 'sc6'},
            {'source': 'i1', 'target': 'sc1'},
            {'source': 'i1', 'target': 'sc3'}
        ]
    }

    assert result == expected
