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
        ExpressionPoint("c", 10, "func1", -28),
        ExpressionPoint("c", 25, "func1", -79),
        ExpressionPoint("c", 50, "func2", -61),
        ExpressionPoint("c", 60, "func3", -92),
        ExpressionPoint("c", 65, "func3", -12),
        ExpressionPoint("c", 70, "func4", -42),
    ]


def test_graph_serialises_valid_input(sample_hierarchy, sample_points):
    graph = IPPMGraph(sample_hierarchy, sample_points)

    result = serialise_graph(graph.graph_last_to_first)

    expected = {
        'directed': True,
        'multigraph': False,
        'graph': {},
        'nodes': [
            {'id': {'channel': 'c',
                    'latency': 10,
                    'transform': 'func1',
                    'logp_value': -28}},
            {'id': {'channel': 'c',
                    'latency': 25,
                    'transform': 'func1',
                    'logp_value': -79}},
            {'id': {'channel': 'c',
                    'latency': 50,
                    'transform': 'func2',
                    'logp_value': -61}},
            {'id': {'channel': 'c',
                    'latency': 60,
                    'transform': 'func3',
                    'logp_value': -92}},
            {'id': {'channel': 'c',
                    'latency': 65,
                    'transform': 'func3',
                    'logp_value': -12}},
            {'id': {'channel': 'c',
                    'latency': 70,
                    'transform': 'func4',
                    'logp_value': -42}},
            {'id': {'channel': 'input stream',
                    'latency': 0,
                    'transform': 'input',
                    'logp_value': -inf}}
        ],
    'edges': [
        {
            'source': {'channel': 'c',
                    'latency': 10,
                    'transform': 'func1',
                    'logp_value': -28},
            'target': {'channel': 'c',
                       'latency': 25,
                       'transform': 'func1',
                       'logp_value': -79
                       }
        },
        {
            'source': {'channel': 'c',
                       'latency': 25,
                       'transform': 'func1',
                       'logp_value': -79},
            'target': {'channel': 'c',
                       'latency': 50,
                       'transform': 'func2',
                       'logp_value': -61
                       }
        },
        {
            'source': {'channel': 'c',
                       'latency': 50,
                       'transform': 'func2',
                       'logp_value': -61},
            'target': {'channel': 'c',
                       'latency': 60,
                       'transform': 'func3',
                       'logp_value': -92
                       }
        },
        {
            'source': {'channel': 'c',
                       'latency': 60,
                       'transform': 'func3',
                       'logp_value': -92},
            'target': {'channel': 'c',
                       'latency': 65,
                       'transform': 'func3',
                       'logp_value': -12
                       }
        },
        {
            'source': {'channel': 'c',
                       'latency': 65,
                       'transform': 'func3',
                       'logp_value': -12},
            'target': {'channel': 'c',
                       'latency': 70,
                       'transform': 'func4',
                       'logp_value': -42
                       }
        },
        {
            'source': {'channel': 'input stream',
                       'latency': 0,
                       'transform': 'input',
                       'logp_value': -inf},
            'target': {'channel': 'c',
                       'latency': 10,
                       'transform': 'func1',
                       'logp_value': -28
                       }
        },
        {
            'source': {'channel': 'input stream',
                       'latency': 0,
                       'transform': 'input',
                       'logp_value': -inf},
            'target': {'channel': 'c',
                       'latency': 50,
                       'transform': 'func2',
                       'logp_value': -61}
        }
    ]}

    assert result == expected
