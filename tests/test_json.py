from math import inf
import pytest
import numpy as np
import json
from networkx import Graph

from kymata.entities.expression import ExpressionPoint
from kymata.io.json import serialise_graph, NumpyJSONEncoder
from kymata.ippm.graph import IPPMGraph, IPPMNode
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


# --- Tests for NumpyJSONEncoder ---

def test_numpy_json_encoder_encodes_numpy_integers():
    """Test that NumpyJSONEncoder correctly encodes numpy.integer types to int."""
    data = {"value": np.int64(123)}
    encoded_data = json.dumps(data, cls=NumpyJSONEncoder)
    assert encoded_data == '{"value": 123}'


def test_numpy_json_encoder_encodes_numpy_floats():
    """Test that NumpyJSONEncoder correctly encodes numpy.floating types to float."""
    data = {"value": np.float32(45.67)}
    encoded_data = json.dumps(data, cls=NumpyJSONEncoder)
    decoded_data = json.loads(encoded_data) # Convert back to number
    # Use pytest.approx for floating-point comparison with a tolerance
    assert decoded_data["value"] == pytest.approx(45.67)


def test_numpy_json_encoder_encodes_numpy_arrays():
    """Test that NumpyJSONEncoder correctly encodes numpy.ndarray types to list."""
    data = {"value": np.array([1, 2, 3])}
    encoded_data = json.dumps(data, cls=NumpyJSONEncoder)
    assert encoded_data == '{"value": [1, 2, 3]}'


def test_numpy_json_encoder_handles_other_types_gracefully():
    """Test that NumpyJSONEncoder falls back to default JSONEncoder for unsupported types."""
    data = {"value": "hello"}
    encoded_data = json.dumps(data, cls=NumpyJSONEncoder)
    assert encoded_data == '{"value": "hello"}'

    # Test with a complex object that default JSONEncoder can't handle directly
    class CustomObject:
        def __init__(self):
            self.a = 1

    with pytest.raises(TypeError):
        json.dumps({"value": CustomObject()}, cls=NumpyJSONEncoder)


# --- Tests for serialise_graph ---

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
                'logp_value': -28,
                'KID': 'unassigned',
            },
            {
                'node_id': 'sc2',
                'hemisphere': 'scalp',
                'is_input_node': False,
                'channel': 'c2',
                'latency': 25,
                'transform': 'func1',
                'logp_value': -79,
                'KID': 'unassigned',
            },
            {
                'node_id': 'sc3',
                'hemisphere': 'scalp',
                'is_input_node': False,
                'channel': 'c3',
                'latency': 50,
                'transform': 'func2',
                'logp_value': -61,
                'KID': 'unassigned',
            },
            {
                'node_id': 'sc4',
                'hemisphere': 'scalp',
                'is_input_node': False,
                'channel': 'c4',
                'latency': 60,
                'transform': 'func3',
                'logp_value': -92,
                'KID': 'unassigned',
            },
            {
                'node_id': 'sc5',
                'hemisphere': 'scalp',
                'is_input_node': False,
                'channel': 'c5',
                'latency': 65,
                'transform': 'func3',
                'logp_value': -12,
                'KID': 'unassigned',
            },
            {
                'node_id': 'sc6',
                'hemisphere': 'scalp',
                'is_input_node': False,
                'channel': 'c6',
                'latency': 70,
                'transform': 'func4',
                'logp_value': -42,
                'KID': 'unassigned',
            },
            {
                'node_id': 'i1',
                'hemisphere': 'scalp',
                'is_input_node': True,
                'channel': 1,
                'latency': 0,
                'transform': 'input',
                'logp_value': -inf,
                'KID': 'unassigned',
            }
        ],
        'edges': [
            {'source': 'sc1', 'target': 'sc2', 'KID': 'unassigned', 'transform': 'func1'},
            {'source': 'sc2', 'target': 'sc3', 'KID': 'unassigned', 'transform': 'func2'},
            {'source': 'sc3', 'target': 'sc4', 'KID': 'unassigned', 'transform': 'func3'},
            {'source': 'sc4', 'target': 'sc5', 'KID': 'unassigned', 'transform': 'func3'},
            {'source': 'sc5', 'target': 'sc6', 'KID': 'unassigned', 'transform': 'func4'},
            {'source': 'i1',  'target': 'sc1', 'KID': 'unassigned', 'transform': 'func1'},
            {'source': 'i1',  'target': 'sc3', 'KID': 'unassigned', 'transform': 'func2'},
        ]
    }

    assert result == expected


def test_serialise_empty_graph():
    """Test serialisation of an empty graph."""
    empty_graph = Graph()
    result = serialise_graph(empty_graph)
    expected = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [],
        "edges": []
    }
    assert result == expected


def test_serialise_graph_with_single_node_no_edges():
    """Test serialisation of a graph with a single node and no edges."""
    single_node_graph = Graph()
    node = IPPMNode("test_node", True, "test_hemi", 1, 0, "input", -inf)
    single_node_graph.add_node(node)
    result = serialise_graph(single_node_graph)
    expected = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {
                "node_id": "test_node",
                "is_input_node": True,
                "hemisphere": "test_hemi",
                "channel": 1,
                "latency": 0,
                "transform": "input",
                "logp_value": -inf,
                'KID': 'unassigned',
            }
        ],
        "edges": []
    }
    assert result == expected


def test_serialise_graph_with_multiple_nodes_and_edges():
    """Test serialisation of a graph with multiple nodes and edges."""
    g = Graph()
    node1 = IPPMNode("n1", False, "h1", "c1", 10, "t1", -1.0)
    node2 = IPPMNode("n2", False, "h2", "c2", 20, "t2", -2.0)
    node3 = IPPMNode("n3", True, "h3", "c3", 30, "t3", -3.0) # An input node

    g.add_nodes_from([node1, node2, node3])
    g.add_edge(node1, node2)
    g.add_edge(node2, node3)

    result = serialise_graph(g)

    expected_nodes = [
        {
            'node_id': 'n1',
            'is_input_node': False,
            'hemisphere': 'h1',
            'channel': 'c1',
            'latency': 10,
            'transform': 't1',
            'logp_value': -1.0,
                'KID': 'unassigned',
        },
        {
            'node_id': 'n2',
            'is_input_node': False,
            'hemisphere': 'h2',
            'channel': 'c2',
            'latency': 20,
            'transform': 't2',
            'logp_value': -2.0,
                'KID': 'unassigned',
        },
        {
            'node_id': 'n3',
            'is_input_node': True,
            'hemisphere': 'h3',
            'channel': 'c3',
            'latency': 30,
            'transform': 't3',
            'logp_value': -3.0,
                'KID': 'unassigned',
        }
    ]

    expected_edges = [
        {'source': 'n1', 'target': 'n2'},
        {'source': 'n2', 'target': 'n3'},
    ]

    # Sort nodes and edges for consistent comparison
    result['nodes'].sort(key=lambda x: x['node_id'])
    result['edges'].sort(key=lambda x: (x['source'], x['target']))
    expected_nodes.sort(key=lambda x: x['node_id'])
    expected_edges.sort(key=lambda x: (x['source'], x['target']))

    assert result['nodes'] == expected_nodes
    assert result['edges'] == expected_edges
    assert result['directed']
    assert not result['multigraph']
    assert result['graph'] == {}

