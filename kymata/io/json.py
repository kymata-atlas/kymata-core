from json import JSONEncoder
from typing import Any

import numpy as np
from networkx import Graph

from kymata.entities.expression import ExpressionPoint


class NumpyJSONEncoder(JSONEncoder):
    """
    A JSON encoder for use with Numpy datatypes.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def serialise_expression_point(point: ExpressionPoint) -> dict[str, Any]:
    """
    Serialise an expression point to a dictionary suitable to pass to json.dumps(..., cls=NumpyJSONEncoder).

    Args:
        point (ExpressionPoint): The point to serialise.

    Returns:
        dict: A dictionary suitable to pass to json.dumps(..., cls=NumpyJSONEncoder)
    """
    return {
        "channel": point.channel,
        "latency": point.latency,
        "transform": point.transform,
        "logp_value": point.logp_value,
    }


def serialise_graph(graph: Graph) -> dict:
    """
    Serialize a networkx.Graph into a dictionary suitable for passing to json.dumps(..., cls=NumpyJSONEncoder).
    Args:
        graph:

    Returns:

    """

    from networkx import node_link_data

    d = node_link_data(graph, edges="edges")
    # Appropriately serialise the expression points in nodes and edges
    d["nodes"] = [
        {
            "id": serialise_expression_point(node["id"])
        }
        for node in d["nodes"]
    ]
    d["edges"] = [
        {
            "source": serialise_expression_point(edge["source"]),
            "target": serialise_expression_point(edge["target"]),
        }
        for edge in d["edges"]
    ]
    return d
