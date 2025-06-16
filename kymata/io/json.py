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


def serialise_graph(graph: Graph, hemisphere: str) -> dict:
    """
    Serialize a networkx.Graph into a dictionary suitable for passing to json.dumps(..., cls=NumpyJSONEncoder).

    This version creates unique node IDs and adds metadata as requested.

    Args:
        graph: The networkx graph to serialize.
        hemisphere (str): The hemisphere block name (e.g., "left" or "right").

    Returns:
        dict: A dictionary in the desired format.
    """

    # A short code for the hemisphere, e.g. "left" -> "LH"
    hemi_code = "RH" if hemisphere.lower() == "right" else "LH"

    nodes = []
    node_id_map = {}  # Map from the ExpressionPoint object to its new unique node_id
    input_node_counter = 1

    # First, iterate through nodes to create the new node structure and the ID map
    for point_obj in graph.nodes:
        is_input = point_obj.channel == "input stream"

        channel_val = input_node_counter if is_input else point_obj.channel

        if is_input:
            node_id = f"{hemi_code}_i{channel_val}"
            input_node_counter += 1
        else:
            node_id = f"{hemi_code}_h{channel_val}"

        node_id_map[point_obj] = node_id

        nodes.append({
            "node_id": node_id,
            "is_input_node": is_input,
            "hemisphere": hemi_code,
            "channel": channel_val,
            "latency": point_obj.latency,
            "transform": point_obj.transform,
            "logp_value": point_obj.logp_value,
        })

    # Now, create the edges using the new node_ids from our map
    edges = []
    for source_obj, target_obj in graph.edges:
        edges.append({
            "source": node_id_map[source_obj],
            "target": node_id_map[target_obj],
        })

    # Assemble the final dictionary
    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "edges": edges
    }
