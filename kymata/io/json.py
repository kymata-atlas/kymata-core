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


def serialise_graph(graph: Graph) -> dict:
    """
    Serialize a networkx.Graph into a dictionary. It reads metadata directly
    from the IPPMNode objects that constitute the graph.
    """
    nodes = []
    for node_obj in graph.nodes:
        # node_obj is now an IPPMNode instance
        nodes.append({
            "node_id": node_obj.node_id,
            "is_input_node": node_obj.is_input_node,
            "hemisphere": node_obj.hemisphere,
            "channel": node_obj.channel,
            "latency": node_obj.latency,
            "transform": node_obj.transform,
            "logp_value": node_obj.logp_value,
        })

    edges = []
    # The source and target of an edge are the IPPMNode objects themselves
    for source_obj, target_obj in graph.edges:
        edges.append({
            "source": source_obj.node_id,
            "target": target_obj.node_id,
        })

    # Assemble the final dictionary in the same format as before
    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "edges": edges
    }
