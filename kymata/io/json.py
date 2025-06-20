from json import JSONEncoder
from math import isinf

import numpy as np
from networkx import Graph

from kymata.ippm.graph import IPPMNode


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
    node: IPPMNode
    for node in graph.nodes:
        nodes.append({
            "node_id":       node.node_id,
            "is_input_node": node.is_input,
            "hemisphere":    node.hemisphere,
            "channel":       node.channel,
            "latency":       node.latency,
            "transform":     node.transform,
            # For purposes of serialisation, we treat "zero" probabilities as just very small
            "logp_value":    node.logp_value
        })

    edges = []
    for source, target in graph.edges:
        edges.append({
            "source": source.node_id,
            "target": target.node_id,
        })

    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "edges": edges
    }
