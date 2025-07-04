from json import JSONEncoder

import numpy as np
from networkx import Graph

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
    for node in graph.nodes:
        nodes.append({
            "node_id":       node.node_id,
            "is_input_node": node.is_input,
            "hemisphere":    node.hemisphere,
            "channel":       node.channel,
            "latency":       node.latency,
            "transform":     node.transform,
            "logp_value":    node.logp_value,
            "KID":           node.KID if node.KID is not None else "unassigned"
        })

    source: IPPMNode
    target: IPPMNode
    edges = []
    for source, target, edge_attrs in graph.edges(data=True):
        kid = edge_attrs.get('KID', None)
        edges.append({
            "source":    source.node_id,
            "target":    target.node_id,
            "transform": edge_attrs.get('transform', 'unknown'),
            "KID":       kid if kid is not None else "unassigned"
        })

    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "edges": edges
    }