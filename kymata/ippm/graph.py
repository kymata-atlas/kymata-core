from copy import deepcopy
from enum import StrEnum
from typing import NamedTuple

from networkx import DiGraph

from kymata.entities.expression import ExpressionPoint
from kymata.ippm.hierarchy import PointCloud, CandidateTransformList


class NodePosition(NamedTuple):
    x: float
    y: float


class YOrdinateStyle(StrEnum):
    """
    Enumeration for Y-ordinate plotting styles.

    Attributes:
        progressive: Points are plotted with increasing y ordinates.
        centered: Points are vertically centered.
    """
    progressive = "progressive"
    centered    = "centered"


class IPPMGraph:
    """
    Represents an actual IPPM graph, with nodes relating to actual expression points.

    IPPMGraph.graph nodes are IPPMNode2 objects
    """
    def __init__(self, ctl: CandidateTransformList, points: PointCloud):

        graph = DiGraph()

        # Sort points by latency ascending
        points = deepcopy(points)
        for trans in points.keys():
            points[trans].sort(key=lambda p: p.latency)

        # Copy the nodes for each instance of a point
        # Note: if the transform is not present in the data, it will
        #       not be added to the graph
        for ctl_transform in ctl.transforms:
            for point in points[ctl_transform]:
                graph.add_node(point)

        # Add ALL relevant edges
        node: ExpressionPoint
        other_node: ExpressionPoint
        for node in graph.nodes:
            ctl_predecessors = ctl.graph.predecessors(node.transform)
            ctl_successors = ctl.graph.successors(node.transform)
            for other_node in graph.nodes:
                if other_node.transform in ctl_predecessors:
                    graph.add_edge(other_node, node)
                if other_node.transform in ctl_successors:
                    graph.add_edge(node, other_node)

        self.candidate_transform_list: CandidateTransformList = ctl
        self.points: PointCloud = points
        self.graph_full: DiGraph = graph

    @property
    def graph_last_to_first(self) -> DiGraph:
        """
        A view of the graph in which the last node in each sequence of nodes connects to the first in the next sequence
        """
        def __keep_edge(source: ExpressionPoint, dest: ExpressionPoint) -> bool:
            # If there's an available predecessor, don't keep
            pred: ExpressionPoint
            for pred in self.graph_full.predecessors(source):
                if pred.transform == source.transform:
                    return False
            # If there's a successor, don't keep
            succ: ExpressionPoint
            for succ in self.graph_full.successors(dest):
                if succ.transform == dest.transform:
                    return False
            return True

        subgraph = self.graph_full.edge_subgraph([(s, d) for s, d in self.graph_full.edges if __keep_edge(s, d)])
        assert isinstance(subgraph, DiGraph)

        return subgraph
