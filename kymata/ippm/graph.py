from copy import deepcopy, copy

from networkx import DiGraph

from kymata.entities.expression import ExpressionPoint
from kymata.ippm.hierarchy import GroupedPoints, CandidateTransformList, group_points_by_transform


class IPPMGraph:
    """
    Represents an actual IPPM graph, with nodes relating to actual expression points.

    IPPMGraph.graph nodes are IPPMNode2 objects
    """
    def __init__(self, ctl: CandidateTransformList, points: list[ExpressionPoint]):

        graph = DiGraph()

        # Sort points by latency ascending
        points = deepcopy(points)
        points_by_transform = group_points_by_transform(points)
        for trans in points_by_transform.keys():
            points_by_transform[trans].sort(key=lambda p: p.latency)

        # Copy the nodes for each instance of a point
        # Note: if the transform is not present in the data, it will
        #       not be added to the graph
        for ctl_transform in ctl.transforms:
            for point in points_by_transform[ctl_transform]:
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
        self.points: GroupedPoints = points_by_transform
        self._graph_full: DiGraph = graph

    def __copy__(self):
        return IPPMGraph(ctl=copy(self.candidate_transform_list),
                         points=[p for _, points in self.points.items() for p in points])

    @property
    def transforms(self) -> set[str]:
        """All transforms with nodes in the IPPM graph."""
        return set(self._graph_full.nodes)

    @property
    def inputs(self) -> set[str]:
        """All input transforms with nodes in the IPPM graph."""
        return self.candidate_transform_list.inputs & self.transforms

    @property
    def terminals(self) -> set[str]:
        """Terminal transforms (those with no successors) with nodes in the IPPM graph."""
        return self.candidate_transform_list.terminals & self.transforms

    @property
    def serial_sequence(self) -> list[list[str]]:
        return [
            [t for t in step if t in self.transforms]
            for step in self.candidate_transform_list.serial_sequence
        ]

    @property
    def graph_last_to_first(self) -> DiGraph:
        """
        A view of the graph in which the last node in each sequence of nodes connects to the first in the next sequence
        """
        def __keep_edge(source: ExpressionPoint, dest: ExpressionPoint) -> bool:
            # If there's an available predecessor, don't keep
            pred: ExpressionPoint
            for pred in self._graph_full.predecessors(source):
                if pred.transform == source.transform:
                    return False
            # If there's a successor, don't keep
            succ: ExpressionPoint
            for succ in self._graph_full.successors(dest):
                if succ.transform == dest.transform:
                    return False
            return True

        subgraph = self._graph_full.edge_subgraph([(s, d) for s, d in self._graph_full.edges if __keep_edge(s, d)])
        assert isinstance(subgraph, DiGraph)

        return subgraph
