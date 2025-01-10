from copy import copy

from networkx import DiGraph

from kymata.entities.expression import ExpressionPoint
from kymata.ippm.hierarchy import GroupedPoints, CandidateTransformList, group_points_by_transform, TransformHierarchy


class IPPMGraph:
    """
    Represents an actual IPPM graph, with nodes relating to actual expression points. Built from a
    CandidateTransformList and a set of ExpressionPoints.

    This class constructs a directed graph (DiGraph) where nodes are ExpressionPoints, and edges are dependencies
    between those points based on the candidate transform list. The graph is constructed by sorting points by latency
    and adding edges according to the predecessors and successors of the transforms.

    Args:
        ctl (CandidateTransformList or TransformHierarchy): A CTL that defines the transformation hierarchy.
        original_points (list[ExpressionPoint]): A list of expression points that will be used to build the graph.

    Attributes:
        candidate_transform_list (CandidateTransformList): The candidate transform list used to create the graph.
        points (GroupedPoints): A dictionary grouping points by their associated transforms.
    """
    def __init__(self,
                 ctl: CandidateTransformList | TransformHierarchy,
                 original_points: list[ExpressionPoint]):
        """
        Initializes an IPPMGraph object by constructing a directed graph from a list of expression points and
        a candidate transform list.

        Args:
            ctl (CandidateTransformList): A CTL defining the transform hierarchy.
            original_points (list[ExpressionPoint]): A actual evidential expression points from data.
        """
        if not isinstance(ctl, CandidateTransformList):
            ctl = CandidateTransformList(ctl)

        graph = DiGraph()

        # Sort points by latency ascending
        points_by_transform = group_points_by_transform(original_points)
        for trans in points_by_transform.keys():
            points_by_transform[trans].sort(key=lambda p: p.latency)

        # Copy the nodes for each instance of a point
        # Note: if the transform is not present in the data, it will not be added to the graph
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
                # Add sequential edges between same-transform nodes
                if node.transform == other_node.transform:
                    if node.latency < other_node.latency:
                        graph.add_edge(node, other_node)
                    elif node.latency > other_node.latency:
                        graph.add_edge(other_node, node)

        self.candidate_transform_list: CandidateTransformList = ctl
        self.points: GroupedPoints = points_by_transform
        # The "full" graph has all possible connections between sequences of upstream nodes for the same transform and
        # sequences of downstream nodes for the same transform.
        # It should usually be accessed (for display purposes) via alternative graphs below,
        # e.g. self.graph_last_to_first.
        self._graph_full: DiGraph = graph

    def __copy__(self):
        """
        Creates a shallow copy of the current IPPMGraph instance.

        Returns:
            IPPMGraph: A new IPPMGraph instance with the same candidate transform list and expression points.
        """
        return IPPMGraph(ctl=copy(self.candidate_transform_list),
                         original_points=[p for _, points in self.points.items() for p in points])

    @property
    def transforms(self) -> set[str]:
        """
        All transforms with nodes in the IPPM graph.

        Returns:
            set[str]: A set of transform names present in the graph.
        """
        return set(n.transform for n in self._graph_full.nodes)

    @property
    def inputs(self) -> set[str]:
        """
        All input transforms in the IPPM graph.

        Returns:
            set[str]: A set of input transform names present in the graph.
        """
        return self.candidate_transform_list.inputs & self.transforms

    @property
    def terminals(self) -> set[str]:
        """
        All terminal transforms (those with no successors) with nodes in the IPPM graph.

        Returns:
            set[str]: A set of terminal transform names present in the graph.
        """
        return self.candidate_transform_list.terminals & self.transforms

    @property
    def serial_sequence(self) -> list[list[str]]:
        """
        The serial sequence of transforms in the graph.

        Elements in the returned list are the serial sequence, with each element itself being a list of parallel
        transforms.

        Returns:
            list[list[str]]: A list of lists representing the ordered transforms in the serial sequence.
        """
        return [
            [t for t in step if t in self.transforms]
            for step in self.candidate_transform_list.serial_sequence
        ]

    @property
    def graph_last_to_first(self) -> DiGraph:
        """
        Returns a view of the graph where the last node in each sequence of expressions for a single transform connects
        to the first node of the next of expression points for the downstream transform.

        Returns:
            DiGraph: A directed graph representing this version of the graph.
        """
        def __keep_edge(source: ExpressionPoint, dest: ExpressionPoint) -> bool:
            # Deal with repeated-transform edges
            if source.transform == dest.transform:
                # Only want to keep single path of edges from first to last incidence of a transform
                # So search for intermediate points and reject the edge if any are found
                potential_intermediates = self._graph_full.successors(source)
                pred: ExpressionPoint
                for pred in self._graph_full.predecessors(dest):
                    if pred in potential_intermediates:
                        return False
                return True

            # Deal with other edges
            else:
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
