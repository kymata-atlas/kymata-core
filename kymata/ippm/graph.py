from __future__ import annotations

from copy import copy
from logging import getLogger

from networkx import DiGraph
from numpy import inf

from kymata.entities.expression import ExpressionPoint
from kymata.ippm.hierarchy import GroupedPoints, CandidateTransformList, group_points_by_transform, TransformHierarchy


_logger = getLogger(__file__)


class IPPMGraph:
    """
    Represents an actual IPPM graph, with nodes relating to actual expression points. Built from a
    CandidateTransformList and a set of ExpressionPoints.

    This class constructs a directed graph (DiGraph) where nodes are ExpressionPoints, and edges are dependencies
    between those points based on the candidate transform list. The graph is constructed by sorting points by latency
    and adding edges according to the predecessors and successors of the transforms.

    The exception to the statement that nodes relate to actual expression points are the input nodes, which are allowed
    to not be associated with any true datapoints, in which case they are assumed to be direct sensory inputs with a
    latency of 0ms by definition. In other words, transforms which are inputs (no predecessors) in
    `candidate_transform_list` are *assumed* to be zero-latency input-streams if they do not have associated
    ExpressionPoints in `points`.

    Args:
        ctl (CandidateTransformList or TransformHierarchy): A CTL that defines the transformation hierarchy.
        points (list[ExpressionPoint]): A list of expression points that will be used to build the graph.

    Attributes:
        candidate_transform_list (CandidateTransformList): The candidate transform list used to create the graph.
        points (GroupedPoints): A dictionary grouping points by their associated transforms.
    """
    def __init__(self,
                 ctl: CandidateTransformList | TransformHierarchy,
                 points: list[ExpressionPoint]):
        """
        Initializes an IPPMGraph object by constructing a directed graph from a list of expression points and
        a candidate transform list.

        Args:
            ctl (CandidateTransformList): A CTL defining the transform hierarchy.
            points (list[ExpressionPoint]): A actual evidential expression points from data.
        """
        if not isinstance(ctl, CandidateTransformList):
            ctl = CandidateTransformList(ctl)

        graph = DiGraph()

        # Sort points by latency ascending
        points_by_transform = group_points_by_transform(points)
        for trans in points_by_transform.keys():
            points_by_transform[trans].sort(key=lambda p: p.latency)

        # Add a node for each datapoint corresponding to a transform
        # Note: if the transform is not present in the data, it will not be added to the graph (unless it is an input)
        for transform, points_this_transform in points_by_transform.items():
            if transform not in ctl.transforms:
                raise ValueError(f"Points supplied for transform {transform}, not present in transform list.")
            for point in points_this_transform:
                graph.add_node(point)

        # Also add points for inputs
        for input_transform in ctl.inputs:
            # Two cases to consider:
            if input_transform in points_by_transform.keys() and len(points_by_transform[input_transform]) > 0:
                # 1. Input transform has associated ExpressionPoints data (e.g. looking at a partial IPPM whose inputs
                #    are non-zero-latency transforms themselves). In this case they will have been present in
                #    `points_by_transform` and will have been added as nodes already. However, this is likely to be an
                #    unusual case, so we notify the user it's occurred.
                _logger.info(f"Input transform {input_transform} had associated datapoints")

            else:
                # 2. Input transform had no associated ExpressionPoints data, so is assumed to be an input-stream node,
                #    with latency defined to be 0 seconds, and "zero" probability.
                _logger.debug(f"Input transform {input_transform} had no associated datapoints")
                graph.add_node(input_stream_pseudo_expression_point(input_transform))

        # Add ALL relevant edges
        node: ExpressionPoint
        other_node: ExpressionPoint
        for node in graph.nodes:
            ctl_predecessors = set(ctl.graph.predecessors(node.transform))
            ctl_successors = set(ctl.graph.successors(node.transform))
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
        self.graph_full: DiGraph = graph

    def __copy__(self) -> IPPMGraph:
        """
        Creates a shallow copy of the current IPPMGraph instance.

        Returns:
            IPPMGraph: A new IPPMGraph instance with the same candidate transform list and expression points.
        """
        return IPPMGraph(ctl=copy(self.candidate_transform_list),
                         points=[p for _, points in self.points.items() for p in points])

    def __eq__(self, other: IPPMGraph) -> bool:
        """
        Tests for equality of graphs.

        Graphs are defined to be equal if they have equal CTL and equal points.

        Note that equality tests are exact, so be aware of floating-point comparisons on latencies and log p-values.
        """
        if self.candidate_transform_list != other.candidate_transform_list:
            return False
        if self.points != other.points:
            return False
        return True

    @property
    def transforms(self) -> set[str]:
        """
        All transforms with nodes in the IPPM graph.

        Returns:
            set[str]: A set of transform names present in the graph.
        """
        return set(n.transform for n in self.graph_full.nodes)

    @property
    def inputs(self) -> set[str]:
        """
        All input transforms in the IPPM graph.

        Returns:
            set[str]: A set of input transform names present in the graph.
        """
        return self.candidate_transform_list.inputs

    @property
    def terminals(self) -> set[str]:
        """
        All terminal transforms (those with no successors) with nodes in the IPPM graph.

        Returns:
            set[str]: A set of terminal transform names present in the graph.
        """
        terminal_nodes = set(n for n in self.graph_full.nodes if len(list(self.graph_full.successors(n))) == 0)
        return set(n.transform for n in terminal_nodes)

    @property
    def serial_sequence(self) -> list[list[str]]:
        """
        The serial sequence of transforms in the graph.

        Elements in the returned list are the serial sequence, with each element itself being a list of parallel
        transforms.

        Returns:
            list[list[str]]: A list of lists representing the ordered transforms in the serial sequence.
        """
        subsequence = [
            [t for t in step if t in self.transforms]
            for step in self.candidate_transform_list.serial_sequence
        ]
        # Skip any steps which were completely excluded due to missing data
        return [step for step in subsequence if len(step) > 0]

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
                # Only want to keep single path of edges from first to last incidence of a transform, so reject edge
                # when there are other points wihch could serve as intermediates
                intermediates = set(self.graph_full.successors(source)) & set(self.graph_full.predecessors(dest))
                return len(intermediates) == 0

            # Deal with other edges
            else:
                # We only want edges between the LAST point in a string of same-transform points for the upstream
                # transform, to the FIRST point in the string of same-transform points for the downstream transform
                # (hence "last to first").
                # So if there's an available predecessor in the destination, don't keep
                pred: ExpressionPoint
                for pred in self.graph_full.predecessors(dest):
                    if pred.transform == dest.transform:
                        return False
                # If there's a successor to the source, don't keep
                succ: ExpressionPoint
                for succ in self.graph_full.successors(source):
                    if succ.transform == source.transform:
                        return False
                return True

        subgraph = self.graph_full.edge_subgraph([(s, d) for s, d in self.graph_full.edges if __keep_edge(s, d)])
        assert isinstance(subgraph, DiGraph)

        return subgraph


def input_stream_pseudo_expression_point(input_name: str) -> ExpressionPoint:
    """
    Input-stream transforms get given "pseudo" ExpressionPoints in the graph with a latency and p-value of 0.
    
    Args:
        input_name (str): The name of the input stream. 

    Returns:
        ExpressionPoint

    """
    return ExpressionPoint(transform=input_name,
                           # Input latency defined to be 0ms
                           latency=0,
                           # Input given "zero" probability to ensure it's always present
                           logp_value=-inf,
                           channel="input stream")
