from __future__ import annotations

from math import inf
from typing import Collection

from copy import copy, deepcopy
from enum import StrEnum
from logging import getLogger
from collections import defaultdict

from networkx import DiGraph
from networkx.utils import graphs_equal

from kymata.entities.datatypes import Channel, Latency
from kymata.entities.expression import ExpressionPoint, BLOCK_LEFT, BLOCK_RIGHT, BLOCK_SCALP
from kymata.ippm.hierarchy import CandidateTransformList, group_points_by_transform, TransformHierarchy


_logger = getLogger(__file__)


class IPPMNode:
    """
    A node in the IPPMGraph. It contains all metadata for a single expression point, including its
    hemisphere (when referring to hexel data) and an ID.
    """
    node_id: str
    is_input: bool
    hemisphere: str  # Equivalent to the ExpressionSet `block` the data came from.  # Could even improve this using a `typing.Literal` of the allowed strings
    # Data from the original ExpressionPoint
    channel: Channel  # Can be an int from data or a generated int for an input
    latency: Latency
    transform: str
    logp_value: float
    # For API
    KID: str = "unassigned"

    # Required for NetworkX nodes to be hashable and comparable if used in sets/dicts
    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IPPMNode):
            return False
        return self.node_id == other.node_id

    def __repr__(self) -> str:
        return f"IPPMNode(node_id='{self.node_id}', transform='{self.transform}', KID='{self.KID}')"


def _node_id_from_point(point: ExpressionPoint, block: str, input_idx: int | None) -> str:
    """
    Canonical naming rules for IPPMNodes.

    Args:
        point (ExpressionPoint):
        block: The block (e.g. hemisphere) of the point.
        input_idx (int | None): Supply an int (the index/count of the input channel for this block) if this is
            an input block. Supply None if this is a non-input node.
    """
    if block in {BLOCK_LEFT, BLOCK_RIGHT}:
        if input_idx is not None:
            return f"{block}_i{input_idx}"  # e.g. "left_i4"
        # Hexel
        return f"{block}_h{point.channel}"  # e.g. "right_h1249
    elif block in {BLOCK_SCALP}:
        if input_idx is not None:
            return f"i{input_idx}"  # e.g. "i3"
        # Sensor
        return f"s{point.channel}"  # e.g. "sMEG0123
    else:
        raise NotImplementedError()


class IPPMConnectionStyle(StrEnum):
    """Represents a strategy for connecting nodes of neighbouring transforms in an IPPM."""
    full           = "full"
    last_to_first  = "last-to-first"
    first_to_first = "first-to-first"


class IPPMGraph:
    """
    Represents an actual IPPM graph, with nodes relating to actual expression points. Built from a
    CandidateTransformList and a collection of ExpressionPoints.

    This class constructs a directed graph (DiGraph) where nodes are ExpressionPoints, and edges are dependencies
    between those points based on the candidate transform list. The graph is constructed by sorting points by latency
    and adding edges according to the predecessors and successors of the transforms.

    The exception to the statement that nodes relate to actual expression points are the input nodes, which are allowed
    to not be associated with any true datapoints, in which case they are assumed to be direct sensory inputs with a
    latency of 0ms by definition. In other words, transforms which are inputs (no predecessors) in
    `candidate_transform_list` are *assumed* to be zero-latency input-streams if they do not have associated
    ExpressionPoints in `points`.
    """
    def __init__(self,
                 ctl: CandidateTransformList | TransformHierarchy,
                 points_by_block: dict[str, Collection[ExpressionPoint]]):
        """
        Initializes a single, unified IPPMGraph from potentially multiple blocks of data.

        Args:
            ctl (CandidateTransformList): A CTL defining the transform hierarchy.
            points_by_block (dict): A dictionary mapping block names to their collections of ExpressionPoint data.
        """
        if not isinstance(ctl, CandidateTransformList):
            ctl = CandidateTransformList(ctl)

        # We'll build a graph whose nodes are `IPPMNode`s, each corresponding to either an `ExpressionPoint`, augmented
        # with a little extra metadata.
        graph = DiGraph()

        # Create nodes with metadata from real data points
        self._points_by_transform: dict[str, dict[str, list[ExpressionPoint]]] = dict()  # for testing
        for block, points in points_by_block.items():
            points_by_transform = group_points_by_transform(points)
            for transform, points_this_transform in points_by_transform.items():
                if transform not in ctl.transforms:
                    raise ValueError(f"Points supplied for transform {transform}, not present in transform list.")
                for point in sorted(points_this_transform, key=lambda p: p.latency):
                    graph.add_node(IPPMNode(
                        node_id=_node_id_from_point(point=point, block=block, input_idx=None),
                        is_input=False,
                        hemisphere=block,
                        channel=point.channel,
                        transform=point.transform,
                        latency=point.latency,
                        logp_value=point.logp_value,
                    ))
            self._points_by_transform[block] = points_by_transform

        # Create pseudo-nodes for inputs that don't have real data
        all_transforms_with_points = set(n.transform for n in graph.nodes)

        # Keep track of unique channel numbers for input nodes per hemisphere.
        # We assume that all inputs will be duplicated across hemispheres.
        # hemisphere → running index of input node (1-indexed)
        input_node_idxs: dict[str, int] = defaultdict(lambda: 1)

        for input_transform in ctl.inputs:
            if input_transform in all_transforms_with_points:
                _logger.warning("Transform listed as an input in the CTL had associated data. This is unexpected!")
                continue
            _logger.debug(f"Input transform {input_transform} had no associated datapoints, creating pseudo-node.")
            # We must create a separate input node for each hemisphere it might feed into,
            # to ensure the node has a hemisphere ID as requested.
            for block in points_by_block.keys():
                input_idx = input_node_idxs[block]
                pseudo_point = input_stream_pseudo_expression_point(input_transform)
                node = IPPMNode(
                    node_id=_node_id_from_point(point=pseudo_point, block=block, input_idx=input_idx),
                    is_input=True,
                    hemisphere=block,
                    channel=input_idx,
                    transform=pseudo_point.transform,
                    latency=pseudo_point.latency,
                    logp_value=pseudo_point.logp_value,
                )
                graph.add_node(node)
                input_node_idxs[block] += 1

        # Add ALL relevant edges across the entire unified graph
        node: IPPMNode
        other_node: IPPMNode
        for node in graph.nodes:
            for other_node in graph.nodes:
                # IMPORTANT TEMPORARY LOGIC: Only add edges between nodes of the same hemisphere
                if node.hemisphere != other_node.hemisphere:
                    continue

                # Connect based on the theoretical hierarchy (CTL)
                ctl_predecessors = ctl.immediately_upstream(node.transform)
                ctl_successors = ctl.immediately_downstream(node.transform)

                if other_node.transform in ctl_predecessors:
                    graph.add_edge(other_node, node, transform=node.transform)
                if other_node.transform in ctl_successors:
                    graph.add_edge(node, other_node, transform=other_node.transform)
                # Add sequential edges for nodes of the same transform
                if node.transform == other_node.transform and node.node_id != other_node.node_id:
                    if node.latency < other_node.latency:
                        graph.add_edge(node, other_node, transform=other_node.transform)
                    elif node.latency > other_node.latency:
                        graph.add_edge(other_node, node, transform=node.transform)

        self.candidate_transform_list: CandidateTransformList = ctl
        self.graph_full: DiGraph = graph

    def __copy__(self) -> IPPMGraph:
        """
        Creates a shallow copy of the current IPPMGraph instance.

        Returns:
            IPPMGraph: A new IPPMGraph instance with the same candidate transform list and expression points.
        """
        # as it holds the original ExpressionPoint data grouped correctly.
        # Deep copy the _points_by_transform to ensure independence of the new graph's points.
        copied_points_by_block = deepcopy(self._points_by_transform)

        # When re-initializing IPPMGraph, we need to pass a dict of block -> Collection[ExpressionPoint].
        # The stored self._points_by_transform is block -> (transform -> list[ExpressionPoint]).
        # So, we need to flatten the inner dictionary for each block.
        reconstructed_points_by_block = {}
        for block, transform_points_map in copied_points_by_block.items():
            all_points_in_block = []
            for points_list in transform_points_map.values():
                all_points_in_block.extend(points_list)
            reconstructed_points_by_block[block] = all_points_in_block

        return IPPMGraph(ctl=copy(self.candidate_transform_list), points_by_block=reconstructed_points_by_block)


    def __eq__(self, other: IPPMGraph) -> bool:
        """
        Tests for equality of graphs.

        Graphs are defined to be equal if they have equal CTL and equal points.

        Note that equality tests are exact, so be aware of floating-point comparisons on latencies and log p-values.
        """
        if self.candidate_transform_list != other.candidate_transform_list:
            return False
        return graphs_equal(self.graph_full, other.graph_full)

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
        Returns a copy of the subgraph where the last node in each sequence of expressions for a single transform
        connects to the first node of the next of expression points for the downstream transform.

        Returns:
            DiGraph: A directed graph representing this version of the graph.
        """
        def __keep_edge(source: IPPMNode, dest: IPPMNode) -> bool:
            # Enforce hemisphere constraint
            if source.hemisphere != dest.hemisphere:
                return False

            # Deal with repeated-transform edges
            if source.transform == dest.transform:
                # Only want to keep single path of edges from first to last incidence of a transform, so reject edge
                # when there are other points which could serve as intermediates
                intermediates = set(self.graph_full.successors(source)) & set(self.graph_full.predecessors(dest))
                return len(intermediates) == 0

            # Deal with other edges
            else:
                # We only want edges between the LAST point in a string of same-transform points for the upstream
                # transform, to the FIRST point in the string of same-transform points for the downstream transform
                # (hence "last to first").
                # So if there's an available predecessor in the destination, don't keep
                pred: IPPMNode
                for pred in self.graph_full.predecessors(dest):
                    if pred.transform == dest.transform:
                        return False
                # If there's a successor to the source, don't keep
                succ: IPPMNode
                for succ in self.graph_full.successors(source):
                    if succ.transform == source.transform:
                        return False
                return True

        subgraph = DiGraph(self.graph_full.edge_subgraph([(s, d) for s, d in self.graph_full.edges if __keep_edge(s, d)]))

        # Add orphan nodes
        subgraph.add_nodes_from(self.graph_full.nodes)

        return subgraph

    @property
    def graph_first_to_first(self) -> DiGraph:
        """
        Returns a copy of the subgraph where the first node in each sequence of expressions for a single transform
        connects to the first node of the next of expression points for the downstream transform.

        Returns:
            DiGraph: A directed graph representing this version of the graph.
        """
        def __keep_edge(source: IPPMNode, dest: IPPMNode) -> bool:
            # Enforce hemisphere constraint
            if source.hemisphere != dest.hemisphere:
                return False

            # Deal with repeated-transform edges
            if source.transform == dest.transform:
                # Only want to keep single path of edges from first to last incidence of a transform, so reject edge
                # when there are other points which could serve as intermediates
                intermediates = set(self.graph_full.successors(source)) & set(self.graph_full.predecessors(dest))
                return len(intermediates) == 0

            # Deal with other edges
            else:
                # We only want edges between the FIRST point in a string of same-transform points for the upstream
                # transform, to the FIRST point in the string of same-transform points for the downstream transform
                # (hence "first to first").
                # So if there's an available predecessor in the destination, don't keep
                pred: IPPMNode
                for pred in self.graph_full.predecessors(dest):
                    if pred.transform == dest.transform:
                        return False
                # If there's a predecessor to the source, don't keep
                succ: IPPMNode
                for pred in self.graph_full.predecessors(source):
                    if pred.transform == source.transform:
                        return False
                return True

        subgraph = DiGraph(self.graph_full.edge_subgraph([(s, d) for s, d in self.graph_full.edges if __keep_edge(s, d)]))

        # Add orphan nodes
        subgraph.add_nodes_from(self.graph_full.nodes)

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
                           # Input given "zero" p-value to ensure it's always present
                           logp_value=-inf,
                           channel="input stream")
