from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from copy import copy
from enum import StrEnum
from logging import getLogger
from collections import defaultdict

from networkx import DiGraph
from numpy import inf

from kymata.entities.expression import ExpressionPoint
from kymata.ippm.hierarchy import CandidateTransformList, group_points_by_transform, TransformHierarchy


_logger = getLogger(__file__)


@dataclass(frozen=True, eq=True)
class IPPMNode:
    """
    A node in the IPPMGraph. It is hashable and contains all metadata
    for a single expression point, including its hemisphere and unique ID.
    """
    node_id: str
    is_input_node: bool
    hemisphere: str
    channel: Any  # Can be an int from data or a generated int for an input

    # Data from the original ExpressionPoint
    transform: str
    latency: float
    logp_value: float


class IPPMConnectionStyle(StrEnum):
    """Represents a strategy for connecting nodes of neighbouring transforms in an IPPM."""
    full           = "full"
    last_to_first  = "last-to-first"
    first_to_first = "first-to-first"


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
    """

    def __init__(self,
                 ctl: CandidateTransformList | TransformHierarchy,
                 points_by_block: dict[str, list[ExpressionPoint]]):
        """
        Initializes a single, unified IPPMGraph from potentially multiple blocks of data.

        Args:
            ctl (CandidateTransformList): A CTL defining the transform hierarchy.
            points_by_block (dict): A dictionary mapping block names (e.g., "left") to their lists
                                    of ExpressionPoint data.
        """
        if not isinstance(ctl, CandidateTransformList):
            ctl = CandidateTransformList(ctl)

        graph = DiGraph()
        # Keep track of unique channel numbers for input nodes per hemisphere
        input_node_counters = defaultdict(lambda: 1)

        # Create nodes with metadata from real data points
        for block, points in points_by_block.items():
            hemi_code = "RH" if block.lower() == "right" else "LH"

            points_by_transform = group_points_by_transform(points)
            for transform, points_this_transform in points_by_transform.items():
                points_this_transform.sort(key=lambda p: p.latency)  # Sort by latency
                if transform not in ctl.transforms:
                    raise ValueError(f"Points supplied for transform {transform}, not present in transform list.")

                for point in points_this_transform:
                    node_id = f"{hemi_code}_h{point.channel}"
                    node = IPPMNode(
                        node_id=node_id,
                        is_input_node=False,
                        hemisphere=hemi_code,
                        channel=point.channel,
                        transform=point.transform,
                        latency=point.latency,
                        logp_value=point.logp_value
                    )
                    graph.add_node(node)

        # Create pseudo-nodes for inputs that don't have real data
        all_transforms_with_points = set(n.transform for n in graph.nodes)
        for input_transform in ctl.inputs:
            if input_transform not in all_transforms_with_points:
                _logger.debug(f"Input transform {input_transform} had no associated datapoints, creating pseudo-node.")
                # We must create a separate input node for each hemisphere it might feed into,
                # to ensure the node has a hemisphere ID as requested.
                for block in points_by_block.keys():
                    hemi_code = "RH" if block.lower() == "right" else "LH"
                    channel_val = input_node_counters[block]
                    node_id = f"{hemi_code}_i{channel_val}"

                    pseudo_point = input_stream_pseudo_expression_point(input_transform)
                    node = IPPMNode(
                        node_id=node_id,
                        is_input_node=True,
                        hemisphere=hemi_code,
                        channel=channel_val,
                        transform=pseudo_point.transform,
                        latency=pseudo_point.latency,
                        logp_value=pseudo_point.logp_value
                    )
                    graph.add_node(node)
                    input_node_counters[block] += 1

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
                    graph.add_edge(other_node, node)
                if other_node.transform in ctl_successors:
                    graph.add_edge(node, other_node)
                # Add sequential edges for nodes of the same transform
                if node.transform == other_node.transform and node.node_id != other_node.node_id:
                    if node.latency < other_node.latency:
                        graph.add_edge(node, other_node)
                    elif node.latency > other_node.latency:
                        graph.add_edge(other_node, node)

        self.candidate_transform_list: CandidateTransformList = ctl
        self.graph_full: DiGraph = graph


    def __copy__(self) -> IPPMGraph:
        """
        Creates a shallow copy of the current IPPMGraph instance.

        Returns:
            IPPMGraph: A new IPPMGraph instance with the same candidate transform list and expression points.
        """
        # The original __copy__ method tried to pass 'points' to the IPPMGraph constructor,
        # but the constructor expects 'points_by_block'. This is a fix for that.
        # However, a shallow copy of a graph with nodes that are dataclasses might not be truly shallow
        # if those dataclasses are mutable. For IPPMNode (frozen=True), this is fine.
        points_by_block_copy = defaultdict(list)
        for node in self.graph_full.nodes:
            # We need to reverse the hemisphere code to block name (LH -> left, RH -> right)
            block_name = "left" if node.hemisphere == "LH" else "right"
            # Since IPPMNode is frozen, we can directly create an ExpressionPoint from its attributes
            # for the purpose of recreating points_by_block.
            # Note: The original ExpressionPoint might have more fields, but for this specific
            # constructor usage (which primarily groups by transform and sorts by latency),
            # this should be sufficient.
            ep = ExpressionPoint(
                transform=node.transform,
                latency=node.latency,
                logp_value=node.logp_value,
                channel=node.channel # Assuming channel can be directly used
            )
            points_by_block_copy[block_name].append(ep)

        return IPPMGraph(ctl=copy(self.candidate_transform_list), points_by_block=points_by_block_copy)


    def __eq__(self, other: IPPMGraph) -> bool:
        """
        Tests for equality of graphs.

        Graphs are defined to be equal if they have equal CTL and equal points.

        Note that equality tests are exact, so be aware of floating-point comparisons on latencies and log p-values.
        """
        if self.candidate_transform_list != other.candidate_transform_list:
            return False
        # Comparing graphs directly for equality of nodes and edges is more robust
        # than comparing sets of nodes, as it also checks edge equality.
        return self.graph_full.graph_equality(other.graph_full)


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
