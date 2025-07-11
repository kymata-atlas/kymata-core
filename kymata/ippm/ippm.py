from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

from networkx.relabel import relabel_nodes

from kymata.entities.expression import (
    ExpressionSet, HexelExpressionSet, SensorExpressionSet, BLOCK_SCALP, BLOCK_LEFT, BLOCK_RIGHT)
from kymata.io.atlas import API_URL, verify_kids
from kymata.io.json import NumpyJSONEncoder, serialise_graph
from kymata.ippm.denoising_strategies import (
    MaxPoolingStrategy, DBSCANStrategy, DenoisingStrategy, AdaptiveMaxPoolingStrategy, GMMStrategy, MeanShiftStrategy)
from kymata.ippm.graph import IPPMGraph, IPPMNode
from kymata.ippm.hierarchy import CandidateTransformList, TransformHierarchy

_denoiser_classes = {
    "maxpool": MaxPoolingStrategy,
    "adaptive_maxpool": AdaptiveMaxPoolingStrategy,
    "dbscan": DBSCANStrategy,
    "gmm": GMMStrategy,
    "mean_shift": MeanShiftStrategy,
}
_default_denoiser = "maxpool"


class IPPM:
    """
    IPPM container/constructor object. Use this class as an interface to build a single, unified IPPM graph
    from an expression set.
    """
    def __init__(self,
                 expression_set: ExpressionSet,
                 candidate_transform_list: CandidateTransformList | TransformHierarchy,
                 denoiser: str | None = _default_denoiser,
                 **kwargs: dict[str, Any]):
        """
        Args:
           expression_set (ExpressionSet): The ExpressionSet from which to build the IPPM.
           candidate_transform_list (CandidateTransformList): The CTL (i.e. underlying hypothetical IPPM) to be applied
               to the expression set.
           denoiser (str, optional): The denoising method to be applied to the expression set. Default is None.
           **kwargs: Additional arguments passed to the denoiser.

        Raises:
            ValueError: If any transform in the hierarchy is not found in the expression set, or if the provided denoiser
                is invalid.
        """

        # Validate CTL
        if not isinstance(candidate_transform_list, CandidateTransformList):
            candidate_transform_list = CandidateTransformList(candidate_transform_list)
        for transform in candidate_transform_list.transforms - candidate_transform_list.inputs:
            if transform not in expression_set.transforms:
                raise ValueError(f"Transform {transform} from hierarchy not in expression set")

        expression_set = expression_set[candidate_transform_list.transforms & set(expression_set.transforms)]

        denoising_strategy: DenoisingStrategy | None
        if denoiser is not None:
            try:
                denoising_strategy = _denoiser_classes[denoiser.lower()](**kwargs)
            except KeyError:
                # Argument included inappropriate denoiser name
                raise ValueError(denoiser)
        else:
            denoising_strategy = None

        # Get and group the points to include in the graph
        if isinstance(expression_set, HexelExpressionSet):
            if denoising_strategy is not None:
                points_left, points_right = denoising_strategy.denoise(expression_set)
            else:
                points_left, points_right = expression_set.best_transforms()

            # Group all points by their block (hemisphere) to pass to the graph constructor
            all_points = {
                BLOCK_LEFT: points_left,
                BLOCK_RIGHT: points_right
            }

        elif isinstance(expression_set, SensorExpressionSet):
            if denoising_strategy is not None:
                points = denoising_strategy.denoise(expression_set)
            else:
                points = expression_set.best_transforms()

            # For consistency, we still pass a dictionary, even with one block
            all_points = {BLOCK_SCALP: points}

        else:
            raise NotImplementedError()

        self.graph = IPPMGraph(candidate_transform_list, all_points)

    def to_json(self) -> str:
        """
        Serializes the IPPM into a JSON format suitable for sending to <kymata.org>.

        Returns:
            str: JSON string.
        """
        import json

        jdict = serialise_graph(self.graph.graph_last_to_first)
        return json.dumps(jdict, indent=2, cls=NumpyJSONEncoder)

    def set_kids(self, transform_kids: dict[str, str], verify_against_api: bool = True) -> None:
        """
        Adds KID attributes to edges and nodes in the IPPM graph based on transform mappings.

        Args:
            transform_kids (dict[str, str]): Dictionary mapping transform names to their KID values.
            verify_against_api (bool): If True (the default), verify all supplied KIDs against the Kymata Atlas API.

        Raises:
            ValueError: When verifying against the API (and assuming the API is reachable), if any of the KIDs are not
                valid.

        Warnings:
            Issues a warning if:
            - Transforms are provided that don't exist in the graph.
            - There is a connection issue with the Kymata Atlas API.
        """

        # Verify supplied KIDs against API
        if verify_against_api:
            try:
                verify_kids(transform_kids.values())
            except ConnectionError as e:
                warnings.warn(f"Failed to fetch KID data from {API_URL} ({e}). KIDs will not be verified.")

        # Check KID coverage:
        transforms_not_in_graph = set(transform_kids.keys()) - self.graph.transforms
        if len(transforms_not_in_graph) > 0:
            warnings.warn(f"The following transforms were supplied in the mapping but were not represented in the graph:"
                          f" {sorted(transforms_not_in_graph)}")
        transforms_not_in_mapping = self.graph.transforms - self.graph.inputs - set(transform_kids.keys())
        if len(transforms_not_in_mapping) > 0:
            warnings.warn(f"The following transforms in the graph weren't supplied a KID:"
                          f" {sorted(transforms_not_in_mapping)}")

        # Add KIDs to nodes
        node: IPPMNode
        node_relabel_dict = {
            node: IPPMNode(
                node_id=node.node_id,
                is_input=node.is_input,
                hemisphere=node.hemisphere,
                channel=node.channel,
                latency=node.latency,
                transform=node.transform,
                logp_value=node.logp_value,
                # Replace KIDs from the dictionary
                KID=transform_kids.get(node.transform, node.KID),
            )
            for node in self.graph.graph_full.nodes
        }
        self.graph.graph_full = relabel_nodes(self.graph.graph_full, node_relabel_dict)

        # Add KIDs to edges
        source: IPPMNode
        target: IPPMNode
        for source, target, data in self.graph.graph_full.edges(data=True):
            if target.transform in transform_kids:
                self.graph.graph_full.edges[source, target]["KID"] = transform_kids[target.transform]

    def __add__(self, other: IPPM) -> IPPM:
        """
        Combines two IPPM instances into a new IPPM instance using direct graph merging.

        Args:
            other (IPPM): The other IPPM instance to combine with this one.

        Returns:
            IPPM: A new IPPM instance containing the combined graphs.

        Raises:
            ValueError: If there are conflicting nodes, edges, or incompatible block structures.
            TypeError: If other is not an IPPM instance.
        """
        if not isinstance(other, IPPM):
            raise TypeError(f"Cannot add IPPM with {type(other)}. Both operands must be IPPM instances.")

        # Check for block compatibility
        self_blocks = set(self.graph._points_by_transform.keys())
        other_blocks = set(other.graph._points_by_transform.keys())

        if self_blocks != other_blocks:
            raise ValueError(f"Block mismatch between IPPMs. "
                             f"First IPPM has blocks: {sorted(self_blocks)}, "
                             f"Second IPPM has blocks: {sorted(other_blocks)}. "
                             f"Cannot combine IPPMs with different block structures.")

        # Create new IPPM instance
        combined_ippm = deepcopy(self)
        combined_ippm.graph = IPPMGraph.merge(combined_ippm.graph, other.graph)

        return combined_ippm
