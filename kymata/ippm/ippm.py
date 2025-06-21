from kymata.entities.expression import (
    ExpressionSet, HexelExpressionSet, SensorExpressionSet, BLOCK_SCALP, BLOCK_LEFT, BLOCK_RIGHT)
from kymata.io.json import NumpyJSONEncoder, serialise_graph
from kymata.ippm.denoising_strategies import (
    MaxPoolingStrategy, DBSCANStrategy, DenoisingStrategy, AdaptiveMaxPoolingStrategy, GMMStrategy, MeanShiftStrategy)
from kymata.ippm.graph import IPPMGraph
from kymata.ippm.hierarchy import CandidateTransformList, TransformHierarchy
from kymata.io.atlas import fetch_data_dict
from typing import Any

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

    def set_KIDs(self, transform_KIDs: dict[str, str]) -> None:
        """
        Adds KID attributes to edges and nodes in the IPPM graph based on transform mappings.

        Args:
            transform_KIDs (dict[str, str]): Dictionary mapping transform names to their KID values.

        Raises:
            ValueError: If any transform in the graph is missing a KID mapping.

        Warnings:
            Issues a warning if transforms are provided that don't exist in the graph.
        """
        import warnings

        api_url = "https://kymata.org/api/functions/"
        try:
            # Use the refactored function from atlas.py
            api_data = fetch_data_dict(api_url)
        except ConnectionError as e:
            raise ConnectionError(f"Failed to fetch KID data from {api_url}: {e}")

        # Extract all valid KIDs from the API response
        valid_api_kids = {item["kid"] for item in api_data if "kid" in item}

        # Check if all KIDs in transform_KIDs exist in valid_api_kids
        provided_kids = set(transform_KIDs.values())
        missing_api_kids = provided_kids - valid_api_kids
        if missing_api_kids:
            raise ValueError(f"The following KIDs from transform_KIDs do not exist in the API: {sorted(list(missing_api_kids))}")

        # Add KID attribute to all edges based on the edges' transform
        for graph_view in [self.graph.graph_full, self.graph.graph_last_to_first, self.graph.graph_first_to_first]:
            for u, v, data in graph_view.edges(data=True):
                edge_transform = data.get('transform')
                if edge_transform:
                    kid_value = transform_KIDs.get(edge_transform)
                    if kid_value is not None:
                        graph_view.edges[u, v]['KID'] = kid_value
                    else:
                        graph_view.edges[u, v]['KID'] = "n/a"
                else:
                    graph_view.edges[u, v]['KID'] = "n/a (no transform)"

        # Add KID attributes to the nodes - now directly modifiable
        for graph_view in [self.graph.graph_full, self.graph.graph_last_to_first, self.graph.graph_first_to_first]:
            for node in graph_view.nodes():
                node_transform = node.transform
                kid_value = transform_KIDs.get(node_transform)
                if kid_value is not None:
                    node.KID = kid_value # Direct modification of KID attribute
                else:
                    node.KID = "n/a" # Direct modification of KID attribute

        # Ensure all nodes (except input nodes) have a KID
        nodes_missing_kids = []
        for node in self.graph.graph_full.nodes():
            if not node.is_input and node.KID == "unassigned": # Check node.KID directly
                nodes_missing_kids.append(node.transform)
            elif not node.is_input and node.KID == "n/a": # Check node.KID directly
                nodes_missing_kids.append(node.transform)

        if nodes_missing_kids:
            raise ValueError(f"Missing KID mappings for nodes in graph: {sorted(list(set(nodes_missing_kids)))}")

        # Check whether there are any edges that haven't been assigned a KID
        edges_missing_kids = []
        for source, target, data in self.graph.graph_full.edges(data=True):
            if 'KID' not in data or data['KID'] == "n/a" or data['KID'] == "n/a (no transform)":
                edges_missing_kids.append(f"{source.transform} -> {target.transform}")

        if edges_missing_kids:
            raise ValueError(f"The following edges have not been assigned a KID: {sorted(list(set(edges_missing_kids)))}")

        # Optional: Warn about transforms in transform_KIDs that are not in the graph
        transforms_in_graph = self.graph.transforms
        transforms_not_in_graph = set(transform_KIDs.keys()) - transforms_in_graph
        if transforms_not_in_graph:
            warnings.warn(f"The following transforms provided in transform_KIDs do not exist in the graph: {sorted(list(transforms_not_in_graph))}")