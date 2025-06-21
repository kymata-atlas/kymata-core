from kymata.entities.expression import (
    ExpressionSet, HexelExpressionSet, SensorExpressionSet, BLOCK_SCALP, BLOCK_LEFT, BLOCK_RIGHT)
from kymata.io.json import NumpyJSONEncoder, serialise_graph
from kymata.ippm.denoising_strategies import (
    MaxPoolingStrategy, DBSCANStrategy, DenoisingStrategy, AdaptiveMaxPoolingStrategy, GMMStrategy, MeanShiftStrategy)
from kymata.ippm.graph import IPPMGraph
from kymata.ippm.hierarchy import CandidateTransformList, TransformHierarchy
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
