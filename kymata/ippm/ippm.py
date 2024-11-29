from kymata.entities.expression import ExpressionSet, HexelExpressionSet, SensorExpressionSet, BLOCK_SCALP, BLOCK_LEFT, \
    BLOCK_RIGHT
from kymata.ippm.denoising_strategies import MaxPoolingStrategy, DBSCANStrategy, DenoisingStrategy
from kymata.ippm.graph import IPPMGraph
from kymata.ippm.hierarchy import CandidateTransformList

_denoiser_classes = {
    "maxpooler": MaxPoolingStrategy,
    "dbscan": DBSCANStrategy,
}
_default_denoiser = "maxpooler"
_default_denoiser_kwargs = {
    "dbscan": {
        "should_normalise": True,
        "should_cluster_only_latency": True,
        "should_max_pool": False,
        "exclude_points_above_n_sigma": 5,
        "should_shuffle": True,
        "eps": 0.005,
        "min_samples": 1,
        "metric": "cosine",
        "algorithm": "auto",
        "leaf_size": 30,
        "n_jobs": -1,
    }
}


class IPPM:
    def __init__(self,
                 expression_set: ExpressionSet,
                 hierarchy: CandidateTransformList,
                 denoiser: str | None = _default_denoiser,
                 **kwargs):

        # Update kwargs
        for kw, arg in _default_denoiser_kwargs.items():
            if kw not in kwargs:
                kwargs[kw] = arg

        # Validate CTL
        for transform in hierarchy.transforms:
            if transform not in expression_set.transforms:
                raise ValueError(f"Transform {transform} from hierarchy not in expression set")
        expression_set = expression_set[hierarchy.transforms]

        denoising_strategy: DenoisingStrategy | None
        if denoiser is not None:
            try:
                denoising_strategy = _denoiser_classes[denoiser.lower()](**kwargs)
            except KeyError:
                # Argument included inappropriate denoiser name
                raise ValueError(denoiser)
        else:
            denoising_strategy = None

        # Do the denoising
        if denoising_strategy is not None:
            expression_set = denoising_strategy.denoise(expression_set)

        # Build the graph
        self._graphs: dict[str, IPPMGraph] = dict()
        if isinstance(expression_set, HexelExpressionSet):
            btl, btr = expression_set.best_transforms()
            self._graphs[BLOCK_LEFT] = IPPMGraph(hierarchy, btl)
            self._graphs[BLOCK_RIGHT] = IPPMGraph(hierarchy, btr)
        elif isinstance(expression_set, SensorExpressionSet):
            self._graphs[BLOCK_SCALP] = IPPMGraph(hierarchy, expression_set.best_transforms())
        else:
            raise NotImplementedError()

    def __getitem__(self, block: str) -> IPPMGraph:
        return self._graphs[block]

    def __contains__(self, block: str) -> bool:
        return block in self._graphs
