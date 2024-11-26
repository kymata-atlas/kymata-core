from kymata.entities.expression import ExpressionSet, HexelExpressionSet, SensorExpressionSet, BLOCK_SCALP, BLOCK_LEFT, \
    BLOCK_RIGHT
from kymata.ippm.denoising_strategies import MaxPoolingStrategy, DBSCANStrategy, DenoisingStrategy
from kymata.ippm.graph import IPPMGraph
from kymata.ippm.hierarchy import CandidateTransformList
from kymata.math.probability import sidak_correct, p_threshold_for_sigmas, p_to_logp

_denoiser_classes = {
    "maxpooler": MaxPoolingStrategy,
    "dbscan": DBSCANStrategy,
}
_default_denoiser = "maxpooler"
_default_denoiser_kwargs = {
    "dbscan": {
        "denoise_should_normalise": True,
        "denoise_should_cluster_only_latency": True,
        "denoise_should_max_pool": False,
        # "denoise_exclude_logp_vals_above": 5-sigma,  # Default value set below
        "denoise_should_shuffle": True,
        "denoise_eps": 0.005,
        "denoise_min_samples": 1,
        "denoise_metric": "cosine",
        "denoise_algorithm": "auto",
        "denoise_leaf_size": 30,
        "denoise_n_jobs": -1,
    }
}


def _get_n_channels(es: ExpressionSet):
    if isinstance(es, SensorExpressionSet):
        return len(es.sensors)
    if isinstance(es, HexelExpressionSet):
        return len(es.hexels_left) + len(es.hexels_right)
    raise NotImplementedError()


class IPPM:
    def __init__(self,
                 expression_set: ExpressionSet,
                 hierarchy: CandidateTransformList,
                 denoiser: str | None = _default_denoiser,
                 **kwargs):

        # update kwargs
        for kw, arg in _default_denoiser_kwargs.items():
            if kw not in kwargs:
                kwargs[kw] = arg

        # filter kwargs
        denoiser_kwargs = {
            k.removeprefix("denoise_"): v
            for k, v in kwargs.items()
            if k.startswith("denoise_")
        }
        builder_kwargs = {
            k: v
            for k, v in kwargs.items()
            # Remaining
            if not k.startswith("denoise_")
        }
        # Set remaining needed default values
        if "denoise_exclude_logp_vals_above" not in denoiser_kwargs:
            n_comparisons = len(expression_set.transforms) * len(expression_set.latencies) * _get_n_channels(expression_set)
            denoiser_kwargs["denoise_exclude_logp_vals_above"] = p_to_logp(sidak_correct(p_threshold_for_sigmas(5), n_comparisons=n_comparisons))

        # Validate CTL
        for transform in hierarchy.transforms:
            if transform not in expression_set.transforms:
                raise ValueError(f"Transform {transform} from hierarchy not in expression set")
        expression_set = expression_set[hierarchy.transforms]

        denoising_strategy: DenoisingStrategy | None
        if denoiser is not None:
            try:
                denoising_strategy = _denoiser_classes[denoiser.lower()](**denoiser_kwargs)
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
