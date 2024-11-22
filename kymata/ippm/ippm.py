from typing import Optional
from warnings import warn

from kymata.entities.expression import ExpressionSet, HexelExpressionSet, SensorExpressionSet
from kymata.ippm.build import IPPMBuilder, SpikeDict
from kymata.ippm.denoising_strategies import MaxPoolingStrategy, DBSCANStrategy, DenoisingStrategy
from kymata.ippm.hierarchy import TransformHierarchy, CandidateTransformList
from kymata.ippm.plot import plot_ippm


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
        "denoise_normal_dist_threshold": 5,
        "denoise_should_exclude_insignificant": True,
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
                 spikes: SpikeDict | tuple[SpikeDict, SpikeDict],
                 inputs: list[str],
                 hierarchy: TransformHierarchy):

        # Deal with multiple hemispheres
        if isinstance(spikes, tuple):
            spikes_left, spikes_right = spikes
            warn("IPPM not implemented for multiple hemispheres yet. "
                 "Just using left hemisphere.")
            spikes = spikes_left

        self._builder = IPPMBuilder(spikes, inputs, hierarchy)

    @classmethod
    def from_expression_set(cls,
                            expression_set: ExpressionSet,
                            hierarchy: CandidateTransformList,
                            merge_hemis: bool = False,
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
            if not k.startswith("denoise_")
        }
        # Set remaining needed kwargs
        denoiser_kwargs["denoise_n_timepoints"] = len(expression_set.latencies)
        denoiser_kwargs["denoise_n_channels"] = _get_n_channels(expression_set)

        # Validate CTL
        for transform in hierarchy.transforms:
            if transform.name not in expression_set.transforms:
                raise ValueError(f"Transform {transform.name} from hierarchy not in expression set")
        expression_set = expression_set[[h.name for h in hierarchy.transforms]]

        denoising_strategy: DenoisingStrategy | None
        if denoiser is not None:
            try:
                denoising_strategy = _denoiser_classes[denoiser.lower()](
                    **denoiser_kwargs)
            except KeyError:
                # Argument included inappropriate denoiser name
                raise ValueError(denoiser)
        else:
            denoising_strategy = None

        if denoising_strategy is not None:
            expression_set = denoising_strategy.denoise(expression_set)

        return cls(spikes=expression_set.best_transforms(),
                   inputs=[h.name for h in hierarchy.inputs],
                   hierarchy=hierarchy,
                   **builder_kwargs)

    def plot(self, colors: Optional[dict[str, str]] = None):
        if colors is None:
            colors = dict()
        plot_ippm(self._builder.graph, colors)
