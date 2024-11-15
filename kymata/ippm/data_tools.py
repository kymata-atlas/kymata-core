from copy import deepcopy
from typing import NamedTuple, Optional

from kymata.entities.constants import HEMI_LEFT, HEMI_RIGHT
from kymata.entities.expression import HexelExpressionSet, SensorExpressionSet


class ExpressionPairing(NamedTuple):
    """
    A temporal location representing evidence of expression with an associated log p-value.
    """
    latency_ms: float  # [0]
    logp_value: float  # [1]


class IPPMSpike(object):
    """
    A collection of significant points relating to a (hypothetically) single localised effect for a named transform.

    Attributes
    ----------
        transform : the name of the transform who caused the spike.
        best_pairings : best latencies for this transform
    """

    def __init__(self, transform_name: str):
        self.transform: str = transform_name
        self.best_pairings: list[ExpressionPairing] = []

        self.input_stream: Optional[str] = None

    def add_pairing(self, pairing: ExpressionPairing):
        """
        Use this to add new timings.

        Params
        ------
            pairing : Corresponds to the best match to a spike
        """
        self.best_pairings.append(pairing)


# transform_name → spike
SpikeDict = dict[str, IPPMSpike]


def merge_hemispheres(spikes_left: SpikeDict, spikes_right: SpikeDict) -> SpikeDict:
    """Merges the best pairings from left- and right-hemisphere spikes into a single spike."""
    spikes_both: SpikeDict = deepcopy(spikes_left)
    for transform, spikes_right in spikes_right.items():
        if transform not in spikes_left:
            spikes_left[transform] = IPPMSpike(transform)
        spikes_both[transform].best_pairings.extend(spikes_right.best_pairings)
    return spikes_both


def build_spike_dicts_from_hexel_expression_set(expression_set: HexelExpressionSet) -> tuple[SpikeDict, SpikeDict]:
    """
    Builds the spike dictionary from a HexelExpressionSet. This function builds a new dictionary
    which has transform names (fast look-up) and only necessary data.

    Params
    ------
        expression_set : HexelExpressionSet from which to build the dictionary.

    Returns
    -------
        A pair of dicts of the format [trans_name, IPPMSpike(trans_name)], for left- and right-hemispheres respectively.
            Each transform will start with a single spike containing all the significant hexels.
    """
    spikes_left = {}
    spikes_right = {}
    for hemi, best_transforms, spikes_dict in zip([HEMI_LEFT, HEMI_RIGHT], expression_set.best_transforms(), [spikes_left, spikes_right]):
        for ep in best_transforms:
            if ep.transform not in spikes_dict:
                spikes_dict[ep.transform] = IPPMSpike(ep.transform)
            spikes_dict[ep.transform].add_pairing(ExpressionPairing(ep.latency, ep.logp_value))
    return spikes_left, spikes_right


def build_spike_dict_from_sensor_expression_set(expression_set: SensorExpressionSet) -> SpikeDict:
    """
    Builds the spike dictionary from a SensorExpressionSet. This function builds a new dictionary
    which has transform names (fast look-up) and only necessary data.

    Params
    ------
        expression_set : SensorExpressionSet from which to build the dictionary.

    Returns
    -------
        Dict of the format [trans_name, IPPMSpike(trans_name)]. Each transform will start with a single spike containing
            all the significant hexels.
    """
    spikes = {}
    for ep in expression_set.best_transforms():
        if ep.transform not in spikes:
            spikes[ep.transform] = IPPMSpike(ep.transform)
        spikes[ep.transform].add_pairing(ExpressionPairing(ep.latency, ep.logp_value))
    return spikes
