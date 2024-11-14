from typing import NamedTuple, Optional

from kymata.entities.constants import HEMI_LEFT, HEMI_RIGHT
from kymata.entities.expression import HexelExpressionSet, DIM_TRANSFORM, DIM_LATENCY, COL_LOGP_VALUE
from kymata.io.atlas import fetch_data_dict
from kymata.math.p_values import logp_to_p


class ExpressionPairing(NamedTuple):
    """
    A temporal location representing evidence of expression with an associated p-value.
    """
    latency_ms: float
    logp_value: float


class IPPMSpike(object):
    """
    Container to hold data about a spike.

    Attributes
    ----------
        transform : the name of the transform who caused the spike
        right_best_pairings : right hemisphere's best timings for this transform
        left_best_pairings : right hemisphere's best timings for this transform
        description : optional written description
        github_commit : github commit of the transform
    """

    def __init__(
        self,
        transform_name: str,
        description: str = None,
        github_commit: str = None,
    ):
        self.transform: str = transform_name
        self.right_best_pairings: list[ExpressionPairing] = []
        self.left_best_pairings: list[ExpressionPairing] = []
        self.description: Optional[str] = description
        self.github_commit: str = github_commit
        self.color: Optional[str] = None

        self.input_stream: Optional[str] = None

    def add_pairing(self, hemi: str, expr_timing: ExpressionPairing):
        """
        Use this to add new timings.

        Params
        ------
            hemi : left or right
            timing : Corresponds to the best match to a spike
        """
        if hemi == HEMI_LEFT:
            self.left_best_pairings.append(expr_timing)
        else:
            self.right_best_pairings.append(expr_timing)


SpikeDict = dict[str, IPPMSpike]

# Maps function names to lists of parent functions
TransformHierarchy = dict[str, list[str]]


def fetch_spike_dict(api: str) -> SpikeDict:
    """
    Fetches data from Kymata API and converts it into a dictionary of transform names as keys
    and spike objects as values. Advantage of dict is O(1) look-up and spike object is readable
    access to attributes.

    Params
    ------
        api : URL of the API from which to fetch data

    Returns
    -------
        Dictionary containing data in the format [transform name, spike]
    """
    return build_spike_dict_from_api_response(fetch_data_dict(api))


def build_spike_dict_from_expression_set(expression_set: HexelExpressionSet) -> SpikeDict:
    """
    Builds the dictionary from an ExpressionSet. This function builds a new dictionary
    which has transform names (fast look-up) and only necessary data.

    Params
    ------
        dict_ : JSON dictionary of HTTP GET response object.

    Returns
    -------
        Dict of the format [transform name, spike(trans_name, id, left_pairings, right_timings)]
    """
    spikes = {}
    for hemi, best_transforms in zip([HEMI_LEFT, HEMI_RIGHT], expression_set.best_transforms()):
        for _idx, row in best_transforms.iterrows():
            trans = row[DIM_TRANSFORM]
            latency = row[DIM_LATENCY] * 1000  # convert to ms
            logp = row[COL_LOGP_VALUE]
            if trans not in spikes:
                spikes[trans] = IPPMSpike(trans)
            spikes[trans].add_pairing(hemi, ExpressionPairing(latency, logp_to_p(logp)))
    return spikes


def build_spike_dict_from_api_response(dict_: dict) -> SpikeDict:
    """
    Builds the dictionary from response dictionary. Response dictionary has unneccesary
    keys and does not have transform names as keys. This function builds a new dictionary
    which has transform names (fast look-up) and only necessary data.

    Params
    ------
        dict_ : JSON dictionary of HTTP GET response object.

    Returns
    -------
        Dict of the format [transform name, spike(trans_name, id, left_timings, right_timings)]
    """
    spikes = {}
    for hemi in [HEMI_LEFT, HEMI_RIGHT]:
        for _, latency, pval, trans in dict_[hemi]:
            # we have id, latency (ms), pvalue (log_10), transform name.
            # discard id as it conveys no useful information
            if trans not in spikes:
                # first time seeing transform, so create key and spike object.
                spikes[trans] = IPPMSpike(trans)

            spikes[trans].add_pairing(hemi, ExpressionPairing(latency, pow(10, pval)))

    return spikes


def remove_excess_transforms(to_retain: list[str], spikes: SpikeDict) -> SpikeDict:
    """
    Utility function to distill the spikes down to a subset of transforms. Use this to visualise a subset of transforms
    for time-series. E.g., you want the time-series for one transform, so just pass it wrapped in a list as to_retain

    Parameters
    ----------
    to_retain: list of transforms we want to retain in the spikes dict
    spikes: dict transform_name as key and spike object as value. Spikes contain timings for left/right.

    Returns
    -------
    spikes but all transforms that aren't in to_retain are filtered.
    """

    transforms = list(
        spikes.keys()
    )  # need this because we can't remove from the dict while also iterating over it.
    for trans in transforms:
        if trans not in to_retain:
            # delete
            spikes.pop(trans)
    return spikes


def copy_hemisphere(
    spikes_to: SpikeDict,
    spikes_from: SpikeDict,
    hemi_to: str,
    hemi_from: str,
    trans: str = None,
):
    """
    Utility function to copy a hemisphere onto another one. The primary use-case is to plot the denoised hemisphere against the
    noisy hemisphere using the same spike object. I.e., copy right hemisphere to left; denoise on right; plot right vs left.

    Parameters
    ----------
    spikes_to: Spikes we are writing into. Could be (de)noisy spikes.
    spikes_from: Spikes we are copying from
    hemi_to: the hemisphere we index into when we write into spikes_to. E.g., spikes_to[hemi_to] = spikes_from[hemi_from]
    hemi_from: the hemisphere we index into when we copy the spikes from spikes_from.
    trans: if != None, we only copy one transform. Otherwise, we copy all.

    Returns
    -------
    Nothing, everything is done in-place. I.e., spikes_to is now updated.
    """
    if trans:
        # copy only one transform
        if hemi_to == HEMI_RIGHT and hemi_from == HEMI_RIGHT:
            spikes_to[trans].right_best_pairings = spikes_from[trans].right_best_pairings
        elif hemi_to == HEMI_RIGHT and hemi_from == HEMI_LEFT:
            spikes_to[trans].right_best_pairings = spikes_from[trans].left_best_pairings
        elif hemi_to == HEMI_LEFT and hemi_from == HEMI_RIGHT:
            spikes_to[trans].left_best_pairings = spikes_from[trans].right_best_pairings
        else:
            spikes_to[trans].left_best_pairings = spikes_from[trans].left_best_pairings
        return

    for trans in spikes_from.keys():
        if hemi_to == HEMI_RIGHT and hemi_from == HEMI_RIGHT:
            spikes_to[trans].right_best_pairings = spikes_from[trans].right_best_pairings
        elif hemi_to == HEMI_RIGHT and hemi_from == HEMI_LEFT:
            spikes_to[trans].right_best_pairings = spikes_from[trans].left_best_pairings
        elif hemi_to == HEMI_LEFT and hemi_from == HEMI_RIGHT:
            spikes_to[trans].left_best_pairings = spikes_from[trans].right_best_pairings
        else:
            spikes_to[trans].left_best_pairings = spikes_from[trans].left_best_pairings
