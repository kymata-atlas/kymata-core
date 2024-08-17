import json
import math
from typing import NamedTuple, Optional

import requests

from kymata.entities.constants import HEMI_LEFT, HEMI_RIGHT
from kymata.entities.expression import HexelExpressionSet, DIM_FUNCTION, DIM_LATENCY, COL_LOGP_VALUE


class Pairing(NamedTuple):
    latency: float
    pvalue: float


class IPPMNode(NamedTuple):
    """
    A node to be drawn in an IPPM graph.
    """
    magnitude: float
    position: tuple[float, float]
    inc_edges: list


class IPPMSpike(object):
    """
    Container to hold data about a spike.

    Attributes
    ----------
        function : the name of the function who caused the spike
        right_best_pairings : right hemisphere best pairings. pvalues are taken to the base 10 by default. latency is in milliseconds
        left_best_pairings : right hemisphere best pairings. same info as right for (latency, pvalue)
        description : optional written description
        github_commit : github commit of the function
    """

    def __init__(
        self,
        function_name: str,
        description: str = None,
        github_commit: str = None,
    ):
        self.function: str = function_name
        self.right_best_pairings: list[Pairing] = []
        self.left_best_pairings: list[Pairing] = []
        self.description: str = description
        self.github_commit: str = github_commit
        self.color: Optional[str] = None

        self.input_stream: Optional[str] = None

    def add_pairing(self, hemi: str, pairing: Pairing):
        """
        Use this to add new pairings. Pair = (latency (ms), pvalue (log_10))

        Params
        ------
            hemi : left or right
            pairing : Corresponds to the best match to a spike of form (latency (ms), pvalue (log_10))
        """
        if hemi == HEMI_LEFT:
            self.left_best_pairings.append(pairing)
        else:
            self.right_best_pairings.append(pairing)


SpikeDict = dict[str, IPPMSpike]

# Maps function names to lists of parent functions
TransformHierarchy = dict[str, list[str]]

# Maps function names to nodes
IPPMGraph = dict[str, IPPMNode]


def fetch_data(api: str) -> SpikeDict:
    """
    Fetches data from Kymata API and converts it into a dictionary of function names as keys
    and spike objects as values. Advantage of dict is O(1) look-up and spike object is readable
    access to attributes.

    Params
    ------
        api : URL of the API from which to fetch data

    Returns
    -------
        Dictionary containing data in the format [function name, spike]
    """
    response = requests.get(api)
    resp_dict = json.loads(response.text)
    return build_spike_dict_from_api_response(resp_dict)


def build_spike_dict_from_expression_set(expression_set: HexelExpressionSet) -> SpikeDict:
    """
    Builds the dictionary from an ExpressionSet. This function builds a new dictionary
    which has function names (fast look-up) and only necessary data.

    Params
    ------
        dict_ : JSON dictionary of HTTP GET response object.

    Returns
    -------
        Dict of the format [function name, spike(func_name, id, left_pairings, right_pairings)]
    """
    best_functions_left, best_functions_right = expression_set.best_functions()
    spikes = {}
    for hemi in [HEMI_LEFT, HEMI_RIGHT]:
        best_functions = (
            best_functions_left if hemi == HEMI_LEFT else best_functions_right
        )
        for _idx, row in best_functions.iterrows():
            func = row[DIM_FUNCTION]
            latency = row[DIM_LATENCY] * 1000  # convert to ms
            pval = row[COL_LOGP_VALUE]
            if func not in spikes:
                spikes[func] = IPPMSpike(func)
            spikes[func].add_pairing(hemi, Pairing(latency, pval))
    return spikes


def build_spike_dict_from_api_response(dict_: dict) -> SpikeDict:
    """
    Builds the dictionary from response dictionary. Response dictionary has unneccesary
    keys and does not have function names as keys. This function builds a new dictionary
    which has function names (fast look-up) and only necessary data.

    Params
    ------
        dict_ : JSON dictionary of HTTP GET response object.

    Returns
    -------
        Dict of the format [function name, spike(func_name, id, left_pairings, right_pairings)]
    """
    spikes = {}
    for hemi in [HEMI_LEFT, HEMI_RIGHT]:
        for _, latency, pval, func in dict_[hemi]:
            # we have id, latency (ms), pvalue (log_10), function name.
            # discard id as it conveys no useful information
            if func not in spikes:
                # first time seeing function, so create key and spike object.
                spikes[func] = IPPMSpike(func)

            spikes[func].add_pairing(hemi, Pairing(latency, pow(10, pval)))

    return spikes


def convert_to_power10(spikes: SpikeDict) -> SpikeDict:
    """
    Utility function to take data from the .nkg format and convert it to power of 10, so it can be used for IPPMs.

    Parameters
    ------------
    spikes: dict function_name as key and spike object as value. Spikes contain pairings for left/right.

    Returns
    --------
    same dict but the pairings are all raised to power x. E.g., pairings = [(lat1, x), ..., (latn, xn)] -> [(lat1, 10^x), ..., (latn, 10^xn)]
    """
    for func in spikes.keys():
        spikes[func].right_best_pairings = list(
            map(lambda x: (x[0], math.pow(10, x[1])), spikes[func].right_best_pairings)
        )
        spikes[func].left_best_pairings = list(
            map(lambda x: (x[0], math.pow(10, x[1])), spikes[func].left_best_pairings)
        )
    return spikes


def remove_excess_funcs(to_retain: list[str], spikes: SpikeDict) -> SpikeDict:
    """
    Utility function to distill the spikes down to a subset of functions. Use this to visualise a subset of functions for time-series.
    E.g., you want the time-series for one function, so just pass it wrapped in a list as to_retain

    Parameters
    ----------
    to_retain: list of functions we want to retain in the spikes dict
    spikes: dict function_name as key and spike object as value. Spikes contain pairings for left/right.

    Returns
    -------
    spikes but all functions that aren't in to_retain are filtered.
    """

    funcs = list(
        spikes.keys()
    )  # need this because we can't remove from the dict while also iterating over it.
    for func in funcs:
        if func not in to_retain:
            # delete
            spikes.pop(func)
    return spikes


def copy_hemisphere(
    spikes_to: SpikeDict,
    spikes_from: SpikeDict,
    hemi_to: str,
    hemi_from: str,
    func: str = None,
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
    func: if func != None, we only copy one function. Otherwise, we copy all.

    Returns
    -------
    Nothing, everything is done in-place. I.e., spikes_to is now updated.
    """
    if func:
        # copy only one function
        if hemi_to == HEMI_RIGHT and hemi_from == HEMI_RIGHT:
            spikes_to[func].right_best_pairings = spikes_from[func].right_best_pairings
        elif hemi_to == HEMI_RIGHT and hemi_from == HEMI_LEFT:
            spikes_to[func].right_best_pairings = spikes_from[func].left_best_pairings
        elif hemi_to == HEMI_LEFT and hemi_from == HEMI_RIGHT:
            spikes_to[func].left_best_pairings = spikes_from[func].right_best_pairings
        else:
            spikes_to[func].left_best_pairings = spikes_from[func].left_best_pairings
        return

    for func in spikes_from.keys():
        if hemi_to == HEMI_RIGHT and hemi_from == HEMI_RIGHT:
            spikes_to[func].right_best_pairings = spikes_from[func].right_best_pairings
        elif hemi_to == HEMI_RIGHT and hemi_from == HEMI_LEFT:
            spikes_to[func].right_best_pairings = spikes_from[func].left_best_pairings
        elif hemi_to == HEMI_LEFT and hemi_from == HEMI_RIGHT:
            spikes_to[func].left_best_pairings = spikes_from[func].right_best_pairings
        else:
            spikes_to[func].left_best_pairings = spikes_from[func].left_best_pairings
