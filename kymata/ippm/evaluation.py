"""
Metrics for evaluating IPPMs
"""
from statistics import NormalDist

from kymata.entities.constants import HEMI_LEFT, HEMI_RIGHT
from kymata.entities.expression import ExpressionPoint
from kymata.ippm.build import IPPMGraph, SpikeDict
from kymata.ippm.hierarchy import TransformHierarchy


def causality_violation_score(
    denoised_spikes: SpikeDict,
    hierarchy: TransformHierarchy,
    hemi: str,
    inputs: list[str],
) -> tuple[float, int, int]:
    """
    Assumption: spikes are denoised. Otherwise, it doesn't really make sense to check the min/max latency of noisy spikes.

    A score calculated on denoised spikes that calculates the proportion of arrows in IPPM that are going backward in time.
    It assumes that the function hierarchy is correct, which may not always be correct, so you must use it with caution.

    Algorithm
    ----------
    violations = 0
    total_arrows = 0
    for each func_name, parents_list in hierarchy:
        child_lat = min(spikes[func])
        for parent in parents_list:
            parent_lat = max(spikes[parent])
            if child_lat < parent_lat:
                violations++
            total_arrows++
    return violations / total_arrows if total_arrows > 0 else 0
    """

    assert hemi == HEMI_LEFT or hemi == HEMI_RIGHT

    def get_latency(trans_points: list[ExpressionPoint], mini: bool):
        return (
            (
                min(trans_points, key=lambda x: x.latency)
                if hemi == HEMI_LEFT
                else min(trans_points, key=lambda x: x.latency)
            )
            if mini
            else (
                max(trans_points, key=lambda x: x.latency)
                if hemi == HEMI_LEFT
                else max(trans_points, key=lambda x: x.latency)
            )
        )

    causality_violations = 0
    total_arrows = 0
    for trans, inc_edges in hierarchy.items():
        # essentially: if max(parent_spikes_latency) > min(child_spikes_latency), there will be a backwards arrow in time.
        # arrows go from latest inc_edge spike to the earliest func spike

        if trans in inputs:
            continue

        if hemi == HEMI_LEFT:
            if len(denoised_spikes[trans]) == 0:
                continue
        else:
            if len(denoised_spikes[trans]) == 0:
                continue

        child_latency = get_latency(denoised_spikes[trans], mini=True)[0]
        for inc_edge in inc_edges:
            if inc_edge in inputs:
                # input node, so parent latency is 0
                parent_latency = 0
                if child_latency < parent_latency:
                    causality_violations += 1
                total_arrows += 1
                continue

            # We need to ensure the function has significant spikes
            if hemi == HEMI_LEFT:
                if len(denoised_spikes[inc_edge]) == 0:
                    continue
            else:
                if len(denoised_spikes[inc_edge]) == 0:
                    continue

            parent_latency = get_latency(denoised_spikes[inc_edge], mini=False)[0]
            if child_latency < parent_latency:
                causality_violations += 1
            total_arrows += 1

    return (
        causality_violations / total_arrows if total_arrows != 0 else 0,
        causality_violations,
        total_arrows,
    )


def transform_recall(
    noisy_spikes: SpikeDict,
    transforms: list[str],
    ippm_dict: IPPMGraph,
    hemi: str,
) -> tuple[float, int, int]:
    """
    This is the second scoring metric: transform recall. It illustrates what proportion out of functions in the
    noisy spikes are detected as part of IPPM. E.g., 9 functions but only 8 found => 8/9 = function recall. Use this
    along with causality violation to evaluate IPPMs and analyse their strengths and weaknesses.

    One thing to note is that the recall depends upon the nature of the dataset. If certain functions have no
    significant spikes, there is an inherent bias present in the dataset. We can never get the function recall to be
    perfect no matter what algorithm we employ. Therefore, the function recall is based on what we can actually do
    with a dataset. E.g., 9 functions in the hierarchy but in the noisy spikes we find only 7 of the 9 functions.
    Moreover, after denoising we find that there are only 6 functions in the hierarchy. The recall will be 6/7
    rather than 6/9 since there were only 7 to be found to begin with.

    Params
    ------
    spikes: the noisy spikes that we denoise and feed into IPPMBuilder. It must be the same dataset.
    funcs: list of functions that are in our hierarchy. Don't include the input function, e.g., input_cochlear.
    ippm_dict: the return value from IPPMBuilder. It contains node names as keys and Node objects as values.
    hemi: left or right

    Returns
    -------
    A ratio indicating how many channels were incorporated into the IPPM out of all relevant channels.
    """
    assert hemi == HEMI_RIGHT or hemi == HEMI_LEFT

    # Step 1: Calculate significance level
    # TODO: hard-coded alpha
    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    # TODO: hard-coded correction
    bonferroni_corrected_alpha = 1 - (pow((1 - alpha), (1 / (2 * 201 * 200000))))
    funcs_present_in_data = 0
    detected_funcs = 0
    for trans in transforms:
        point: ExpressionPoint
        for point in noisy_spikes[trans]:
            # Step 2: Find a pairing that is significant
            if point.logp_value <= bonferroni_corrected_alpha:
                funcs_present_in_data += 1

                # Step 3: Found a function, look in ippm_dict.keys() for the function.
                for node_name in ippm_dict.keys():
                    if trans in node_name:
                        # Step 4: If found, then increment detected_funcs. Also increment funcs_pressent
                        detected_funcs += 1
                        break
                break

    # Step 3: Return [ratio, numerator, denominator] primarily because both the denominator and numerator can vary.
    return (
        detected_funcs / funcs_present_in_data if funcs_present_in_data > 0 else 0,
        detected_funcs,
        funcs_present_in_data,
    )
