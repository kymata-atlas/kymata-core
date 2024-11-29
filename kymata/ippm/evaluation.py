"""
Metrics for evaluating IPPMs
"""

from kymata.entities.expression import ExpressionPoint
from kymata.ippm.graph import IPPMGraph
from kymata.ippm.hierarchy import group_points_by_transform


def causality_violation_score(ippm: IPPMGraph) -> tuple[float, int, int]:
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
    return violations / total_arrows
           (ratio, num, denom)
    """

    causality_violations = 0
    total_arrows = 0
    for trans in ippm.candidate_transform_list.transforms:
        inc_edges = ippm.candidate_transform_list.graph.in_edges(trans)
        # essentially: if max(parent_spikes_latency) > min(child_spikes_latency), there will be a backwards arrow in time.
        # arrows go from latest inc_edge spike to the earliest func spike

        if trans in ippm.candidate_transform_list.inputs:
            continue
        if len(ippm.points[trans]) == 0:
            continue

        child_latency = _point_with_min_latency(ippm.points[trans]).latency
        for inc_edge in inc_edges:
            if inc_edge in ippm.candidate_transform_list.inputs:
                # input node, so parent latency is 0
                parent_latency = 0
                if child_latency < parent_latency:
                    causality_violations += 1
                total_arrows += 1
                continue

            # We need to ensure the function has significant spikes
            if len(ippm.points[inc_edge]) == 0:
                continue

            parent_latency = _point_with_max_latency(ippm.points[trans]).latency
            if child_latency < parent_latency:
                causality_violations += 1
            total_arrows += 1

    return (
        causality_violations / total_arrows if total_arrows != 0 else 0,
        causality_violations,
        total_arrows,
    )


def _point_with_min_latency(trans_points: list[ExpressionPoint]) -> ExpressionPoint:
    return min(trans_points, key=lambda p: p.latency)


def _point_with_max_latency(trans_points: list[ExpressionPoint]) -> ExpressionPoint:
    return max(trans_points, key=lambda p: p.latency)


def transform_recall(ippm: IPPMGraph, noisy_points: list[ExpressionPoint]) -> tuple[float, int, int]:
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
    ippm: The IPPM graph to evaluate.
    noisy_points: the ExpressionPoints in the original (not denoised) dataset

    Returns
    -------
    A ratio indicating how many transforms were incorporated into the IPPM out of all relevant transforms.
    (ratio, num, denom)
    """

    trans_present_in_data = set(
        trans
        for trans, points in group_points_by_transform(noisy_points).items()
        if len(points) > 0
    )

    n_detected_transforms = len(ippm.graph_last_to_first.nodes)
    n_transforms_in_data = len(trans_present_in_data)

    return (
        n_detected_transforms / n_transforms_in_data if n_transforms_in_data > 0 else 0,
        n_detected_transforms,
        n_transforms_in_data,
    )
