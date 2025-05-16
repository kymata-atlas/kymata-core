"""
Metrics for evaluating IPPMs
"""
from numpy import sign

from kymata.entities.expression import ExpressionPoint
from kymata.ippm.graph import IPPMGraph, IPPMConnectionStyle
from kymata.ippm.hierarchy import group_points_by_transform


def relative_causality_violation_score(ippm_1: IPPMGraph, ippm_2: IPPMGraph,
                                       connection_style: IPPMConnectionStyle = IPPMConnectionStyle.first_to_first,
                                       ) -> tuple[float, int, int]:
    """
    Computes the relative causality violation score between two IPPMs for a given IPPMConnectionStyle.

    Args:
        ippm_1 (IPPMGraph):
        ippm_2 (IPPMGraph):
        connection_style (IPPMConnectionStyle, optional): The connection style to use. Must be one where there is at
            most one edge between any pair of transforms. Defaults to IPPMConnectionStyle.first_to_first.

    Returns:
        float: violations / total edges
        int: violations
        int: total edges
    """
    # Will only work where there's at most one edge connecting each pair of transforms
    if connection_style not in {IPPMConnectionStyle.first_to_first, IPPMConnectionStyle.last_to_first}:
        raise NotImplementedError(f"IPPMs must have at most one edge connecting each pair of transforms:"
                                  f" {connection_style} is unsupported")

    # First we assert that the CTL of each graph is the same
    if ippm_1.candidate_transform_list != ippm_2.candidate_transform_list:
        raise ValueError("IPPMs must have the same CTL")

    # Now we can traverse the nodes and edges in the CTL and check for edges in each downstream graph.
    # Note that this means we automatically ignore edges between runs of the same transform.
    violations = 0
    edges_intersection = 0
    for edge in ippm_1.candidate_transform_list.graph.edges:
        if not edge in ippm_1.graph_full.edges:
            continue
        if not edge in ippm_2.graph_full.edges:
            continue

        # Get the corresponding edge from each graph
        # (We know there will be only one of each
        ippm_1_source, ippm_1_target = ippm_1.edges_between_transforms(*edge)[0]
        ippm_2_source, ippm_2_target = ippm_2.edges_between_transforms(*edge)[0]
        order_in_ippm_1 = sign(ippm_1_target.latency - ippm_1_source.latency)
        order_in_ippm_2 = sign(ippm_2_target.latency - ippm_2_source.latency)
        if order_in_ippm_1 != order_in_ippm_2:
            violations += 1
        edges_intersection += 1

    return (
        violations / edges_intersection if edges_intersection != 0 else 0,
        violations,
        edges_intersection,
    )


def causality_violation_score(ippm: IPPMGraph) -> tuple[float, int, int]:
    """
    Assumption: expression points are denoised. Otherwise, it doesn't really make sense to check the min/max latency of
    noisy expression points.

    A score calculated on denoised expression points that calculates the proportion of arrows in IPPM that are going
    backward in time. It assumes that the function hierarchy is correct, which may not always be correct, so you must
    use it with caution.

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

    points_by_transform = group_points_by_transform(ippm.graph_full.nodes)

    causality_violations = 0
    total_arrows = 0
    for transform in ippm.candidate_transform_list.transforms:
        # essentially: if max(parent_spikes_latency) > min(child_spikes_latency), there will be a backwards arrow in
        # time.
        # arrows go from latest inc_edge spike to the earliest func spike

        # Inputs can't cause a causality violation
        if transform in ippm.candidate_transform_list.inputs:
            continue

        # If there aren't any points for this transform it can't cause a causality violation
        if len(points_by_transform[transform]) == 0:
            continue

        earliest_latency_this_trans = _point_with_min_latency(points_by_transform[transform])

        upstream_transforms = ippm.candidate_transform_list.immediately_upstream(transform)
        for upstream in upstream_transforms:
            # Get latest upstream latency
            if upstream in ippm.candidate_transform_list.inputs:
                latest_latency_upstream = 0
            else:
                upstream_points = points_by_transform[upstream]
                if len(upstream_points) == 0:
                    continue
                latest_latency_upstream = _point_with_max_latency(upstream_points).latency

            # Check for causality violation
            if earliest_latency_this_trans.latency < latest_latency_upstream:
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


def transform_recall(ippm_graph: IPPMGraph, noisy_points: list[ExpressionPoint]) -> tuple[float, int, int]:
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

    In case the denominator is 0, the ratio will be set to 0 rather than raise an exception.
    """

    trans_present_in_original_data = set(
        trans
        for trans, points in group_points_by_transform(noisy_points).items()
        if len(points) > 0
    )
    trans_present_in_graph = set(
        n.transform
        for n in ippm_graph.graph_last_to_first.nodes
    )

    n_detected_transforms = len(trans_present_in_graph)
    n_transforms_in_data = len(trans_present_in_original_data)

    return (
        n_detected_transforms / n_transforms_in_data if n_transforms_in_data > 0 else 0,  # ratio
        n_detected_transforms,  # num
        n_transforms_in_data,   # denom
    )
