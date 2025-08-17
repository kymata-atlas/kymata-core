"""
Metrics for evaluating IPPMs
"""

from kymata.entities.expression import ExpressionPoint
from kymata.ippm.graph import IPPMGraph
from kymata.ippm.hierarchy import group_points_by_transform


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
        if transform not in points_by_transform.keys() or len(points_by_transform[transform]) == 0:
            continue

        earliest_latency_this_trans = _point_with_min_latency(points_by_transform[transform])

        upstream_transforms = ippm.candidate_transform_list.immediately_upstream(transform)
        for upstream in upstream_transforms:
            # Get latest upstream latency
            if upstream in ippm.candidate_transform_list.inputs:
                latest_latency_upstream = 0
            else:
                try:
                    upstream_points = points_by_transform[upstream]
                    if len(upstream_points) == 0:
                        continue
                # If the upstream function has no points, it can't cause a causality violation
                except KeyError:
                    continue
                latest_latency_upstream = _point_with_max_latency(upstream_points).latency

            # Check for causality violation
            if earliest_latency_this_trans.latency < latest_latency_upstream:
                causality_violations += 1
                print(f"Counting violation between {transform} and {upstream}")
            total_arrows += 1
            print(f"Counting edge between {transform} and {upstream}")

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
        if n.transform not in ippm_graph.inputs
    )

    n_detected_transforms = len(trans_present_in_graph)
    n_transforms_in_data = len(trans_present_in_original_data)

    return (
        n_detected_transforms / n_transforms_in_data if n_transforms_in_data > 0 else 0,  # ratio
        n_detected_transforms,  # num
        n_transforms_in_data,   # denom
    )


def null_edge_difference(graph1: IPPMGraph, graph2: IPPMGraph) -> float:
    '''
        This metric is used to detect extra/missing null edges between two IPPMs. 
        Let S1 = set of null edges in IPPM 1, i.e., { (u,v) in IPPM_1 such that u.transform == v.transform }
            S2 = set of null edges in IPPM 2
        
        Then,
            null_edge_difference (NED): | (S1 UNION S2) DIFFERENCE (S1 INTERSECTION S2) | divided by | S1 UNION S2 |

        CONCEPTUAL EXPLANATION: what proportion of null edges across both maps is extra or missing? This is the question this metric answers
        If you get 0, then the maps agree perfectly on the null edges
        If you get 1, then the maps disagree completely on the null edges.
        If you get (0, 1), then the maps partially agree on the null edges.

        In terms of importance, TR is most important, then CV, then NED 
    '''
    assert graph1.candidate_transform_list == graph2.candidate_transform_list, "CTLs must be the same for both IPPMs!"

    s1 = _generate_null_edge_set(graph1)
    s2 = _generate_null_edge_set(graph2)
    union = s1.union(s2)
    intersection = s1.intersection(s2)
    if len(union) > 0:
        return len(union.difference(intersection)) / len(union)
    return 0


def _generate_null_edge_set(graph: IPPMGraph) -> set:
    ctl = graph.candidate_transform_list
    # We need to do it per-transform to ensure labelling is consistent for each IPPM, i.e., add same index
    null_edges_per_transform = {transform: [] for transform in ctl.transforms}
    for edge_from, edge_to in graph.graph_full.edges:
        if edge_from.transform != edge_to.transform:
            continue

        transform = edge_from.transform
        edge_label = f"{transform}_{len(null_edges_per_transform[transform])}"
        null_edges_per_transform[transform].append(edge_label)
    
    return set([edge for edge_list in null_edges_per_transform.values() for edge in edge_list]) # unpack into set of null edge labels of form transform_idx



