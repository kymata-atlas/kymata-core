"""
Tests for kymata.ippm.evaluation
"""
from kymata.entities.expression import ExpressionPoint
from kymata.ippm.evaluation import causality_violation_score, transform_recall, null_edge_difference
from kymata.ippm.graph import IPPMGraph

def test_null_edge_difference_with_no_null_edges():
    """
        There are 2 null edges, one for f3 and one for f5
         Map 1 is f0 -> f1 -> f2              -> f5 -> f5
                                -> f3 -> f3 
                                -> f4 ->
    """
    ctl = {"f5": ["f4", "f3"], "f4": ["f2"], "f3": ["f2"], "f2": ["f1"], "f1": ["f0"], "f0": []}
    map1_points = [
        ExpressionPoint("c1", 50,  "f1", -50),
        ExpressionPoint("c2", 72, "f2", -25),
        ExpressionPoint("c3", 90, "f3", -77),
        ExpressionPoint("c4", 100, "f4", -39),
        ExpressionPoint("c5", 140, "f5", -75),
    ]
    map2_points = [
        ExpressionPoint("c1", 50,  "f1", -50),
        ExpressionPoint("c2", 72, "f2", -25),
        ExpressionPoint("c3", 90, "f3", -77),
        ExpressionPoint("c4", 100, "f4", -39),
    ]
    map1 = IPPMGraph(ctl, points_by_block={"merged": map1_points})
    map2 = IPPMGraph(ctl, points_by_block={"merged": map2_points})

    wtd = null_edge_difference(map1, map2)
    assert wtd == 0


def test_null_edge_difference_with_perfect_agreement():
    """
        There are 2 null edges, one for f3 and one for f5
         Map 1 is f0 -> f1 -> f2              -> f5 -> f5
                                -> f3 -> f3 
                                -> f4 ->
    """
    ctl = {"f5": ["f4", "f3"], "f4": ["f2"], "f3": ["f2"], "f2": ["f1"], "f1": ["f0"], "f0": []}
    map1_points = [
        ExpressionPoint("c1", 50,  "f1", -50),
        ExpressionPoint("c2", 72, "f2", -25),
        ExpressionPoint("c3", 80,  "f3", -55),
        ExpressionPoint("c4", 90, "f3", -77),
        ExpressionPoint("c5", 100, "f4", -39),
        ExpressionPoint("c6", 110, "f5", -48),
        ExpressionPoint("c7", 140, "f5", -75),
    ]
    map2_points = [
        ExpressionPoint("c1", 72, "f2", -25),
        ExpressionPoint("c2", 80,  "f3", -55),
        ExpressionPoint("c3", 90, "f3", -77),
        ExpressionPoint("c4", 100, "f4", -39),
        ExpressionPoint("c5", 110, "f5", -48),
        ExpressionPoint("c6", 140, "f5", -75),
    ]
    map1 = IPPMGraph(ctl, points_by_block={"merged": map1_points})
    map2 = IPPMGraph(ctl, points_by_block={"merged": map2_points})

    wtd = null_edge_difference(map1, map2)
    assert wtd == 0

def test_null_edge_difference_with_perfect_disagreement():
    """
        There are 2 null edges, one for f3 and one for f5
        Map 1 is f0 -> f1 -> f2              -> f5 -> f5
                                -> f3 -> f3 
                                -> f4 ->
    """
    ctl = {"f5": ["f4", "f3"], "f4": ["f2"], "f3": ["f2"], "f2": ["f1"], "f1": ["f0"], "f0": []}
    map1_points = [
        ExpressionPoint("c1", 50,  "f1", -50),
        ExpressionPoint("c2", 72, "f2", -25),
        ExpressionPoint("c3", 80,  "f3", -55),
        ExpressionPoint("c4", 90, "f3", -77),
        ExpressionPoint("c5", 100, "f4", -39),
        ExpressionPoint("c6", 110, "f5", -48),
        ExpressionPoint("c7", 140, "f5", -75),
    ]
    map2_points = [
        # it is missing all null edges => perfect disagreement
        ExpressionPoint("c1", 50,  "f1", -50),
        ExpressionPoint("c2", 72, "f2", -25),
        ExpressionPoint("c3", 90, "f3", -77),
        ExpressionPoint("c4", 100, "f4", -39),
        ExpressionPoint("c5", 140, "f5", -75),
    ]
    map1 = IPPMGraph(ctl, points_by_block={"merged": map1_points})
    map2 = IPPMGraph(ctl, points_by_block={"merged": map2_points})

    wtd = null_edge_difference(map1, map2)
    assert wtd == 1

def test_null_edge_difference_with_partial_agreement():
    """
        There are 2 null edges, one for f3 and one for f5
         Map 1 is f0 -> f1 -> f2              -> f5 -> f5
                                -> f3 -> f3 
                                -> f4 ->
    """
    ctl = {"f5": ["f4", "f3"], "f4": ["f2"], "f3": ["f2"], "f2": ["f1"], "f1": ["f0"], "f0": []}
    map1_points = [
        ExpressionPoint("c1", 50,  "f1", -50),
        ExpressionPoint("c2", 72, "f2", -25),
        ExpressionPoint("c3", 80,  "f3", -55),
        ExpressionPoint("c4", 90, "f3", -77),
        ExpressionPoint("c5", 100, "f4", -39),
        ExpressionPoint("c6", 110, "f5", -48),
        ExpressionPoint("c7", 140, "f5", -75),
    ]
    map2_points = [
        # it is missing all null edges => perfect disagreement
        ExpressionPoint("c1", 50,  "f1", -50),
        ExpressionPoint("c2", 80,  "f3", -55),
        ExpressionPoint("c3", 90, "f3", -77),
        ExpressionPoint("c4", 100, "f4", -39),
        ExpressionPoint("c5", 140, "f5", -75),
    ]
    map1 = IPPMGraph(ctl, points_by_block={"merged": map1_points})
    map2 = IPPMGraph(ctl, points_by_block={"merged": map2_points})

    wtd = null_edge_difference(map1, map2)
    assert wtd == 0.5


def test_causality_violation_with_right_hemi_should_succeed():
    test_points = [
        ExpressionPoint("c", 50,  "f1", -50),
        ExpressionPoint("c", 100, "f1", -25),
        ExpressionPoint("c", 75,  "f2", -55),
        ExpressionPoint("c", 110, "f2", -77),
        ExpressionPoint("c", 120, "f3", -39),
        ExpressionPoint("c", 100, "f4", -19),
        ExpressionPoint("c", 150, "f4", -75),
    ]
    test_hierarchy = {"f4": ["f3"], "f3": ["f1", "f2"], "f2": ["f1"], "f1": []}

    ippm_graph = IPPMGraph(test_hierarchy, points_by_block={"scalp": test_points})
    ratio, num, denom = causality_violation_score(ippm_graph)

    assert ratio == 0.25
    assert num == 1
    assert denom == 4


def test_causality_violation_with_single_function_should_return_0():
    test_points = [
        ExpressionPoint("c1", 50, "f1", -50),
        ExpressionPoint("c2", 100, "f1", -25),
    ]
    test_hierarchy = {"f1": []}

    ippm_graph = IPPMGraph(test_hierarchy, points_by_block={"scalp": test_points})
    ratio, num, denom = causality_violation_score(ippm_graph)

    assert ratio == 0
    assert num == 0
    assert denom == 0 # Added this assertion as it's logically correct for no arrows


def test_causality_violation_with_single_edge_should_return_0():
    test_points = [
        ExpressionPoint("c1", 50, "f1", -50),
        ExpressionPoint("c2", 100, "f1", -25),
        ExpressionPoint("c3", 110, "f2", -50),
    ]
    test_hierarchy = {"f2": ["f1"], "f1": []}

    ippm_graph = IPPMGraph(test_hierarchy, points_by_block={"scalp": test_points})
    ratio, num, denom = causality_violation_score(ippm_graph)

    assert ratio == 0
    assert num == 0
    assert denom == 1 # Expected 1 arrow: f1 -> f2


def test_transform_recall_with_no_trans_should_return_0():
    noisy_points = [
        ExpressionPoint("c1", 10, "f1", -30),
        ExpressionPoint("c2", 15, "f1", -35),
        ExpressionPoint("c3", 25, "f2", -50),
        ExpressionPoint("c3", 30, "f2", -2),
    ]
    denoised_points = []

    test_graph = IPPMGraph(dict(), points_by_block={"scalp": denoised_points})

    ratio, numer, denom = transform_recall(test_graph, noisy_points)

    assert ratio == 0
    assert numer == 0
    assert denom == 2 # f1 and f2 are present in noisy_points


def test_transform_recall_with_all_trans_found_should_return_1():
    noisy_points = [
        ExpressionPoint("c1", 10, "f1", -30),
        ExpressionPoint("c2", 15, "f1", -35),
        ExpressionPoint("c3", 25, "f2", -50),
        ExpressionPoint("c3", 30, "f2", -2),
    ]
    denoised_points = [
        ExpressionPoint("c1", 10, "f1", -30),
        ExpressionPoint("c2", 15, "f1", -35),
        ExpressionPoint("c3", 25, "f2", -50),
        ExpressionPoint("c3", 30, "f2", -2),
    ]

    test_graph = IPPMGraph({"input": [], "f1": ["input"], "f2": ["f1"]}, points_by_block={"scalp": denoised_points})

    ratio, numer, denom = transform_recall(test_graph, noisy_points)

    assert ratio == 1
    assert numer == 2 # f1, f2 detected
    assert denom == 2 # f1, f2 in noisy data


def test_transform_recall_with_half_trans_found_should_return_correct_ratio():
    noisy_points = [
        ExpressionPoint("c1", 10, "f1", -30),
        ExpressionPoint("c2", 15, "f1", -35),
        ExpressionPoint("c3", 25, "f2", -50),
        ExpressionPoint("c3", 30, "f2", -2),
    ]
    denoised_points = [
        ExpressionPoint("c1", 10, "f1", -30),
        ExpressionPoint("c2", 15, "f1", -35),
    ]

    test_graph = IPPMGraph({"input": [], "f1": ["input"], }, points_by_block={"scalp": denoised_points})

    ratio, numer, denom = transform_recall(test_graph, noisy_points)

    assert ratio == 1 / 2
    assert numer == 1 # f1 detected
    assert denom == 2 # f1, f2 in noisy data