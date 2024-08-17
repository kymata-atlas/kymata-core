"""
Tests for kymata.ippm.evaluation
"""
import pytest

from kymata.entities.constants import HEMI_RIGHT, HEMI_LEFT
from kymata.ippm.data_tools import IPPMSpike, IPPMNode
from kymata.ippm.evaluation import causality_violation_score, transform_recall


def test_causality_violation_with_right_hemi_should_succeed():
    test_hexels = {
        "f1": IPPMSpike("f1"),
        "f2": IPPMSpike("f2"),
        "f3": IPPMSpike("f3"),
        "f4": IPPMSpike("f4"),
    }
    test_hexels["f1"].right_best_pairings = [(50, 1e-50), (100, 1e-25)]
    test_hexels["f2"].right_best_pairings = [(75, 1e-55), (110, 1e-77)]
    test_hexels["f3"].right_best_pairings = [(120, 1e-39)]
    test_hexels["f4"].right_best_pairings = [(100, 1e-19), (150, 1e-75)]
    test_hierarchy = {"f4": ["f3"], "f3": ["f1", "f2"], "f2": ["f1"], "f1": []}

    assert causality_violation_score(test_hexels, test_hierarchy, HEMI_RIGHT, ["f1"]) == (0.25, 1, 4)


def test_causality_violation_with_left_hemi_should_succeed():
    test_hexels = {
        "f1": IPPMSpike("f1"),
        "f2": IPPMSpike("f2"),
        "f3": IPPMSpike("f3"),
        "f4": IPPMSpike("f4"),
    }
    test_hexels["f1"].left_best_pairings = [(50, 1e-50), (100, 1e-25)]
    test_hexels["f2"].left_best_pairings = [(75, 1e-55), (110, 1e-77)]
    test_hexels["f3"].left_best_pairings = [(120, 1e-39)]
    test_hexels["f4"].left_best_pairings = [(100, 1e-19), (150, 1e-75)]
    test_hierarchy = {"f4": ["f3"], "f3": ["f1", "f2"], "f2": ["f1"], "f1": []}
    assert causality_violation_score(
        test_hexels, test_hierarchy, HEMI_LEFT, ["f1"]
    ) == (0.25, 1, 4)


def test_causality_violation_with_single_function_should_return_0():
    test_hexels = {"f1": IPPMSpike("f1")}
    test_hexels["f1"].left_best_pairings = [(50, 1e-50), (100, 1e-25)]
    test_hierarchy = {"f1": []}

    assert causality_violation_score(
        test_hexels, test_hierarchy, HEMI_LEFT, ["f1"]
    ) == (0, 0, 0)


def test_causality_violation_with_single_edge_should_return_0():
    test_hexels = {"f1": IPPMSpike("f1"), "f2": IPPMSpike("f2")}
    test_hexels["f1"].left_best_pairings = [(50, 1e-50), (100, 1e-25)]
    test_hexels["f2"].left_best_pairings = [(110, 1e-50)]
    test_hierarchy = {"f2": ["f1"], "f1": []}

    assert causality_violation_score(
        test_hexels, test_hierarchy, HEMI_LEFT, ["f1"]
    ) == (0, 0, 1)


def test_transform_recall_with_no_funcs_should_return_0():
    test_hexels = {"f1": IPPMSpike("f1"), "f2": IPPMSpike("f2")}
    test_hexels["f1"].left_best_pairings = []
    test_hexels["f2"].left_best_pairings = [
        (10, 1e-1)
    ]  # should be > alpha, so not significant
    test_ippm = {}
    funcs = ["f1", "f2"]
    ratio, numer, denom = transform_recall(
        test_hexels, funcs, test_ippm, HEMI_LEFT
    )

    assert ratio == 0
    assert numer == 0
    assert denom == 0


def test_transform_recall_with_all_funcs_found_should_return_1():
    test_hexels = {"f1": IPPMSpike("f1"), "f2": IPPMSpike("f2")}
    test_hexels["f1"].left_best_pairings = [(10, 1e-30), (15, 1e-35)]
    test_hexels["f2"].left_best_pairings = [(25, 1e-50), (30, 1e-2)]
    test_ippm = {
        "f1-0": IPPMNode(1e-30, 10, []),
        "f1-1": IPPMNode(1e-35, 15, ["f1-0"]),
        "f2-0": IPPMNode(1e-50, 25, ["f1-1"]),
    }
    funcs = ["f1", "f2"]
    ratio, numer, denom = transform_recall(
        test_hexels, funcs, test_ippm, HEMI_LEFT
    )

    assert ratio == 1
    assert numer == 2
    assert denom == 2


def test_transform_recall_with_invalid_hemi_input_should_raise_exception():
    with pytest.raises(AssertionError):
        transform_recall({}, [], {}, "invalidHemisphere")


def test_transform_recall_with_valid_input_right_hemi_should_return_success():
    test_hexels = {"f1": IPPMSpike("f1"), "f2": IPPMSpike("f2")}
    test_hexels["f1"].left_best_pairings = [(10, 1e-30), (15, 1e-35)]
    test_hexels["f2"].left_best_pairings = [(25, 1e-50), (30, 1e-2)]
    test_ippm = {"f1-0": IPPMNode(1e-30, 10, []), "f1-1": IPPMNode(1e-35, 15, ["f1-0"])}
    funcs = ["f1", "f2"]
    ratio, numer, denom = transform_recall(test_hexels, funcs, test_ippm, HEMI_LEFT)

    assert ratio == 1 / 2
    assert numer == 1
    assert denom == 2
