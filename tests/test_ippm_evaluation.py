"""
Tests for kymata.ippm.evaluation
"""
import pytest

from kymata.entities.constants import HEMI_RIGHT, HEMI_LEFT
from kymata.entities.expression import ExpressionPoint
from kymata.ippm.evaluation import causality_violation_score, transform_recall
from kymata.ippm.graph import IPPMGraph
from kymata.ippm.hierarchy import CandidateTransformList


def test_causality_violation_with_right_hemi_should_succeed():
    test_hexels = {
        "f1": [ExpressionPoint("c", 50, "f1", 1e-50), ExpressionPoint("c", 100, "f1", 1e-25)],
        "f2": [ExpressionPoint("c", 75, "f2", 1e-55), ExpressionPoint("c", 110, "f2", 1e-77)],
        "f3": [ExpressionPoint("c", 120, "f3", 1e-39)],
        "f4": [ExpressionPoint("c", 100, "f4", 1e-19), ExpressionPoint("c", 150, "f4", 1e-75)],
    }
    test_hierarchy = {"f4": ["f3"], "f3": ["f1", "f2"], "f2": ["f1"], "f1": []}

    assert causality_violation_score(test_hexels, test_hierarchy, HEMI_RIGHT, ["f1"]) == (0.25, 1, 4)


def test_causality_violation_with_left_hemi_should_succeed():
    test_hexels = {
        "f1": [ExpressionPoint("c", 50, "f1", 1e-50), ExpressionPoint("c", 100, "f1", 1e-25)],
        "f2": [ExpressionPoint("c", 75, "f2", 1e-55), ExpressionPoint("c", 110, "f2", 1e-77)],
        "f3": [ExpressionPoint("c", 120, "f3", 1e-39)],
        "f4": [ExpressionPoint("c", 100, "f4", 1e-19), ExpressionPoint("c", 150, "f4", 1e-75)],
    }
    test_hierarchy = {"f4": ["f3"], "f3": ["f1", "f2"], "f2": ["f1"], "f1": []}
    assert causality_violation_score(
        test_hexels, test_hierarchy, HEMI_LEFT, ["f1"]
    ) == (0.25, 1, 4)


def test_causality_violation_with_single_function_should_return_0():
    test_hexels = {"f1": [ExpressionPoint("c1", 50, "f1", 1e-50), ExpressionPoint("c2", 100, "f1", 1e-25)]}
    test_hierarchy = {"f1": []}

    assert causality_violation_score(
        test_hexels, test_hierarchy, HEMI_LEFT, ["f1"]
    ) == (0, 0, 0)


def test_causality_violation_with_single_edge_should_return_0():
    test_hexels = {
        "f1": [ExpressionPoint("c1", 50, "f1", 1e-50), ExpressionPoint("c2", 100, "f1", 1e-25)],
        "f2": [ExpressionPoint("c3", 110, "f2", 1e-50)],
    }
    test_hierarchy = {"f2": ["f1"], "f1": []}

    assert causality_violation_score(
        test_hexels, test_hierarchy, HEMI_LEFT, ["f1"]
    ) == (0, 0, 1)


def test_transform_recall_with_no_trans_should_return_0():
    noisy_points = [
        ExpressionPoint("c1", 10, "f1", -30),
        ExpressionPoint("c2", 15, "f1", -35),
        ExpressionPoint("c3", 25, "f2", -50),
        ExpressionPoint("c3", 30, "f2", -2),
    ]
    denoised_points = []

    test_graph = IPPMGraph(dict(), denoised_points)

    ratio, numer, denom = transform_recall(test_graph, noisy_points)

    assert ratio == 0
    assert numer == 0


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

    test_graph = IPPMGraph({"f1": [], "f2": ["f1"]}, denoised_points)

    ratio, numer, denom = transform_recall(test_graph, noisy_points)

    assert ratio == 1
    assert numer == 2
    assert denom == 2


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

    test_graph = IPPMGraph({"f1": []}, denoised_points)

    ratio, numer, denom = transform_recall(test_graph, noisy_points)

    assert ratio == 1 / 2
    assert numer == 1
    assert denom == 2
