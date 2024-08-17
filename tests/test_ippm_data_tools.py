"""
Tests for kymata.ippm.data_tools
"""

from kymata.entities.constants import HEMI_RIGHT, HEMI_LEFT
from kymata.ippm.data_tools import (
    IPPMSpike, build_spike_dict_from_api_response,
    convert_to_power10, copy_hemisphere, remove_excess_funcs,
)


def test_hexel():
    hexel = IPPMSpike("test", "test description", "test commit")
    test_right_pairings = [(20, 10e-3), (50, 0.000012), (611, 0.00053)]
    test_left_pairings = [(122, 0.32), (523, 0.00578), (200, 0.0006)]
    for left, right in zip(test_right_pairings, test_left_pairings):
        hexel.add_pairing(HEMI_RIGHT, left)
        hexel.add_pairing(HEMI_LEFT, right)

    assert hexel.function == "test"
    assert hexel.description == "test description"
    assert hexel.github_commit == "test commit"
    assert hexel.left_best_pairings == [(122, 0.32), (523, 0.00578), (200, 0.0006)]
    assert hexel.right_best_pairings == [(20, 10e-3), (50, 0.000012), (611, 0.00053)]


def test_build_hexel_dict():
    test_dict = {
        HEMI_LEFT: [[2, 1, 0.012, "left1"], [2, 14, 0.213, "left1"]],
        HEMI_RIGHT: [[3, 51, 0.1244, "left1"], [4, 345, 0.557, "right1"]],
    }

    hexels = build_spike_dict_from_api_response(test_dict)

    # check functions are saved correctly
    assert list(hexels.keys()) == ["left1", "right1"]
    # check p value is stored and calculated correctly
    assert hexels["left1"].left_best_pairings == [
        (1, pow(10, 0.012)),
        (14, pow(10, 0.213)),
    ]
    assert hexels["left1"].right_best_pairings == [(51, pow(10, 0.1244))]


def test_Should_convertToPower10_When_validInput():
    hexels = {"f1": IPPMSpike("f1")}
    hexels["f1"].right_best_pairings = [(10, -50), (20, -10), (30, -20), (40, -3)]
    converted = convert_to_power10(hexels)
    assert converted["f1"].right_best_pairings == [
        (10, 1e-50),
        (20, 1e-10),
        (30, 1e-20),
        (40, 1e-3),
    ]


def test_Should_removeExcessFuncs_When_validInput():
    hexels = {"f1": IPPMSpike("f1"), "f2": IPPMSpike("f2"), "f3": IPPMSpike("f3")}
    to_retain = ["f2"]
    filtered = remove_excess_funcs(to_retain, hexels)
    assert list(filtered.keys()) == to_retain


def test_Should_copyHemisphere_When_validInput():
    hexels = {"f1": IPPMSpike("f1")}
    hexels["f1"].right_best_pairings = [(20, 1e-20), (23, 1e-32), (35, 1e-44)]
    hexels["f1"].left_best_pairings = [(10, 1e-20), (21, 1e-55)]
    copy_hemisphere(
        spikes_to=hexels,
        spikes_from=hexels,
        hemi_to=HEMI_RIGHT,
        hemi_from=HEMI_LEFT,
        func="f1",
    )
    assert hexels["f1"].right_best_pairings == hexels["f1"].left_best_pairings
