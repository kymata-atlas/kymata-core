"""
Tests for kymata.ippm.data_tools
"""

from kymata.entities.constants import HEMI_RIGHT, HEMI_LEFT
from kymata.ippm.data_tools import IPPMSpike, copy_hemisphere


def test_hexel():
    hexel = IPPMSpike("test")
    test_right_pairings = [(20, 10e-3), (50, 0.000012), (611, 0.00053)]
    test_left_pairings = [(122, 0.32), (523, 0.00578), (200, 0.0006)]
    for left, right in zip(test_right_pairings, test_left_pairings):
        hexel.add_pairing(HEMI_RIGHT, left)
        hexel.add_pairing(HEMI_LEFT, right)

    assert hexel.transform == "test"
    assert hexel.left_best_pairings == [(122, 0.32), (523, 0.00578), (200, 0.0006)]
    assert hexel.right_best_pairings == [(20, 10e-3), (50, 0.000012), (611, 0.00053)]


def test_Should_copyHemisphere_When_validInput():
    hexels = {"f1": IPPMSpike("f1")}
    hexels["f1"].right_best_pairings = [(20, 1e-20), (23, 1e-32), (35, 1e-44)]
    hexels["f1"].left_best_pairings = [(10, 1e-20), (21, 1e-55)]
    copy_hemisphere(
        spikes_to=hexels,
        spikes_from=hexels,
        hemi_to=HEMI_RIGHT,
        hemi_from=HEMI_LEFT,
        trans="f1",
    )
    assert hexels["f1"].right_best_pairings == hexels["f1"].left_best_pairings
