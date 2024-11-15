"""
Tests for kymata.ippm.data_tools
"""

from kymata.ippm.data_tools import IPPMSpike, ExpressionPairing


def test_spike():
    spike_left = IPPMSpike("test")
    spike_right = IPPMSpike("test")
    test_right_pairings = [
        ExpressionPairing(20, 10e-3),
        ExpressionPairing(50, 0.000012),
        ExpressionPairing(611, 0.00053)]
    test_left_pairings = [
        ExpressionPairing(122, 0.32),
        ExpressionPairing(523, 0.00578),
        ExpressionPairing(200, 0.0006)]
    for left, right in zip(test_right_pairings, test_left_pairings):
        spike_right.add_pairing(left)
        spike_left.add_pairing(right)

    assert spike_left.transform == "test"
    assert spike_right.transform == "test"
    assert spike_left.best_pairings == [(122, 0.32), (523, 0.00578), (200, 0.0006)]
    assert spike_right.best_pairings == [(20, 10e-3), (50, 0.000012), (611, 0.00053)]
