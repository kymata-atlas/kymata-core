"""
Tests for kymata.ippm.data_tools
"""

from kymata.ippm.data_tools import IPPMSpike, ExpressionPairing


def test_spike():
    spike_left = IPPMSpike("test")
    spike_right = IPPMSpike("test")
    test_right_pairings = [
        ExpressionPairing(20, -2),
        ExpressionPairing(50, -5),
        ExpressionPairing(611, -3.2)]
    test_left_pairings = [
        ExpressionPairing(122, -0.5),
        ExpressionPairing(523, -2.2),
        ExpressionPairing(200, -3.2)]
    for left, right in zip(test_right_pairings, test_left_pairings):
        spike_right.add_pairing(left)
        spike_left.add_pairing(right)

    assert spike_left.transform == "test"
    assert spike_right.transform == "test"
    assert spike_left.best_pairings == [ExpressionPairing(122, -0.5),
                                        ExpressionPairing(523, -2.2),
                                        ExpressionPairing(200, -3.2)]
    assert spike_right.best_pairings == [ExpressionPairing(20, -2),
                                         ExpressionPairing(50, -5),
                                         ExpressionPairing(611, -3.2)]
