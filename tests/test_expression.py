import numpy as np
from kymata.entities.expression import p_to_logp, logp_to_p


def test_log_p_single_value():
    p = 0.000_000_000_001
    assert p_to_logp(p) == -12


def test_log_p_single_array():
    ps = np.array([1, 0.1, 0.01, 0.001, 0.0001])
    assert np.array_equal(p_to_logp(ps), np.array([0, -1, -2, -3, -4]))


def test_unlog_p_single_value():
    logp = -10
    assert logp_to_p(logp) == 0.000_000_000_1


def test_unlog_p_array():
    logps = np.array([-4, -3, -2, -1, 0])
    assert np.array_equal(logp_to_p(logps), np.array([0.000_1, 0.001, 0.01, 0.1, 1]))
