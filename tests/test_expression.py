import pytest
import numpy as np

from kymata.entities.expression import p_to_logp, logp_to_p, SensorExpressionSet, HexelExpressionSet


def test_log_p_single_value():
    p = 0.000_000_000_001
    assert np.isclose(p_to_logp(p), -12)


def test_log_p_single_array():
    ps = np.array([1, 0.1, 0.01, 0.001, 0.0001])
    assert np.isclose(p_to_logp(ps), np.array([0, -1, -2, -3, -4])).all()


def test_unlog_p_single_value():
    logp = -10
    assert np.isclose(logp_to_p(logp), 0.000_000_000_1)


def test_unlog_p_array():
    logps = np.array([-4, -3, -2, -1, 0])
    assert np.isclose(logp_to_p(logps), np.array([0.000_1, 0.001, 0.01, 0.1, 1])).all()


# Test ExpressionSet arg validations

def test_ses_validation_input_lengths_two_functions_one_dataset():
    with pytest.raises(Exception):
        SensorExpressionSet(functions=["first", "second"],
                            sensors=list("abcde"),
                            latencies=range(10),
                            data=np.random.randn(5, 10),
                            )


def test_ses_validation_input_lengths_two_functions_two_datasets():
    SensorExpressionSet(functions=["first", "second"],
                        sensors=list("abcde"),
                        latencies=range(10),
                        data=[np.random.randn(5, 10) for _ in range(2)],
                        )


def test_ses_validation_input_lengths_two_functions_three_datasets():
    with pytest.raises(Exception):
        SensorExpressionSet(functions=["first", "second"],
                            sensors=list("abcde"),
                            latencies=range(10),
                            data=[np.random.randn(5, 10) for _ in range(3)],
                            )


def test_hes_validation_input_lengths_two_functions_one_dataset():
    with pytest.raises(Exception):
        HexelExpressionSet(functions=["first", "second"],
                           hexels=range(5),
                           latencies=range(10),
                           data_lh=np.random.randn(5, 10),
                           data_rh=np.random.randn(5, 10),
                           )


def test_hes_validation_input_lengths_two_functions_two_datasets():
    HexelExpressionSet(functions=["first", "second"],
                       hexels=range(5),
                       latencies=range(10),
                       data_lh=[np.random.randn(5, 10) for _ in range(2)],
                       data_rh=[np.random.randn(5, 10) for _ in range(2)],
                       )


def test_hes_validation_input_lengths_two_functions_three_datasets():
    with pytest.raises(Exception):
        HexelExpressionSet(functions=["first", "second"],
                           hexels=range(5),
                           latencies=range(10),
                           data_lh=[np.random.randn(5, 10) for _ in range(3)],
                           data_rh=[np.random.randn(5, 10) for _ in range(3)],
                           )


def test_ses_validation_duplicated_functions():
    with pytest.raises(Exception):
        SensorExpressionSet(functions=["dupe", "dupe"],
                            sensors=list("abcde"),
                            latencies=range(10),
                            data=[np.random.randn(5, 10) for _ in range(2)],
                            )


def test_hes_validation_input_mismatched_layers():
    with pytest.raises(Exception):
        HexelExpressionSet(functions="function",
                           hexels=range(5),
                           latencies=range(10),
                           data_lh=np.random.randn(5, 10),
                           data_rh=np.random.randn(6, 10),
                           )


def test_hes_validation_mixmatched_latencies_between_functions():
    with pytest.raises(Exception):
        HexelExpressionSet(functions=["first", "second"],
                           hexels=range(5),
                           latencies=range(10),
                           data_lh=[np.random.randn(5, 10), np.random.randn(5, 11)],
                           data_rh=[np.random.randn(5, 10), np.random.randn(5, 11)],
                           )


def test_hes_validation_mixmatched_hexels_between_functions():
    with pytest.raises(Exception):
        HexelExpressionSet(functions=["first", "second"],
                           hexels=range(5),
                           latencies=range(10),
                           data_lh=[np.random.randn(5, 10), np.random.randn(4, 10)],
                           data_rh=[np.random.randn(5, 10), np.random.randn(4, 10)],
                           )


def test_hes_validation_mixmatched_hexels_between_layers():
    with pytest.raises(Exception):
        HexelExpressionSet(functions=["first", "second"],
                           hexels=range(5),
                           latencies=range(10),
                           data_lh=[np.random.randn(5, 10), np.random.randn(5, 10)],
                           data_rh=[np.random.randn(4, 10), np.random.randn(4, 10)],
                           )
