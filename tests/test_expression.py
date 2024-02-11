import pytest
import numpy as np

from kymata.entities.expression import SensorExpressionSet, HexelExpressionSet, DIM_FUNCTION, DIM_LATENCY
from kymata.math.p_values import p_to_logp, logp_to_p


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


def test_hes_hexels():
    from numpy import array, array_equal
    es = HexelExpressionSet(functions="function",
                            hexels_lh=range(5),
                            hexels_rh=range(5),
                            latencies=range(10),
                            data_lh=np.random.randn(5, 10),
                            data_rh=np.random.randn(5, 10),
                            )
    assert array_equal(es.hexels_left, array(range(5)))
    assert array_equal(es.hexels_right, array(range(5)))


def test_ses_best_function():
    from numpy import array
    from numpy.typing import NDArray
    from pandas import DataFrame
    sensors = [str(i) for i in range(4)]
    function_a_data: NDArray = array(p_to_logp(array([
        # 0   1   2  latencies
        [ 1, .1,  1],  # 0
        [ 1,  1, .2],  # 1
        [.1,  1,  1],  # 2
        [.2,  1,  1],  # 3 sensors
    ])))
    function_b_data: NDArray = array(p_to_logp(array([
        [ 1,  1, .2],
        [ 1, .1,  1],
        [ 1, .2,  1],
        [ 1,  1, .1],
    ])))
    es = SensorExpressionSet(functions=["a", "b"],
                             sensors=sensors,  # 4
                             latencies=range(3),
                             data=[function_a_data, function_b_data])
    best_function_df: DataFrame = es.best_functions()
    correct: DataFrame = DataFrame.from_dict({
        "sensor":               ["0", "1", "2", "3"],
        DIM_FUNCTION:           ["a", "b", "a", "b"],
        DIM_LATENCY:            [ 1,   1,   0,   2 ],
        "value":      p_to_logp([.1,  .1,  .1,  .1 ]),
    })
    assert DataFrame(best_function_df == correct).values.all()


# Test ExpressionSet arg validations

def test_ses_validation_input_lengths_two_functions_one_dataset():
    with pytest.raises(AssertionError):
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
    with pytest.raises(AssertionError):
        SensorExpressionSet(functions=["first", "second"],
                            sensors=list("abcde"),
                            latencies=range(10),
                            data=[np.random.randn(5, 10) for _ in range(3)],
                            )


def test_hes_validation_input_lengths_two_functions_one_dataset():
    with pytest.raises(AssertionError):
        HexelExpressionSet(functions=["first", "second"],
                           hexels_lh=range(5),
                           hexels_rh=range(5),
                           latencies=range(10),
                           data_lh=np.random.randn(5, 10),
                           data_rh=np.random.randn(5, 10),
                           )


def test_hes_validation_input_lengths_two_functions_two_datasets():
    HexelExpressionSet(functions=["first", "second"],
                       hexels_lh=range(5),
                       hexels_rh=range(5),
                       latencies=range(10),
                       data_lh=[np.random.randn(5, 10) for _ in range(2)],
                       data_rh=[np.random.randn(5, 10) for _ in range(2)],
                       )


def test_hes_validation_input_lengths_two_functions_three_datasets():
    with pytest.raises(AssertionError):
        HexelExpressionSet(functions=["first", "second"],
                           hexels_lh=range(5),
                           hexels_rh=range(5),
                           latencies=range(10),
                           data_lh=[np.random.randn(5, 10) for _ in range(3)],
                           data_rh=[np.random.randn(5, 10) for _ in range(3)],
                           )


def test_ses_validation_duplicated_functions():
    with pytest.raises(AssertionError):
        SensorExpressionSet(functions=["dupe", "dupe"],
                            sensors=list("abcde"),
                            latencies=range(10),
                            data=[np.random.randn(5, 10) for _ in range(2)],
                            )


def test_hes_validation_input_mismatched_blocks_concordent_channels():
    HexelExpressionSet(functions="function",
                       hexels_lh=range(5),
                       hexels_rh=range(6),
                       latencies=range(10),
                       data_lh=np.random.randn(5, 10),
                       data_rh=np.random.randn(6, 10),
                       )


def test_hes_validation_input_mismatched_blocks_concordent_channels_two_functions():
    HexelExpressionSet(functions=["first", "second"],
                       hexels_lh=range(5),
                       hexels_rh=range(6),
                       latencies=range(10),
                       data_lh=[np.random.randn(5, 10), np.random.randn(5, 10)],
                       data_rh=[np.random.randn(6, 10), np.random.randn(6, 10)],
                       )


def test_hes_validation_input_mismatched_blocks_discordent_channels():
    with pytest.raises(AssertionError):
        HexelExpressionSet(functions="function",
                           hexels_lh=range(5),
                           hexels_rh=range(5),
                           latencies=range(10),
                           data_lh=np.random.randn(5, 10),
                           data_rh=np.random.randn(6, 10),
                           )


def test_hes_validation_mixmatched_latencies_between_functions():
    with pytest.raises(AssertionError):
        HexelExpressionSet(functions=["first", "second"],
                           hexels_lh=range(5),
                           hexels_rh=range(6),
                           latencies=range(10),
                           data_lh=[np.random.randn(5, 10), np.random.randn(5, 11)],
                           data_rh=[np.random.randn(6, 10), np.random.randn(6, 11)],
                           )


def test_hes_validation_mixmatched_hexels_between_functions():
    with pytest.raises(AssertionError):
        HexelExpressionSet(functions=["first", "second"],
                           hexels_lh=range(5),
                           hexels_rh=range(6),
                           latencies=range(10),
                           data_lh=[np.random.randn(5, 10), np.random.randn(4, 10)],
                           data_rh=[np.random.randn(6, 10), np.random.randn(6, 10)],
                           )
