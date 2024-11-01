from copy import copy

import pytest
import numpy as np

from kymata.entities.expression import (
    SensorExpressionSet,
    HexelExpressionSet,
    DIM_FUNCTION,
    DIM_LATENCY,
    combine, COL_LOGP_VALUE, DIM_SENSOR,
)
from kymata.math.p_values import p_to_logp, logp_to_p


_data_dtype = np.float32


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


@pytest.fixture
def hexel_expression_set_5_hexels() -> HexelExpressionSet:
    return HexelExpressionSet(
        functions="function",
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=np.random.randn(5, 10).astype(_data_dtype),
        data_rh=np.random.randn(5, 10).astype(_data_dtype),
    )


@pytest.fixture
def sensor_expression_set_4_sensors_3_latencies() -> SensorExpressionSet:
    from numpy import array
    from numpy.typing import NDArray

    sensors = [str(i) for i in range(4)]
    function_a_data: NDArray = array(
        p_to_logp(
            array(
                [
                    # 0   1   2  latencies
                    [1, 0.1, 1],  # 0
                    [1, 1, 0.2],  # 1
                    [0.1, 1, 1],  # 2
                    [0.2, 1, 1],  # 3 sensors
                ]
            )
        )
    )
    function_b_data: NDArray = array(
        p_to_logp(
            array(
                [
                    [1, 1, 0.2],
                    [1, 0.1, 1],
                    [1, 0.2, 1],
                    [1, 1, 0.1],
                ]
            )
        )
    )
    return SensorExpressionSet(
        functions=["a", "b"],
        sensors=sensors,  # 4
        latencies=range(3),
        data=[function_a_data, function_b_data],
    )


@pytest.fixture
def sensor_expression_set_4_sensors_4_latencies() -> SensorExpressionSet:
    from numpy import array
    from numpy.typing import NDArray

    sensors = [str(i) for i in range(4)]
    function_a_data: NDArray = array(
        p_to_logp(
            array(
                [
                    # 0   1   2   3  latencies
                    [1, 0.1, 1, 1],  # 0
                    [1, 1, 0.2, 1],  # 1
                    [0.1, 1, 1, 1],  # 2
                    [0.2, 1, 1, 1],  # 3 sensors
                ]
            )
        )
    )
    function_b_data: NDArray = array(
        p_to_logp(
            array(
                [
                    [1, 1, 0.2, 1],
                    [1, 0.1, 1, 1],
                    [1, 0.2, 1, 1],
                    [1, 1, 0.1, 1],
                ]
            )
        )
    )
    return SensorExpressionSet(
        functions=["a", "b"],
        sensors=sensors,  # 4
        latencies=range(4),
        data=[function_a_data, function_b_data],
    )


@pytest.fixture
def sensor_expression_set_4_sensors_4_different_latencies() -> SensorExpressionSet:
    from numpy import array
    from numpy.typing import NDArray

    sensors = [str(i) for i in range(4)]
    function_a_data: NDArray = array(
        p_to_logp(
            array(
                [
                    # 0   1   2   3  latencies
                    [1, 0.1, 1, 1],  # 0
                    [1, 1, 0.2, 1],  # 1
                    [0.1, 1, 1, 1],  # 2
                    [0.2, 1, 1, 1],  # 3 sensors
                ]
            )
        )
    )
    function_b_data: NDArray = array(
        p_to_logp(
            array(
                [
                    [1, 1, 0.2, 1],
                    [1, 0.1, 1, 1],
                    [1, 0.2, 1, 1],
                    [1, 1, 0.1, 1],
                ]
            )
        )
    )
    return SensorExpressionSet(
        functions=["a", "b"],
        sensors=sensors,  # 4
        latencies=range(1, 5),
        data=[function_a_data, function_b_data],
    )


@pytest.fixture
def sensor_expression_set_5_sensors() -> SensorExpressionSet:
    from numpy import array
    from numpy.typing import NDArray

    sensors = [str(i) for i in range(5)]
    function_a_data: NDArray = array(
        p_to_logp(
            array(
                [
                    # 0   1   2  latencies
                    [1, 0.1, 1],  # 0
                    [1, 1, 0.2],  # 1
                    [0.1, 1, 1],  # 2
                    [0.2, 1, 1],  # 3
                    [1, 0.1, 1],  # 4 sensors
                ]
            )
        )
    )
    function_b_data: NDArray = array(
        p_to_logp(
            array(
                [
                    [1, 1, 0.2],
                    [1, 0.1, 1],
                    [1, 0.2, 1],
                    [1, 1, 0.1],
                    [1, 1, 0.1],
                ]
            )
        )
    )
    return SensorExpressionSet(
        functions=["a", "b"],
        sensors=sensors,  # 5
        latencies=range(3),
        data=[function_a_data, function_b_data],
    )


def test_copy_ses(sensor_expression_set_4_sensors_3_latencies):
    c = copy(sensor_expression_set_4_sensors_3_latencies)
    assert sensor_expression_set_4_sensors_3_latencies == c


def test_copy_hes(hexel_expression_set_5_hexels):
    c = copy(hexel_expression_set_5_hexels)
    assert hexel_expression_set_5_hexels == c


def test_hes_hexels_left_equals_right(hexel_expression_set_5_hexels):
    from numpy import array, array_equal

    assert array_equal(hexel_expression_set_5_hexels.hexels_left, array(range(5)))
    assert array_equal(hexel_expression_set_5_hexels.hexels_right, array(range(5)))


def test_ses_best_function():
    from numpy import array
    from numpy.typing import NDArray
    from pandas import DataFrame

    sensors = [str(i) for i in range(4)]
    function_a_data: NDArray = array(
        p_to_logp(
            array(
                [
                    # 0   1   2  latencies
                    [1, 0.1, 1],  # 0
                    [1, 1, 0.2],  # 1
                    [0.1, 1, 1],  # 2
                    [0.2, 1, 1],  # 3 sensors
                ]
            )
        )
    )
    function_b_data: NDArray = array(
        p_to_logp(
            array(
                [
                    [1, 1, 0.2],
                    [1, 0.1, 1],
                    [1, 0.2, 1],
                    [1, 1, 0.1],
                ]
            )
        )
    )
    es = SensorExpressionSet(
        functions=["a", "b"],
        sensors=sensors,  # 4
        latencies=range(3),
        data=[function_a_data, function_b_data],
    )
    best_function_df: DataFrame = es.best_functions()
    correct: DataFrame = DataFrame.from_dict(
        {
            DIM_SENSOR: ["0", "1", "2", "3"],
            DIM_FUNCTION: ["a", "b", "a", "b"],
            DIM_LATENCY: [1, 1, 0, 2],
            COL_LOGP_VALUE: p_to_logp([0.1, 0.1, 0.1, 0.1]),
        }
    )
    assert DataFrame(best_function_df == correct).values.all()


def test_ses_best_function_with_one_channel_all_1s():
    from numpy import array
    from numpy.typing import NDArray
    from pandas import DataFrame

    sensors = [str(i) for i in range(4)]
    function_a_data: NDArray = array(
        p_to_logp(
            array(
                [
                    #  0    1    2  latencies
                    [1, 1, 1],  # 0  <-- set sensor 0 to 1 for some reason
                    [1, 1, 0.2],  # 1
                    [0.1, 1, 1],  # 2
                    [0.2, 1, 1],  # 3 sensors
                ]
            )
        )
    )
    function_b_data: NDArray = array(
        p_to_logp(
            array(
                [
                    [1, 1, 1],
                    [1, 0.1, 1],
                    [1, 0.2, 1],
                    [1, 1, 0.1],
                ]
            )
        )
    )
    es = SensorExpressionSet(
        functions=["a", "b"],
        sensors=sensors,  # 4
        latencies=range(3),
        data=[function_a_data, function_b_data],
    )
    best_function_df: DataFrame = es.best_functions()
    correct: DataFrame = DataFrame.from_dict(
        {
            DIM_SENSOR: ["1", "2", "3"],
            DIM_FUNCTION: ["b", "a", "b"],
            DIM_LATENCY: [1, 0, 2],
            COL_LOGP_VALUE: p_to_logp([0.1, 0.1, 0.1]),
        }
    )
    assert DataFrame(best_function_df == correct).values.all()


def test_ses_best_function_with_one_channel_all_nans():
    from numpy import array, nan
    from numpy.typing import NDArray
    from pandas import DataFrame

    sensors = [str(i) for i in range(4)]
    function_a_data: NDArray = array(
        p_to_logp(
            array(
                [
                    #  0    1    2  latencies
                    [nan, nan, nan],  # 0  <-- set sensor 0 to nans for some reason
                    [1, 1, 0.2],  # 1
                    [0.1, 1, 1],  # 2
                    [0.2, 1, 1],  # 3 sensors
                ]
            )
        )
    )
    function_b_data: NDArray = array(
        p_to_logp(
            array(
                [
                    [nan, nan, nan],
                    [1, 0.1, 1],
                    [1, 0.2, 1],
                    [1, 1, 0.1],
                ]
            )
        )
    )
    es = SensorExpressionSet(
        functions=["a", "b"],
        sensors=sensors,  # 4
        latencies=range(3),
        data=[function_a_data, function_b_data],
    )
    best_function_df: DataFrame = es.best_functions()
    correct: DataFrame = DataFrame.from_dict(
        {
            DIM_SENSOR: ["1", "2", "3"],
            DIM_FUNCTION: ["b", "a", "b"],
            DIM_LATENCY: [1, 0, 2],
            COL_LOGP_VALUE: p_to_logp([0.1, 0.1, 0.1]),
        }
    )
    assert DataFrame(best_function_df == correct).values.all()


# Test ExpressionSet arg validations


def test_ses_validation_input_lengths_two_functions_one_dataset():
    with pytest.raises(ValueError):
        SensorExpressionSet(
            functions=["first", "second"],
            sensors=list("abcde"),
            latencies=range(10),
            data=np.random.randn(5, 10).astype(_data_dtype),
        )


def test_ses_validation_input_lengths_two_functions_two_datasets_sequence():
    SensorExpressionSet(
        functions=["first", "second"],
        sensors=list("abcde"),
        latencies=range(10),
        data=[np.random.randn(5, 10) for _ in range(2)],
    )


def test_ses_validation_input_lengths_two_functions_two_datasets_contiguous():
    SensorExpressionSet(
        functions=["first", "second"],
        sensors=list("abcde"),
        latencies=range(10),
        data=np.random.randn(5, 10, 2).astype(_data_dtype),
    )


def test_ses_validation_input_lengths_one_function_two_datasets_contiguous():
    with pytest.raises(ValueError):
        SensorExpressionSet(
            functions=["first"],
            sensors=list("abcde"),
            latencies=range(10),
            data=np.random.randn(5, 10, 2).astype(_data_dtype),
        )


def test_ses_validation_input_lengths_two_functions_three_datasets_sequence():
    with pytest.raises(ValueError):
        SensorExpressionSet(
            functions=["first", "second"],
            sensors=list("abcde"),
            latencies=range(10),
            data=[np.random.randn(5, 10) for _ in range(3)],
        )


def test_ses_validation_input_lengths_two_functions_three_datasets_contiguous():
    with pytest.raises(ValueError):
        SensorExpressionSet(
            functions=["first", "second"],
            sensors=list("abcde"),
            latencies=range(10),
            data=np.random.randn(5, 10, 3).astype(_data_dtype),
        )


def test_hes_validation_input_lengths_two_functions_one_dataset():
    with pytest.raises(ValueError):
        HexelExpressionSet(
            functions=["first", "second"],
            hexels_lh=range(5),
            hexels_rh=range(5),
            latencies=range(10),
            data_lh=np.random.randn(5, 10).astype(_data_dtype),
            data_rh=np.random.randn(5, 10).astype(_data_dtype),
        )


def test_hes_validation_input_lengths_two_functions_two_datasets_sequence():
    HexelExpressionSet(
        functions=["first", "second"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=[np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)],
        data_rh=[np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)],
    )


def test_hes_validation_input_lengths_two_functions_two_datasets_contiguous():
    HexelExpressionSet(
        functions=["first", "second"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=np.random.randn(5, 10, 2).astype(_data_dtype),
        data_rh=np.random.randn(5, 10, 2).astype(_data_dtype),
    )


def test_hes_validation_input_lengths_two_functions_three_datasets_sequence():
    with pytest.raises(ValueError):
        HexelExpressionSet(
            functions=["first", "second"],
            hexels_lh=range(5),
            hexels_rh=range(5),
            latencies=range(10),
            data_lh=[np.random.randn(5, 10).astype(_data_dtype) for _ in range(3)],
            data_rh=[np.random.randn(5, 10).astype(_data_dtype) for _ in range(3)],
        )


def test_hes_validation_input_lengths_two_functions_three_datasets_contiguous():
    with pytest.raises(ValueError):
        HexelExpressionSet(
            functions=["first", "second"],
            hexels_lh=range(5),
            hexels_rh=range(5),
            latencies=range(10),
            data_lh=np.random.randn(5, 10, 3).astype(_data_dtype),
            data_rh=np.random.randn(5, 10, 3).astype(_data_dtype),
        )


def test_ses_validation_duplicated_functions():
    with pytest.raises(ValueError):
        SensorExpressionSet(
            functions=["dupe", "dupe"],
            sensors=list("abcde"),
            latencies=range(10),
            data=[np.random.randn(5, 10) for _ in range(2)],
        )


def test_hes_validation_input_mismatched_blocks_concordent_channels():
    HexelExpressionSet(
        functions="function",
        hexels_lh=range(5),
        hexels_rh=range(6),
        latencies=range(10),
        data_lh=np.random.randn(5, 10).astype(_data_dtype),
        data_rh=np.random.randn(6, 10).astype(_data_dtype),
    )


def test_hes_validation_input_mismatched_blocks_concordent_channels_two_functions_sequence():
    HexelExpressionSet(
        functions=["first", "second"],
        hexels_lh=range(5),
        hexels_rh=range(6),
        latencies=range(10),
        data_lh=[np.random.randn(5, 10).astype(_data_dtype),
                 np.random.randn(5, 10).astype(_data_dtype)],
        data_rh=[np.random.randn(6, 10).astype(_data_dtype),
                 np.random.randn(6, 10).astype(_data_dtype)],
    )


def test_hes_validation_input_mismatched_blocks_concordent_channels_two_functions_contiguous():
    HexelExpressionSet(
        functions=["first", "second"],
        hexels_lh=range(5),
        hexels_rh=range(6),
        latencies=range(10),
        data_lh=np.random.randn(5, 10, 2).astype(_data_dtype),
        data_rh=np.random.randn(6, 10, 2).astype(_data_dtype),
    )


def test_hes_validation_input_mismatched_blocks_discordent_channels():
    with pytest.raises(ValueError):
        HexelExpressionSet(
            functions="function",
            hexels_lh=range(5),
            hexels_rh=range(5),
            latencies=range(10),
            data_lh=np.random.randn(5, 10).astype(_data_dtype),
            data_rh=np.random.randn(6, 10).astype(_data_dtype),
        )


def test_hes_validation_mixmatched_latencies_between_functions():
    with pytest.raises(ValueError):
        HexelExpressionSet(
            functions=["first", "second"],
            hexels_lh=range(5),
            hexels_rh=range(6),
            latencies=range(10),
            data_lh=[np.random.randn(5, 10).astype(_data_dtype), np.random.randn(5, 11)],
            data_rh=[np.random.randn(6, 10).astype(_data_dtype), np.random.randn(6, 11)],
        )


def test_hes_validation_mixmatched_hexels_between_functions():
    with pytest.raises(ValueError):
        HexelExpressionSet(
            functions=["first", "second"],
            hexels_lh=range(5),
            hexels_rh=range(6),
            latencies=range(10),
            data_lh=[np.random.randn(5, 10).astype(_data_dtype), np.random.randn(4, 10)],
            data_rh=[np.random.randn(6, 10).astype(_data_dtype), np.random.randn(6, 10)],
        )


def test_hes_rename_functions():
    data_left = [np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)]
    data_right = [np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)]

    es = HexelExpressionSet(
        functions=["first", "second"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=data_left,
        data_rh=data_right,
    )
    target_es = HexelExpressionSet(
        functions=["first_renamed", "second_renamed"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=data_left,
        data_rh=data_right,
    )
    assert es != target_es
    es.rename(functions={"first": "first_renamed", "second": "second_renamed"})
    assert es == target_es


def test_hes_rename_functions_noop():
    data_left = [np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)]
    data_right = [np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)]

    es = HexelExpressionSet(
        functions=["first", "second"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=data_left,
        data_rh=data_right,
    )
    target_es = copy(es)

    assert es == target_es
    es.rename()
    assert es == target_es


def test_hes_rename_functions_just_one():
    data_left = [np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)]
    data_right = [np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)]

    es = HexelExpressionSet(
        functions=["first", "second"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=data_left,
        data_rh=data_right,
    )
    target_es = HexelExpressionSet(
        functions=["first_renamed", "second"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=data_left,
        data_rh=data_right,
    )
    assert es != target_es
    es.rename(functions={"first": "first_renamed"})
    assert es == target_es


def test_hes_rename_functions_wrong_name():
    data_left = [np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)]
    data_right = [np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)]

    es = HexelExpressionSet(
        functions=["first", "second"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=data_left,
        data_rh=data_right,
    )
    with pytest.raises(KeyError):
        es.rename(functions={"first": "first_renamed", "missing": "second_renamed"})


def test_hes_rename_hexels():
    data_left = [np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)]
    data_right = [np.random.randn(5, 10).astype(_data_dtype) for _ in range(2)]

    es = HexelExpressionSet(
        functions=["first", "second"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=data_left,
        data_rh=data_right,
    )
    target_es = HexelExpressionSet(
        functions=["first", "second"],
        hexels_lh=range(1, 6),
        hexels_rh=range(1, 6),
        latencies=range(10),
        data_lh=data_left,
        data_rh=data_right,
    )
    assert es != target_es
    es.rename(channels={c: c + 1 for c in range(5)})
    assert es == target_es


def test_combine_vaild_ses_works(sensor_expression_set_4_sensors_3_latencies):
    ses_1 = copy(sensor_expression_set_4_sensors_3_latencies)
    ses_2 = copy(sensor_expression_set_4_sensors_3_latencies)
    ses_2.rename({f: f"{f}+++" for f in ses_2.functions})
    combined = combine([ses_1, ses_2])
    assert np.array_equal(combined.sensors, ses_1.sensors)
    assert np.array_equal(combined.sensors, ses_2.sensors)
    assert np.array_equal(combined.latencies, ses_1.latencies)
    assert np.array_equal(combined.latencies, ses_2.latencies)
    assert set(ses_1.functions) | set(ses_2.functions) == set(combined.functions)


def test_combine_fails_with_mixed_types(
    hexel_expression_set_5_hexels, sensor_expression_set_4_sensors_3_latencies
):
    with pytest.raises(ValueError):
        combine(
            [hexel_expression_set_5_hexels, sensor_expression_set_4_sensors_3_latencies]
        )


def test_combine_fails_with_mismatched_sensor_counts(
    sensor_expression_set_4_sensors_3_latencies, sensor_expression_set_5_sensors
):
    with pytest.raises(ValueError):
        combine(
            [
                sensor_expression_set_4_sensors_3_latencies,
                sensor_expression_set_5_sensors,
            ]
        )


def test_combine_fails_with_mismatched_sensor_names(
    sensor_expression_set_4_sensors_3_latencies,
):
    ses_renamed_sensors: SensorExpressionSet = copy(
        sensor_expression_set_4_sensors_3_latencies
    )
    ses_renamed_sensors.rename(
        channels={
            c: f"{c}'" for c in sensor_expression_set_4_sensors_3_latencies.sensors
        }
    )
    with pytest.raises(ValueError):
        combine([sensor_expression_set_4_sensors_3_latencies, ses_renamed_sensors])


def test_combine_fails_with_mismatched_latency_counts(
    sensor_expression_set_4_sensors_3_latencies,
    sensor_expression_set_4_sensors_4_latencies,
):
    with pytest.raises(ValueError):
        combine(
            [
                sensor_expression_set_4_sensors_3_latencies,
                sensor_expression_set_4_sensors_4_latencies,
            ]
        )


def test_combine_fails_with_mismatched_latencies(
    sensor_expression_set_4_sensors_4_latencies,
    sensor_expression_set_4_sensors_4_different_latencies,
):
    with pytest.raises(ValueError):
        combine(
            [
                sensor_expression_set_4_sensors_4_latencies,
                sensor_expression_set_4_sensors_4_different_latencies,
            ]
        )
