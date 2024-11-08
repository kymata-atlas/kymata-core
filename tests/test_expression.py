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


def test_ses_best_function_with_one_channel_all_1s():
    from numpy import array
    from numpy.typing import NDArray
    from pandas import DataFrame
    sensors = [str(i) for i in range(4)]
    function_a_data: NDArray = array(p_to_logp(array([
        #  0    1    2  latencies
        [  1,   1,   1],  # 0  <-- set sensor 0 to 1 for some reason
        [  1,   1,  .2],  # 1
        [ .1,   1,   1],  # 2
        [ .2,   1,   1],  # 3 sensors
    ])))
    function_b_data: NDArray = array(p_to_logp(array([
        [ 1,    1,   1],
        [ 1,   .1,   1],
        [ 1,   .2,   1],
        [ 1,    1,  .1],
    ])))
    es = SensorExpressionSet(functions=["a", "b"],
                             sensors=sensors,  # 4
                             latencies=range(3),
                             data=[function_a_data, function_b_data])
    best_function_df: DataFrame = es.best_functions()
    correct: DataFrame = DataFrame.from_dict({
        "sensor":               ["1", "2", "3"],
        DIM_FUNCTION:           ["b", "a", "b"],
        DIM_LATENCY:            [  1,   0,   2 ],
        "value":      p_to_logp([ .1,  .1,  .1 ]),
    })
    assert DataFrame(best_function_df == correct).values.all()


def test_ses_best_function_with_one_channel_all_nans():
    from numpy import array, nan
    from numpy.typing import NDArray
    from pandas import DataFrame
    sensors = [str(i) for i in range(4)]
    function_a_data: NDArray = array(p_to_logp(array([
        #  0    1    2  latencies
        [nan, nan, nan],  # 0  <-- set sensor 0 to nans for some reason
        [  1,   1,  .2],  # 1
        [ .1,   1,   1],  # 2
        [ .2,   1,   1],  # 3 sensors
    ])))
    function_b_data: NDArray = array(p_to_logp(array([
        [nan, nan, nan],
        [ 1,   .1,   1],
        [ 1,   .2,   1],
        [ 1,    1,  .1],
    ])))
    es = SensorExpressionSet(functions=["a", "b"],
                             sensors=sensors,  # 4
                             latencies=range(3),
                             data=[function_a_data, function_b_data])
    best_function_df: DataFrame = es.best_functions()
    correct: DataFrame = DataFrame.from_dict({
        "sensor":               ["1", "2", "3"],
        DIM_FUNCTION:           ["b", "a", "b"],
        DIM_LATENCY:            [  1,   0,   2 ],
        "value":      p_to_logp([ .1,  .1,  .1 ]),
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
    with pytest.raises(ValueError):
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


def test_hes_rename_functions():
    data_left = [np.random.randn(5, 10) for _ in range(2)]
    data_right = [np.random.randn(5, 10) for _ in range(2)]

    es = HexelExpressionSet(functions=["first", "second"],
                            hexels_lh=range(5),
                            hexels_rh=range(5),
                            latencies=range(10),
                            data_lh=data_left,
                            data_rh=data_right,
                            )
    target_es = HexelExpressionSet(functions=["first_renamed", "second_renamed"],
                                   hexels_lh=range(5),
                                   hexels_rh=range(5),
                                   latencies=range(10),
                                   data_lh=data_left,
                                   data_rh=data_right,
                                   )
    assert es != target_es
    es.rename(functions={"first": "first_renamed", "second": "second_renamed"})
    assert es == target_es


def test_hes_rename_functions_just_one():
    data_left = [np.random.randn(5, 10) for _ in range(2)]
    data_right = [np.random.randn(5, 10) for _ in range(2)]

    es = HexelExpressionSet(functions=["first", "second"],
                            hexels_lh=range(5),
                            hexels_rh=range(5),
                            latencies=range(10),
                            data_lh=data_left,
                            data_rh=data_right,
                            )
    target_es = HexelExpressionSet(functions=["first_renamed", "second"],
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
    data_left = [np.random.randn(5, 10) for _ in range(2)]
    data_right = [np.random.randn(5, 10) for _ in range(2)]

    es = HexelExpressionSet(functions=["first", "second"],
                            hexels_lh=range(5),
                            hexels_rh=range(5),
                            latencies=range(10),
                            data_lh=data_left,
                            data_rh=data_right,
                            )
    with pytest.raises(KeyError):
        es.rename(transforms={"first": "first_renamed", "missing": "second_renamed"})


def test_hes_rename_hexels():
    data_left = [np.random.randn(5, 10) for _ in range(2)]
    data_right = [np.random.randn(5, 10) for _ in range(2)]

    es = HexelExpressionSet(
        transforms=["first", "second"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=data_left,
        data_rh=data_right,
    )
    target_es = HexelExpressionSet(
        transforms=["first", "second"],
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
    ses_2.rename({f: f"{f}+++" for f in ses_2.transforms})
    combined = combine([ses_1, ses_2])
    assert np.array_equal(combined.sensors, ses_1.sensors)
    assert np.array_equal(combined.sensors, ses_2.sensors)
    assert np.array_equal(combined.latencies, ses_1.latencies)
    assert np.array_equal(combined.latencies, ses_2.latencies)
    assert set(ses_1.transforms) | set(ses_2.transforms) == set(combined.transforms)


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


def test_subset_transforms_one():
    data_left  = [np.random.randn(5, 10) for _ in range(3)]
    data_right = [np.random.randn(5, 10) for _ in range(3)]

    es = HexelExpressionSet(
        transforms=["first", "second", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=data_left,
        data_rh=data_right,
    )

    first = HexelExpressionSet(
        transforms=["first"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=[data_left[0]],
        data_rh=[data_right[0]],
    )

    assert es["first"] == first


def test_subset_transforms_two():
    data_left  = [np.random.randn(5, 10) for _ in range(3)]
    data_right = [np.random.randn(5, 10) for _ in range(3)]

    es = HexelExpressionSet(
        transforms=["first", "second", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=data_left,
        data_rh=data_right,
    )

    first_two = HexelExpressionSet(
        transforms=["first", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=range(10),
        data_lh=[data_left[0], data_left[2]],
        data_rh=[data_right[0], data_right[2]],
    )

    assert es["first", "third"] == first_two


def test_time_crop_both_endpoints_inside():
    #              0    1    2    3    4  5   6   7   8   9  10
    latencies = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]
    data_left  = [np.random.randn(5, len(latencies)) for _ in range(3)]
    data_right = [np.random.randn(5, len(latencies)) for _ in range(3)]

    es = HexelExpressionSet(
        transforms=["first", "second", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=latencies,
        data_lh=data_left,
        data_rh=data_right,
    )

    cropped = HexelExpressionSet(
        transforms=["first", "second", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=[-.2, -.1, 0, .1, .2, .3],
        data_lh=[d[:, 3:9] for d in data_left],
        data_rh=[d[:, 3:9] for d in data_right],
    )

    # Inclusive of endpoints
    assert es.crop_time(-.2, .3) == cropped

    # Endpoints between timepoints
    assert es.crop_time(-.25, .35) == cropped


def test_time_crop_half_open_left():
    #              0    1    2    3    4  5   6   7   8   9  10
    latencies = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]
    data_left  = [np.random.randn(5, len(latencies)) for _ in range(3)]
    data_right = [np.random.randn(5, len(latencies)) for _ in range(3)]

    es = HexelExpressionSet(
        transforms=["first", "second", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=latencies,
        data_lh=data_left,
        data_rh=data_right,
    )

    cropped = HexelExpressionSet(
        transforms=["first", "second", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=[-.5, -.4, -.3, -.2, -.1],
        data_lh=[d[:, :5] for d in data_left],
        data_rh=[d[:, :5] for d in data_right],
    )

    # Inclusive of endpoints
    assert es.crop_time(None, -.1) == cropped

    # Endpoints between timepoints
    assert es.crop_time(None, -.05) == cropped


def test_time_crop_half_open_right():
    #              0    1    2    3    4  5   6   7   8   9  10
    latencies = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]
    data_left  = [np.random.randn(5, len(latencies)) for _ in range(3)]
    data_right = [np.random.randn(5, len(latencies)) for _ in range(3)]

    es = HexelExpressionSet(
        transforms=["first", "second", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=latencies,
        data_lh=data_left,
        data_rh=data_right,
    )

    cropped = HexelExpressionSet(
        transforms=["first", "second", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=[-.1, 0, .1, .2, .3, .4, .5],
        data_lh=[d[:, 4:] for d in data_left],
        data_rh=[d[:, 4:] for d in data_right],
    )

    # Inclusive of endpoints
    assert es.crop_time(-.1, None) == cropped

    # Endpoints between timepoints
    assert es.crop_time(-.15, None) == cropped


def test_time_crop_contains_whole():
    #              0    1    2    3    4  5   6   7   8   9  10
    latencies = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]
    data_left  = [np.random.randn(5, len(latencies)) for _ in range(3)]
    data_right = [np.random.randn(5, len(latencies)) for _ in range(3)]

    es = HexelExpressionSet(
        transforms=["first", "second", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=latencies,
        data_lh=data_left,
        data_rh=data_right,
    )

    assert es.crop_time(-1, 1) == es


def test_time_crop_one_datapoint_at_end():
    #              0    1    2    3    4  5   6   7   8   9  10
    latencies = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]
    data  = [np.random.randn(5, len(latencies)) for _ in range(3)]

    es = SensorExpressionSet(
        transforms=["first", "second", "third"],
        sensors=range(5),
        latencies=latencies,
        data=data,
    )

    cropped = SensorExpressionSet(
        transforms=["first", "second", "third"],
        sensors=range(5),
        latencies=[.5],
        data=[d[:, -1:] for d in data],
    )

    assert es.crop_time(.401, 1) == cropped


def test_time_crop_no_datapoints_between():
    #              0    1    2    3    4  5   6   7   8   9  10
    latencies = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]
    data_left  = [np.random.randn(5, len(latencies)) for _ in range(3)]
    data_right = [np.random.randn(5, len(latencies)) for _ in range(3)]

    es = HexelExpressionSet(
        transforms=["first", "second", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=latencies,
        data_lh=data_left,
        data_rh=data_right,
    )

    with pytest.raises(IndexError):
        es.crop_time(0.01, 0.02)


def test_time_crop_outside_range():
    #              0    1    2    3    4  5   6   7   8   9  10
    latencies = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]
    data_left  = [np.random.randn(5, len(latencies)) for _ in range(3)]
    data_right = [np.random.randn(5, len(latencies)) for _ in range(3)]

    es = HexelExpressionSet(
        transforms=["first", "second", "third"],
        hexels_lh=range(5),
        hexels_rh=range(5),
        latencies=latencies,
        data_lh=data_left,
        data_rh=data_right,
    )

    with pytest.raises(IndexError):
        es.crop_time(-1, -.8)

    with pytest.raises(IndexError):
        es.crop_time(1, 2)
