from copy import deepcopy
from functools import reduce
from math import isclose
from unittest.mock import patch, MagicMock

import pandas as pd
from pandas.testing import assert_frame_equal

import math
from kymata.ippm.data_tools import IPPMHexel
from kymata.ippm.denoising_strategies import DenoisingStrategy

test_data_func1 = [
    [-100, 1e-50],
    [-90, 1e-34],
    [-95, 1e-8],
    [-75, 1e-75],
    [-70, 1e-27],
    [0, 1e-1],
    [30, 1e-100],
    [32, 1e-93],
    [35, 1e-72],
    [50, 1e-9],
    [176, 1e-50],
    [199, 1e-90],
    [200, 1e-50],
    [210, 1e-44],
    [211, 1e-55]
]
significant_test_data_func1 = [
    [-100, 1e-50],
    [-90, 1e-34],
    [-75, 1e-75],
    [-70, 1e-27],
    [30, 1e-100],
    [32, 1e-93],
    [35, 1e-72],
    [176, 1e-50],
    [199, 1e-90],
    [200, 1e-50],
    [210, 1e-44],
    [211, 1e-55]
]
significant_test_data_func1_labels = [0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2]
test_df_func1 = pd.DataFrame(significant_test_data_func1, columns=["Latency", "Mag"])
denoised_func1 = [
    (-75, 1e-75),
    (30, 1e-100),
    (199, 1e-90)
]

test_empty_df = pd.DataFrame([], columns=["Latency", "Mag"])

test_data_func2 = [
    [-30, 1e-2],
    [23, 1e-44],
    [26, 1e-59],
    [30, 1e-99],
    [130, 1e-81],
    [131, 1e-23],
    [131, 1e-76],
    [131, 1e-4],
    [200, 1e-2]
]
significant_test_data_func2 = [
    [23, 1e-44],
    [26, 1e-59],
    [30, 1e-99],
    [130, 1e-81],
    [131, 1e-23],
    [131, 1e-76],
]
test_df_func2 = pd.DataFrame(significant_test_data_func2, columns=["Latency", "Mag"])
denoised_func2 = [
    (30, 1e-99),
    (130, 1e-81)
]

noisy_test_hexels = {
    "func1": IPPMHexel("func1"),
    "func2": IPPMHexel("func2")
}
noisy_test_hexels["func1"].right_best_pairings = test_data_func1
noisy_test_hexels["func2"].right_best_pairings = test_data_func2

denoised_test_hexels = {
    "func1": IPPMHexel("func1"),
    "func2": IPPMHexel("func2")
}
denoised_test_hexels["func1"].right_best_pairings = denoised_func1
denoised_test_hexels["func2"].right_best_pairings = denoised_func2


def test_DenoisingStrategy_EstimateThresholdForSignificance_Successfully():
    expected_threshold = 3.55e-15
    actual_threshold = DenoisingStrategy._estimate_threshold_for_significance(5)
    assert isclose(expected_threshold, actual_threshold, abs_tol=1e-15)


@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._map_hexels_to_df")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._preprocess")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._get_denoised_time_series")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._postprocess")
def test_DenoisingStrategy_Denoise_Successfully(
        mock_postprocess,
        mock_get_denoised,
        mock_preprocess,
        mock_map_hexels
    ):
    expected_hexels = deepcopy(noisy_test_hexels)
    func1_hexel = deepcopy(expected_hexels["func1"])  # return value from _postprocess
    func2_hexel = deepcopy(expected_hexels["func2"])  # return value from _postprocess
    func1_hexel.right_best_pairings = denoised_func1
    func2_hexel.right_best_pairings = denoised_func2
    expected_hexels["func1"].right_best_pairings = denoised_func1
    expected_hexels["func2"].right_best_pairings = denoised_func2

    # To mock a generator, you have to return an iterable.
    mock_map_hexels.return_value = iter([("func1", test_df_func1), ("func2", test_df_func2)])
    mock_preprocess.side_effect = [test_df_func1, test_df_func2]
    clusterer = MagicMock()
    clusterer.fit.return_value = clusterer
    mock_get_denoised.side_effect = [denoised_func1, denoised_func2]
    mock_postprocess.side_effect = [func1_hexel, func2_hexel]

    strategy = DenoisingStrategy("rightHemisphere")
    strategy._clusterer = clusterer
    actual_hexels = strategy.denoise(noisy_test_hexels)

    assert actual_hexels["func1"].right_best_pairings == expected_hexels["func1"].right_best_pairings
    assert actual_hexels["func2"].right_best_pairings == expected_hexels["func2"].right_best_pairings


@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._filter_out_insignificant_spikes")
def test_DenoisingStrategy_MapHexelsToDF_Successfully(mock_filter):
    mock_filter.side_effect = [significant_test_data_func1, significant_test_data_func2]
    strategy = DenoisingStrategy("rightHemisphere")
    actual_dfs = []
    for func, df in strategy._map_hexels_to_df(noisy_test_hexels):
        actual_dfs.append((func, df))

    assert actual_dfs[0][0] == "func1"
    assert actual_dfs[1][0] == "func2"
    assert_frame_equal(actual_dfs[0][1], test_df_func1)
    assert_frame_equal(actual_dfs[1][1], test_df_func2)


def test_DenoisingStrategy_FilterOutInsignificantSpikes_Successfully():
    strategy = DenoisingStrategy("rightHemisphere")
    actual_datapoints = strategy._filter_out_insignificant_spikes(test_data_func1)
    expected_datapoints = significant_test_data_func1
    assert actual_datapoints == expected_datapoints


def test_DenoisingStrategy_UpdatePairings_Successfully():
    actual_hexel = deepcopy(noisy_test_hexels["func1"])
    strategy = DenoisingStrategy("rightHemisphere")
    actual_hexel = strategy._update_pairings(actual_hexel, denoised_func1)
    assert actual_hexel.right_best_pairings == denoised_func1


def test_DenoisingStrategy_Preprocess_Successfully():
    df = deepcopy(test_df_func2)
    latencies_only_test_data_2 = map(lambda x: x[0], significant_test_data_func2)
    sqrt_sum_squared_latencies = math.sqrt(sum(map(lambda x: x**2, latencies_only_test_data_2)))
    normed_latencies = list(map(lambda x: x[0] / sqrt_sum_squared_latencies, significant_test_data_func2))
    expected_df = pd.DataFrame(normed_latencies, columns=["Latency"])
    expected_df["Mag"] = df["Mag"]

    strategy = DenoisingStrategy("rightHemisphere", should_normalise=True, should_cluster_only_latency=True)
    preprocessed_df = strategy._preprocess(df)

    assert_frame_equal(expected_df, preprocessed_df)


def test_DenoisingStrategy_GetDenoisedTimeSeries_Successfully():
    mocked_clusterer = MagicMock()
    mocked_clusterer.labels_ = significant_test_data_func1_labels
    strategy = DenoisingStrategy("rightHemisphere")
    strategy._clusterer = mocked_clusterer
    actual = strategy._get_denoised_time_series(test_df_func1)

    assert denoised_func1 == actual


@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._perform_max_pooling")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._update_pairings")
def test_DenoisingStrategy_Postprocess_Successfully(mock_update_pairings, mock_perform_max):
    mock_update_pairings.return_value = denoised_test_hexels["func1"]

    max_pooled_hexel = deepcopy(denoised_test_hexels["func1"])
    max_pooled_hexel.right_best_pairings = [(30, 1e-100)]
    mock_perform_max.return_value = max_pooled_hexel

    strategy = DenoisingStrategy("rightHemisphere", should_max_pool=True)
    actual_hexel = strategy._postprocess(noisy_test_hexels["func1"], denoised_func1)

    assert actual_hexel == max_pooled_hexel


def test_DenoisingStrategy_PerformMaxPooling_Successfully():
    max_pooled_hexel = deepcopy(denoised_test_hexels["func1"])
    max_pooled_hexel.right_best_pairings = [(30, 1e-100)]

    strategy = DenoisingStrategy("rightHemisphere")
    actual_max_pooled = strategy._perform_max_pooling(denoised_test_hexels["func1"])

    assert actual_max_pooled.right_best_pairings == max_pooled_hexel.right_best_pairings
