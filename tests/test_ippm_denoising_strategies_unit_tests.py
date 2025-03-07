from copy import deepcopy
from math import isclose
from unittest.mock import patch, MagicMock

import pytest

from kymata.entities.constants import HEMI_RIGHT
from kymata.entities.expression import ExpressionPoint
from kymata.ippm.denoising_strategies import DenoisingStrategy

n_timepoints = 201
n_hexels = 200_000

test_data_trans1 = [
    ExpressionPoint("c", -100, "f1", -50),
    ExpressionPoint("c",  -90, "f1", -34),
    ExpressionPoint("c",  -95, "f1", -8),
    ExpressionPoint("c",  -75, "f1", -75),
    ExpressionPoint("c",  -70, "f1", -27),
    ExpressionPoint("c",    0, "f1", -1),
    ExpressionPoint("c",   30, "f1", -100),
    ExpressionPoint("c",   32, "f1", -93),
    ExpressionPoint("c",   35, "f1", -72),
    ExpressionPoint("c",   50, "f1", -9),
    ExpressionPoint("c",  176, "f1", -50),
    ExpressionPoint("c",  199, "f1", -90),
    ExpressionPoint("c",  200, "f1", -50),
    ExpressionPoint("c",  210, "f1", -44),
    ExpressionPoint("c",  211, "f1", -55),
]
significant_test_data_trans1 = [
    ExpressionPoint("c", -100, "f1", -50),
    ExpressionPoint("c",  -90, "f1", -34),
    ExpressionPoint("c",  -75, "f1", -75),
    ExpressionPoint("c",  -70, "f1", -27),
    ExpressionPoint("c",   30, "f1", -100),
    ExpressionPoint("c",   32, "f1", -93),
    ExpressionPoint("c",   35, "f1", -72),
    ExpressionPoint("c",  176, "f1", -50),
    ExpressionPoint("c",  199, "f1", -90),
    ExpressionPoint("c",  200, "f1", -50),
    ExpressionPoint("c",  210, "f1", -44),
    ExpressionPoint("c",  211, "f1", -55),
]
significant_test_data_trans1_labels = [0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2]
denoised_trans1 = [
    ExpressionPoint("c", -75, "f1", -75),
    ExpressionPoint("c",  30, "f1", -100),
    ExpressionPoint("c", 199, "f1", -90)
]

test_data_empty = []

test_data_trans2 = [
    ExpressionPoint("c", -30, "f2", -2),
    ExpressionPoint("c",  23, "f2", -44),
    ExpressionPoint("c",  26, "f2", -59),
    ExpressionPoint("c",  30, "f2", -99),
    ExpressionPoint("c", 130, "f2", -81),
    ExpressionPoint("c", 131, "f2", -23),
    ExpressionPoint("c", 131, "f2", -76),
    ExpressionPoint("c", 131, "f2", -4),
    ExpressionPoint("c", 200, "f2", -2),
]
significant_test_data_trans2 = [
    ExpressionPoint("c",  23, "f2", -44),
    ExpressionPoint("c",  26, "f2", -59),
    ExpressionPoint("c",  30, "f2", -99),
    ExpressionPoint("c", 130, "f2", -81),
    ExpressionPoint("c", 131, "f2", -23),
    ExpressionPoint("c", 131, "f2", -76),
]
denoised_trans2 = [
    ExpressionPoint("c",  30, "f2", -99),
    ExpressionPoint("c", 130, "f", -81)
]

noisy_test_spikes = {
    "trans1": test_data_trans1,
    "trans2": test_data_trans2,
}
denoised_test_spikes = {
    "trans1": denoised_trans1,
    "trans2": denoised_trans2,
}


@pytest.mark.skip(reason="Currently skipped due to mocking which @caiw doesn't understand. "
                         "See [Issue #441](https://github.com/kymata-atlas/kymata-core/issues/441)")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._map_spikes_to_pairings")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._preprocess")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._get_denoised_time_series")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._postprocess")
def test_DenoisingStrategy_Denoise_Successfully(mock_postprocess, mock_get_denoised, mock_preprocess, mock_map_spikes):
    expected_spikes = deepcopy(noisy_test_spikes)
    expected_spikes["trans1"] = denoised_trans1
    expected_spikes["trans2"] = denoised_trans2
    trans1_spike = denoised_trans1
    trans2_spike = denoised_trans2

    # To mock a generator, you have to return an iterable.
    mock_map_spikes.return_value = iter(
        [("trans1", test_data_trans1), ("trans2", test_data_trans2)]
    )
    mock_preprocess.side_effect = [test_data_trans1, test_data_trans2]
    clusterer = MagicMock()
    clusterer.fit.return_value = clusterer
    mock_get_denoised.side_effect = [denoised_trans1, denoised_trans2]
    mock_postprocess.side_effect = [trans1_spike, trans2_spike]

    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels)
    strategy._clusterer = clusterer
    actual_spikes = strategy.denoise(noisy_test_spikes)

    assert (actual_spikes["trans1"] == expected_spikes["trans1"])
    assert (actual_spikes["trans2"] == expected_spikes["trans2"])


@pytest.mark.skip(reason="Currently skipped due to mocking which @caiw doesn't understand. "
                         "See [Issue #441](https://github.com/kymata-atlas/kymata-core/issues/441)")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._filter_out_insignificant_pairings")
def test_DenoisingStrategy_MapSpikesToPairings_Successfully(mock_filter):
    mock_filter.side_effect = [significant_test_data_trans1, significant_test_data_trans2]
    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels)

    actual_spikes: list[tuple[str, list[ExpressionPoint]]] = list(strategy._apply_filters(noisy_test_spikes))

    assert len(actual_spikes) == 2
    assert actual_spikes[0][0] == "trans1"
    assert actual_spikes[1][0] == "trans2"

    assert (
        set(pairing.latency_ms
            for _name, pairings in actual_spikes
            for pairing in pairings)
        ==
        set(pairing.latency_ms
            for pairing in test_data_trans1)
    )
    assert (
        set(pairing.logp_value
            for _name, pairings in actual_spikes
            for pairing in pairings)
        ==
        set(pairing.logp_value
            for pairing in test_data_trans1)
    )


@pytest.mark.skip(reason="Currently skipped because DenoisingStrategy is an ABC. "
                         "See [Issue #441](https://github.com/kymata-atlas/kymata-core/issues/441)")
def test_DenoisingStrategy_Preprocess_Successfully():
    test_data = deepcopy(test_data_trans2)
    latencies_only_test_data = [p.latency for p in test_data]
    sum_latency = sum(latencies_only_test_data)
    normed_latencies = [latency / sum_latency for latency in list(latencies_only_test_data)]

    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels,
                                 should_normalise=True, should_cluster_only_latency=True)
    preprocessed_pairings = strategy._preprocess(test_data)

    assert [p.latency_ms for p in preprocessed_pairings] == normed_latencies


@pytest.mark.skip(reason="Currently skipped because DenoisingStrategy is an ABC. "
                         "See [Issue #441](https://github.com/kymata-atlas/kymata-core/issues/441)")
def test_DenoisingStrategy_GetDenoisedTimeSeries_Successfully():
    mocked_clusterer = MagicMock()
    mocked_clusterer.labels_ = significant_test_data_trans1_labels
    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels)
    strategy._clusterer = mocked_clusterer
    actual = strategy._get_denoised_time_series(test_data_trans1)

    assert actual == denoised_trans1


def test_DenoisingStrategy_PerformMaxPooling_Successfully():
    max_pooled_spike = [ExpressionPoint("c", 30, "f1", -100)]

    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels)
    actual_max_pooled = strategy._perform_max_pooling(denoised_test_spikes["trans1"])

    assert actual_max_pooled == max_pooled_spike
