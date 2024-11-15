from copy import deepcopy
from math import isclose
from unittest.mock import patch, MagicMock

from kymata.entities.constants import HEMI_RIGHT
from kymata.ippm.data_tools import IPPMSpike, ExpressionPairing
from kymata.ippm.denoising_strategies import DenoisingStrategy

n_timepoints = 201
n_hexels = 200_000

test_data_trans1 = [
    ExpressionPairing(-100, -50),
    ExpressionPairing(-90, -34),
    ExpressionPairing(-95, -8),
    ExpressionPairing(-75, -75),
    ExpressionPairing(-70, -27),
    ExpressionPairing(0, -1),
    ExpressionPairing(30, -100),
    ExpressionPairing(32, -93),
    ExpressionPairing(35, -72),
    ExpressionPairing(50, -9),
    ExpressionPairing(176, -50),
    ExpressionPairing(199, -90),
    ExpressionPairing(200, -50),
    ExpressionPairing(210, -44),
    ExpressionPairing(211, -55),
]
significant_test_data_trans1 = [
    ExpressionPairing(-100, -50),
    ExpressionPairing(-90, -34),
    ExpressionPairing(-75, -75),
    ExpressionPairing(-70, -27),
    ExpressionPairing(30, -100),
    ExpressionPairing(32, -93),
    ExpressionPairing(35, -72),
    ExpressionPairing(176, -50),
    ExpressionPairing(199, -90),
    ExpressionPairing(200, -50),
    ExpressionPairing(210, -44),
    ExpressionPairing(211, -55),
]
significant_test_data_trans1_labels = [0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2]
denoised_trans1 = [
    ExpressionPairing(-75, -75),
    ExpressionPairing(30, -100),
    ExpressionPairing(199, -90)
]

test_data_empty = []

test_data_trans2 = [
    ExpressionPairing(-30, -2),
    ExpressionPairing(23, -44),
    ExpressionPairing(26, -59),
    ExpressionPairing(30, -99),
    ExpressionPairing(130, -81),
    ExpressionPairing(131, -23),
    ExpressionPairing(131, -76),
    ExpressionPairing(131, -4),
    ExpressionPairing(200, -2),
]
significant_test_data_trans2 = [
    ExpressionPairing(23, -44),
    ExpressionPairing(26, -59),
    ExpressionPairing(30, -99),
    ExpressionPairing(130, -81),
    ExpressionPairing(131, -23),
    ExpressionPairing(131, -76),
]
denoised_trans2 = [
    ExpressionPairing(30, -99),
    ExpressionPairing(130, -81)
]

noisy_test_spikes = {"trans1": IPPMSpike("trans1"), "trans2": IPPMSpike("trans2")}
noisy_test_spikes["trans1"].right_best_pairings = test_data_trans1
noisy_test_spikes["trans2"].right_best_pairings = test_data_trans2

denoised_test_spikes = {"trans1": IPPMSpike("trans1"), "trans2": IPPMSpike("trans2")}
denoised_test_spikes["trans1"].right_best_pairings = denoised_trans1
denoised_test_spikes["trans2"].right_best_pairings = denoised_trans2


def test_DenoisingStrategy_EstimateThresholdForSignificance_Successfully():
    expected_threshold = 3.55e-15
    actual_threshold = DenoisingStrategy._estimate_threshold_for_significance(5, n_timepoints=n_timepoints, n_hexels=n_hexels)
    assert isclose(expected_threshold, actual_threshold, abs_tol=15)


@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._map_spikes_to_pairings")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._preprocess")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._get_denoised_time_series")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._postprocess")
def test_DenoisingStrategy_Denoise_Successfully(
    mock_postprocess, mock_get_denoised, mock_preprocess, mock_map_spikes
):
    expected_spikes = deepcopy(noisy_test_spikes)
    trans1_spike = deepcopy(expected_spikes["trans1"])  # return value from _postprocess
    trans2_spike = deepcopy(expected_spikes["trans2"])  # return value from _postprocess
    trans1_spike.right_best_pairings = denoised_trans1
    trans2_spike.right_best_pairings = denoised_trans2
    expected_spikes["trans1"].right_best_pairings = denoised_trans1
    expected_spikes["trans2"].right_best_pairings = denoised_trans2

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

    assert (
        actual_spikes["trans1"].right_best_pairings ==
        expected_spikes["trans1"].right_best_pairings
    )
    assert (
        actual_spikes["trans2"].right_best_pairings ==
        expected_spikes["trans2"].right_best_pairings
    )


@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._filter_out_insignificant_pairings")
def test_DenoisingStrategy_MapSpikesToPairings_Successfully(mock_filter):
    mock_filter.side_effect = [significant_test_data_trans1, significant_test_data_trans2]
    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels)

    actual_spikes: list[tuple[str, list[ExpressionPairing]]] = list(strategy._map_spikes_to_pairings(noisy_test_spikes))

    assert len(actual_spikes) == 2
    assert actual_spikes[0][0] == "trans1"
    assert actual_spikes[1][0] == "trans2"

    assert (
        set(pairing.latency_ms for _name, pairings in actual_spikes for pairing in pairings) ==
        set(pairing.latency_ms for pairing in test_data_trans1)
    )
    assert (
            set(pairing.logp_value for _name, pairings in actual_spikes for pairing in pairings) ==
            set(pairing.logp_value for pairing in test_data_trans1)
    )


def test_DenoisingStrategy_FilterOutInsignificantSpikes_Successfully():
    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels)
    actual_datapoints = strategy._filter_out_insignificant_pairings(test_data_trans1)
    expected_datapoints = significant_test_data_trans1
    assert actual_datapoints == expected_datapoints


def test_DenoisingStrategy_UpdatePairings_Successfully():
    actual_spike = deepcopy(noisy_test_spikes["trans1"])
    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels)
    actual_spike = strategy._update_pairings(actual_spike, denoised_trans1)
    assert actual_spike.right_best_pairings == denoised_trans1


def test_DenoisingStrategy_Preprocess_Successfully():
    test_data = deepcopy(test_data_trans2)
    latencies_only_test_data = [p.latency_ms for p in test_data]
    sum_latency = sum(latencies_only_test_data)
    normed_latencies = [latency / sum_latency for latency in list(latencies_only_test_data)]

    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels,
                                 should_normalise=True, should_cluster_only_latency=True)
    preprocessed_pairings = strategy._preprocess(test_data)

    assert [p.latency_ms for p in preprocessed_pairings] == normed_latencies


def test_DenoisingStrategy_GetDenoisedTimeSeries_Successfully():
    mocked_clusterer = MagicMock()
    mocked_clusterer.labels_ = significant_test_data_trans1_labels
    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels)
    strategy._clusterer = mocked_clusterer
    actual = strategy._get_denoised_time_series(test_data_trans1)

    assert actual == denoised_trans1


@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._perform_max_pooling")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._update_pairings")
def test_DenoisingStrategy_Postprocess_Successfully(mock_update_pairings, mock_perform_max):
    mock_update_pairings.return_value = denoised_test_spikes["trans1"]

    max_pooled_spike = deepcopy(denoised_test_spikes["trans1"])
    max_pooled_spike.right_best_pairings = [ExpressionPairing(30, -100)]
    mock_perform_max.return_value = max_pooled_spike

    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels,
                                 should_max_pool=True)
    actual_spike = strategy._postprocess(noisy_test_spikes["trans1"], denoised_trans1)

    assert actual_spike == max_pooled_spike


def test_DenoisingStrategy_PerformMaxPooling_Successfully():
    max_pooled_spike = deepcopy(denoised_test_spikes["trans1"])
    max_pooled_spike.right_best_pairings = [ExpressionPairing(30, -100)]

    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels)
    actual_max_pooled = strategy._perform_max_pooling(denoised_test_spikes["trans1"])

    assert actual_max_pooled.right_best_pairings == max_pooled_spike.right_best_pairings
