import math
from copy import deepcopy
from typing import List
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from numpy._typing import ArrayLike

from kymata.entities.constants import HEMI_RIGHT, HEMI_LEFT
from kymata.entities.expression import ExpressionPoint, HexelExpressionSet
from kymata.ippm.denoising_strategies import DenoisingStrategy, MaxPoolingStrategy
from kymata.math.probability import p_to_logp

"""
    To construct a HexelExpressionSet, we need to create a matrix with logp values. The dimensions for the matrix are
    (DIM_CHANNEL, DIM_LATENCY, DIM_TRANSFORM). Fill in values up to 5*12 = [0,60] (inclusive) or 13 values. 
    Note that each (channel, latency, transform) -> single magnitude, so at most 1 value per c, l, h.

    The denoising procedure currently operates by extracting the most significant point per channel, followed
    by grouping them into a key, value struct where key is transform id, denoising it using a clusterer, then returning
    the extracted points.
"""
N_CHANNELS = 8
N_LATENCIES = 13  # goes from 0, 5, 10, ..., 60 (inclusive)
N_TIMEPOINTS = 201
N_HEXELS = 200_000

# Scenario: [cluster1, noise, cluster2, noise]
noisy_data1 = [
    # Cluster 1
    ExpressionPoint(0, 0, "f1", -50),
    ExpressionPoint(2, 5, "f1", -34),
    ExpressionPoint(1, 5, "f1", -70),

    # Noise 1
    ExpressionPoint(5, 15, "f1", -15),
    ExpressionPoint(6, 25, "f1", -7),

    # Cluster 2
    ExpressionPoint(3, 30, "f1", -100),
    ExpressionPoint(4, 35, "f1", -50),

    # Noise 2
    ExpressionPoint(7, 40, "f1", -3),

    # Magnitudes per channel must be less than the clustered ones cf HexelExpressionSet.best_transforms
    ExpressionPoint(1, 10, "f1", -20),
    ExpressionPoint(1, 45, "f1", -9),
    ExpressionPoint(0, 15, "f1", -20),
    ExpressionPoint(2, 15, "f1", -22),
    ExpressionPoint(2, 30, "f1", -11),
    ExpressionPoint(3, 50, "f1", -3),
    ExpressionPoint(4, 40, "f1", -45),
]
# best transforms = take max per channel. This is what is clustered on
best_data1 = noisy_data1[:8]
grouped_data1 = {"f1": best_data1}
denoised_data1 = [
    ExpressionPoint(1, 5, "f1", -70),
    ExpressionPoint(3, 30, "f1", -100)
]

# Scenario: [noise, cluster1, noise]
noisy_data2 = [
    # Noise
    ExpressionPoint(0, 0, "f2", -2),
    ExpressionPoint(1, 5, "f2", -50),

    # Cluster
    ExpressionPoint(2, 25, "f2", -99),
    ExpressionPoint(3, 25, "f2", -78),
    ExpressionPoint(4, 30, "f2", -23),

    # Noise
    ExpressionPoint(5, 40, "f2", -77),
    ExpressionPoint(6, 50, "f2", -54),
    ExpressionPoint(7, 55, "f2", -4),

    # Noise
    ExpressionPoint(0, 35, "f2", -1),
    ExpressionPoint(1, 15, "f2", -44),
    ExpressionPoint(2, 40, "f2", -59),
    ExpressionPoint(3, 20, "f2", -37),
    ExpressionPoint(4, 55, "f2", -5),
    ExpressionPoint(5, 60, "f2", -23),
    ExpressionPoint(6, 55, "f2", -26),
    ExpressionPoint(7, 5, "f2", -2),
]
best_data2 = noisy_data2[:8]  # first 8 are max over channel
grouped_data2 = {"f2": best_data2}
denoised_data2 = [
    ExpressionPoint(2, 25, "f2", -99)
]


def create_data_block(
        test_data: List[ExpressionPoint],
        transform_idx: int,
        time_range: int,
        n_channels: int
) -> ArrayLike:
    # default value for transform. Set transform_idx values manually, zero out other transforms
    default_values = [np.nan, 0] if transform_idx == 0 else [0, np.nan]
    data = np.array([
        [default_values for _ in range(time_range)] for _ in range(n_channels)
    ])

    for point in test_data:
        latency_idx = point.latency // 5
        data[point.channel][latency_idx][transform_idx] = point.logp_value

    data = np.nan_to_num(data, nan=0)
    return data


noisy_test_hexel_expr_set = HexelExpressionSet(
    transforms=["f1", "f2"],
    hexels_lh=[i for i in range(N_CHANNELS)],
    hexels_rh=[i for i in range(N_CHANNELS)],
    latencies=[i for i in range(0, N_LATENCIES * 5, 5)],
    # Data follows (DIM_CHANNEL, DIM_LATENCY, DIM_TRANSFORM)
    data_lh=create_data_block(
        noisy_data1,
        transform_idx=0,
        time_range=N_LATENCIES,
        n_channels=N_CHANNELS
    ),
    data_rh=create_data_block(
        noisy_data2,
        transform_idx=1,
        time_range=N_LATENCIES,
        n_channels=N_CHANNELS
    )
)


@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._denoise_spikes")
@patch("kymata.ippm.denoising_strategies.group_points_by_transform")
@patch("kymata.ippm.denoising_strategies.HexelExpressionSet.best_transforms")
def test_DenoisingStrategy_DenoiseHexelExpressionSet_Successfully(
        mock_best_transforms,
        mock_group_by,
        mock_denoise,
):
    # set-up mocks
    mock_best_transforms.return_value = best_data1, best_data2
    mock_group_by.side_effect = [grouped_data2, grouped_data1]
    mock_denoise.side_effect = [
        {"f2": denoised_data2},
        {"f1": denoised_data1}
    ]

    hexel_expr_set = deepcopy(noisy_test_hexel_expr_set)
    strategy = DenoisingStrategy()
    actual_spikes_left, actual_spikes_right = strategy._denoise_hexel_expression_set(hexel_expr_set)

    assert actual_spikes_left == denoised_data1
    assert actual_spikes_right == denoised_data2


@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._get_denoised_time_series")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._preprocess")
@patch("kymata.ippm.denoising_strategies.DenoisingStrategy._filter_out_insignificant_points")
def test_DenoisingStrategy_DenoiseSpikes_Successfully(
        mock_filter_points,
        mock_preprocess,
        mock_get_denoised
):
    mock_filter_points.return_value = list(filter(lambda x: x.logp_value < -5, best_data1))
    mock_preprocess.return_value = [
        [0, -50], [30, -100], [5, -34], [15, -15], [5, -70], [25, -7], [35, -50]
    ]
    mock_clusterer = MagicMock()
    mock_clusterer.labels_ = [0, 0, 0, -1, 1, 1]
    mock_get_denoised.return_value = denoised_data1

    strategy = DenoisingStrategy()
    strategy._clusterer = mock_clusterer
    actual = strategy._denoise_spikes(grouped_data1, -5)

    assert actual["f1"] == denoised_data1


def test_DenoisingStrategy_Preprocess_Successfully():
    test_data = deepcopy(noisy_data2)
    latencies_only_test_data = [p.latency for p in test_data]
    sum_latency = sum(latencies_only_test_data)
    normed_latencies = [latency / sum_latency for latency in list(latencies_only_test_data)]

    strategy = MaxPoolingStrategy(should_normalise=True, should_cluster_only_latency=True)
    preprocessed_pairings = strategy._preprocess(test_data)

    assert [p for p in preprocessed_pairings] == normed_latencies


def test_DenoisingStrategy_GetDenoisedTimeSeries_Successfully():
    signif_best_data1 = list(filter(lambda x: x.logp_value < -10, best_data1))
    mocked_clusterer = MagicMock()
    mocked_clusterer.labels_ = [0, 0, 0, -1, 1, 1]
    strategy = MaxPoolingStrategy()
    strategy._clusterer = mocked_clusterer
    actual = strategy._get_denoised_time_series(signif_best_data1)

    assert actual == denoised_data1


def test_DenoisingStrategy_PerformMaxPooling_Successfully():
    max_pooled_spike = [ExpressionPoint(3, 30, "f1", -100)]

    strategy = DenoisingStrategy(HEMI_RIGHT, n_timepoints=N_TIMEPOINTS, n_hexels=N_HEXELS)
    actual_max_pooled = strategy._perform_max_pooling(denoised_data1)

    assert actual_max_pooled == max_pooled_spike
