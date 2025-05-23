from unittest.mock import patch, MagicMock

import numpy as np

from kymata.entities.expression import ExpressionPoint
from kymata.ippm.cluster import MaxPoolClusterer, AdaptiveMaxPoolClusterer, GMMClusterer, points_to_matrix

_test_dtype = np.float32

test_data = points_to_matrix([
    ExpressionPoint("c", -100, "f", -50),
    ExpressionPoint("c", -90, "f", -34),
    ExpressionPoint("c", -95, "f", -8),
    ExpressionPoint("c", -75, "f", -75),
    ExpressionPoint("c", -70, "f", -27),
    ExpressionPoint("c", 0, "f", -1),
    ExpressionPoint("c", 30, "f", -100),
    ExpressionPoint("c", 32, "f", -93),
    ExpressionPoint("c", 35, "f", -72),
    ExpressionPoint("c", 50, "f", -9),
    ExpressionPoint("c", 176, "f", -50),
    ExpressionPoint("c", 199, "f", -90),
    ExpressionPoint("c", 200, "f", -50),
    ExpressionPoint("c", 210, "f", -44),
    ExpressionPoint("c", 211, "f", -55),
]).astype(_test_dtype)
count_of_test_data_per_label = {4: 3, 5: 2, 8: 1, 9: 3, 10: 1, 15: 2, 16: 3}


def test_MaxPoolClusterer_MapLabelToNewLabel_Successfully():
    test_labels = [1, 1, 2, 3, 4, 5, 6, 6, 6, 7, 9]
    expected_labels = [1, 1, 2, 3, 4, 5, -1, -1, -1, 7, 9]
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp.labels_ = test_labels
    actual_labels = mp._map_label_to_new_label(6, -1, mp.labels_)

    assert expected_labels == actual_labels


@patch("kymata.ippm.cluster.MaxPoolClusterer._map_label_to_new_label")
def test_MaxPoolClusterer_TagLabelsBelowSignificanceThresholdAsAnomalies_Successfully(mocked_func):
    final_labels = [4, 4, 4, 5, 5, -1, 9, 9, 9, -1, 15, 15, 16, 16, 16]
    mocked_func.side_effect = [
        [4, 4, 4, 5, 5, -1, 9, 9, 9, 10, 15, 15, 16, 16, 16],
        final_labels,
    ]
    mp = MaxPoolClusterer(label_significance_threshold=2)
    assigned_labels = mp._tag_labels_below_label_significance_threshold_as_anomalies(
        [],
        count_of_test_data_per_label
    )

    assert assigned_labels == final_labels


def test_MaxPoolClusterer_AssignPointsToLabels_Successfully():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    assigned_labels = mp._assign_points_to_labels(test_data)
    expected_labels = [4, 4, 4, 5, 5, 8, 9, 9, 9, 10, 15, 15, 16, 16, 16]

    assert assigned_labels == expected_labels


@patch("kymata.ippm.cluster.MaxPoolClusterer._assign_points_to_labels")
@patch("kymata.ippm.cluster.MaxPoolClusterer._tag_labels_below_label_significance_threshold_as_anomalies")
def test_MaxPoolClusterer_Fit_Successfully(mock_tag_labels, mock_assign_points):
    mock_assign_points.return_value = [4, 4, 4, 5, 5,  8, 9, 9, 9, 10, 15, 15, 16, 16, 16]
    mock_tag_labels.return_value    = [4, 4, 4, 5, 5, -1, 9, 9, 9, -1, 15, 15, 16, 16, 16]

    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_data)
    cluster_labels = mp.labels_
    expected_labels = [4, 4, 4, 5, 5, -1, 9, 9, 9, -1, 15, 15, 16, 16, 16]

    assert cluster_labels == expected_labels


@patch("kymata.ippm.cluster.AdaptiveMaxPoolClusterer._merge_labels")
def test_AdaptiveMaxPoolClusterer_MergeSignificantLabels_Successfully(mock_merge_labels):
    mock_merge_labels.side_effect = [
        [0, 0, 0, 0, 0, -1, 9, 9, 9, -1, 15, 15, 16, 16, 16],
        [0, 0, 0, 0, 0, -1, 1, 1, 1, -1, 15, 15, 16, 16, 16],
        [0, 0, 0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2, 2],
    ]
    amp = AdaptiveMaxPoolClusterer(base_label_size=25)
    amp.labels_ = [4, 4, 4, 5, 5, -1, 9, 9, 9, -1, 15, 15, 16, 16, 16]
    amp.labels_ = amp._merge_significant_labels(amp.labels_, test_data)
    actual_labels = amp.labels_
    expected_labels = [0, 0, 0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2, 2]

    assert actual_labels == expected_labels


@patch("kymata.ippm.cluster.MaxPoolClusterer._assign_points_to_labels")
@patch("kymata.ippm.cluster.MaxPoolClusterer._tag_labels_below_label_significance_threshold_as_anomalies")
@patch("kymata.ippm.cluster.AdaptiveMaxPoolClusterer._merge_significant_labels")
def test_Should_AdaptiveMaxPoolClusterer_Fit_Successfully(
    mock_merge_significant, mock_tag_labels, mock_assign_points
):
    mock_assign_points.return_value = [4, 4, 4, 5, 5,  8, 9, 9, 9, 10, 15, 15, 16, 16, 16]
    mock_tag_labels.return_value    = [4, 4, 4, 5, 5, -1, 9, 9, 9, -1, 15, 15, 16, 16, 16]
    mock_merge_significant.return_value = [0, 0, 0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2, 2]

    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_data)
    cluster_labels = amp.labels_
    expected_labels = [0, 0, 0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2, 2]

    assert cluster_labels == expected_labels


@patch("kymata.ippm.cluster.GaussianMixture")
def test_Should_GMMClusterer_GridSearchForOptimalNumberOfClusters_Successfully(mock_gmm):
    first_gmm_mock = MagicMock(name="first_gmm")
    first_gmm_mock.bic.return_value = 100
    second_gmm_mock = MagicMock(name="second_gmm")
    second_gmm_mock.bic.return_value = 67
    mocked_best_fit_gmm_instance = MagicMock(name="best_fit_gmm")
    mocked_best_fit_gmm_instance.bic.return_value = 7
    mock_gmm.side_effect = [
        first_gmm_mock,
        second_gmm_mock,
        mocked_best_fit_gmm_instance,
    ]

    gmm = GMMClusterer(number_of_clusters_upper_bound=4)
    optimal_model = gmm._grid_search_for_optimal_number_of_clusters(test_data)
    assert optimal_model == mocked_best_fit_gmm_instance


def test_Should_GMMClusterer_TagLowLogLikelihoodPointsAsAnomalous_Successfully():
    mocked_gmm_instance = MagicMock()
    mocked_gmm_instance.score_samples.return_value = [-100, -5, -50]

    gmm = GMMClusterer()
    gmm.labels_ = [0, 1, 2]
    actual_labels = gmm._tag_low_loglikelihood_points_as_anomalous(
        test_data, mocked_gmm_instance
    )

    assert actual_labels == [-1, 1, 2]


@patch("kymata.ippm.cluster.GMMClusterer._grid_search_for_optimal_number_of_clusters")
def test_Should_GMMClusterer_Fit_Successfully(mock_grid_search):
    mocked_optimal_model = MagicMock()
    mocked_optimal_model.predict.return_value = [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4]
    mock_grid_search.return_value = mocked_optimal_model

    gmm = GMMClusterer()
    gmm = gmm.fit(test_data)

    assert gmm.labels_ == [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4]
