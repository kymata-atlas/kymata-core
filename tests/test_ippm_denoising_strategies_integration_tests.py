from copy import deepcopy

from kymata.entities.constants import HEMI_RIGHT
from kymata.entities.expression import ExpressionPoint
from kymata.ippm.denoising_strategies import (
    MaxPoolingStrategy, AdaptiveMaxPoolingStrategy, GMMStrategy, DBSCANStrategy, MeanShiftStrategy)
from kymata.math.probability import p_to_logp, sidak_correct, p_threshold_for_sigmas

n_timepoints = 201
n_hexels = 200_000

test_data_trans1 = [
    ExpressionPoint("c", -100, "trans1", -50),
    ExpressionPoint("c",  -90, "trans1", -34),
    ExpressionPoint("c",  -95, "trans1", -8),
    ExpressionPoint("c",  -75, "trans1", -75),
    ExpressionPoint("c",  -70, "trans1", -27),
    ExpressionPoint("c",    0, "trans1", -1),
    ExpressionPoint("c",   30, "trans1", -100),
    ExpressionPoint("c",   32, "trans1", -93),
    ExpressionPoint("c",   35, "trans1", -72),
    ExpressionPoint("c",   50, "trans1", -9),
    ExpressionPoint("c",  176, "trans1", -50),
    ExpressionPoint("c",  199, "trans1", -90),
    ExpressionPoint("c",  200, "trans1", -50),
    ExpressionPoint("c",  210, "trans1", -44),
    ExpressionPoint("c",  211, "trans1", -55),
]
significant_test_data_trans1 = [
    ExpressionPoint("c", -100, "trans1", -50),
    ExpressionPoint("c",  -90, "trans1", -34),
    ExpressionPoint("c",  -75, "trans1", -75),
    ExpressionPoint("c",  -70, "trans1", -27),
    ExpressionPoint("c",   30, "trans1", -100),
    ExpressionPoint("c",   32, "trans1", -93),
    ExpressionPoint("c",   35, "trans1", -72),
    ExpressionPoint("c",  176, "trans1", -50),
    ExpressionPoint("c",  199, "trans1", -90),
    ExpressionPoint("c",  200, "trans1", -50),
    ExpressionPoint("c",  210, "trans1", -44),
    ExpressionPoint("c",  211, "trans1", -55),
]
significant_test_data_trans1_labels = [0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2]

test_data_trans2 = [
    ExpressionPoint("c", -30, "trans2", -2),
    ExpressionPoint("c",  23, "trans2", -44),
    ExpressionPoint("c",  26, "trans2", -59),
    ExpressionPoint("c",  30, "trans2", -99),
    ExpressionPoint("c", 130, "trans2", -81),
    ExpressionPoint("c", 131, "trans2", -23),
    ExpressionPoint("c", 131, "trans2", -76),
    ExpressionPoint("c", 131, "trans2", -4),
    ExpressionPoint("c", 200, "trans2", -2),
]
significant_test_data_trans2 = [
    ExpressionPoint("c",  23, "trans2", -44),
    ExpressionPoint("c",  26, "trans2", -59),
    ExpressionPoint("c",  30, "trans2", -99),
    ExpressionPoint("c", 130, "trans2", -81),
    ExpressionPoint("c", 131, "trans2", -23),
    ExpressionPoint("c", 131, "trans2", -76),
]

noisy_test_hexels = {
    "trans1":  test_data_trans1,
    "trans2":  test_data_trans2,
}

threshold_logp = p_to_logp(sidak_correct(p_threshold_for_sigmas(5), n_comparisons=200 * 200_000))


# NOTE: Max Pooling is set to true in the first integ test only. IF we set it to true, then they all do same thing.
#       Which means we can't see how well it works without the max pooling
#       For sk-learn algorithms, we use the most logical settings. E.g., normalisation rescales dimension, so
#       eps is very hard to pick prior to preprocessing for DBSCAN. Eps plays the role of bin size in DBSCAN.


def test_MaxPoolingStrategy_AllTrue_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["trans1"] = [ExpressionPoint("c", 30, "trans1", -100)]
    expected_denoised["trans2"] = [ExpressionPoint("c", 30, "trans2", -99)]

    strategy = MaxPoolingStrategy(
        should_normalise=True,
        should_cluster_only_latency=True,
        should_max_pool=True,
        bin_significance_threshold=2,
        should_shuffle=False,  # For predictability
    )
    actual_denoised = strategy._denoise_spikes(noisy_test_hexels, threshold_logp)

    assert (
        set(actual_denoised["trans1"])
        == set(expected_denoised["trans1"])
    )
    assert (
        set(actual_denoised["trans2"])
        == set(expected_denoised["trans2"])
    )


def test_MaxPoolingStrategy_AllDefault_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["trans1"] = [
        ExpressionPoint("c", -75, "trans1", -75),
        ExpressionPoint("c", 30, "trans1", -100),
    ]
    expected_denoised["trans2"] = [("c", 30, "trans2", -99)]

    strategy = MaxPoolingStrategy(HEMI_RIGHT)
    actual_denoised = strategy._denoise_spikes(noisy_test_hexels, threshold_logp)

    assert (
        set(actual_denoised["trans1"])
        == set(expected_denoised["trans1"])
    )
    assert (
        set(actual_denoised["trans2"])
        == set(expected_denoised["trans2"])
    )


def test_AdaptiveMaxPoolingStrategy_AllTrue_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["trans1"] = [("c", 30, "trans1", -100)]
    expected_denoised["trans2"] = [("c", 30, "trans2", -99)]

    strategy = AdaptiveMaxPoolingStrategy(
        should_normalise=True,
        should_cluster_only_latency=True,
        bin_significance_threshold=2,
    )
    actual_denoised = strategy._denoise_spikes(noisy_test_hexels, threshold_logp)

    assert (
        set(actual_denoised["trans1"])
        == set(expected_denoised["trans1"])
    )
    assert (
        set(actual_denoised["trans2"])
        == set(expected_denoised["trans2"])
    )


def test_AdaptiveMaxPoolingStrategy_AllDefault_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["trans1"] = [
        ExpressionPoint("c", -75, "trans1", -75),
        ExpressionPoint("c", 30, "trans1", -100),
        ExpressionPoint("c", 199, "trans1", -90),
    ]
    expected_denoised["trans2"] = [ExpressionPoint("c", 30, "trans2", -99), ExpressionPoint("c", 130, "trans2", -81)]

    strategy = AdaptiveMaxPoolingStrategy(
        bin_significance_threshold=2, base_bin_size=0.025
    )
    actual_denoised = strategy._denoise_spikes(noisy_test_hexels, threshold_logp)

    assert (
        set(actual_denoised["trans1"])
        == set(expected_denoised["trans1"])
    )
    assert (
        set(actual_denoised["trans2"])
        == set(expected_denoised["trans2"])
    )


def test_GMMStrategy_AllTrue_Fit_Successfully():
    random_seed = 40
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["trans1"] = [
        ExpressionPoint("c1", 176, "trans1", -50),
        ExpressionPoint("c1", 30, "trans1", -100),
        ExpressionPoint("c1", 199, "trans1", -90),
        ExpressionPoint("c1", -75, "trans1", -75),
    ]
    expected_denoised["trans2"] = [ExpressionPoint("c", 30, "trans2", -99), ExpressionPoint("c", 130, "trans2", -81)]

    strategy = GMMStrategy(
        should_normalise=True,
        should_shuffle=False,
        should_cluster_only_latency=True,
        number_of_clusters_upper_bound=5,
        random_state=random_seed,
    )
    actual_denoised = strategy._denoise_spikes(noisy_test_hexels, threshold_logp)

    assert (
        set(actual_denoised["trans1"])
        == set(expected_denoised["trans1"])
    )
    assert (
        set(actual_denoised["trans2"])
        == set(expected_denoised["trans2"])
    )


def test_GMMStrategy_AllDefault_Fit_Successfully():
    random_seed = 40
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["trans1"] = [
        ExpressionPoint("c", 30, "trans1", -100),
        ExpressionPoint("c", 199, "trans1", -90),
        ExpressionPoint("c", -75, "trans1", -75),
        ExpressionPoint("c", 176, "trans1", -50),
    ]
    expected_denoised["trans2"] = [
        ExpressionPoint("c", 30, "trans2", -99),
        ExpressionPoint("c", 130, "trans2", -81),
    ]

    strategy = GMMStrategy(number_of_clusters_upper_bound=5, random_state=random_seed, should_evaluate_using_aic=False)
    actual_denoised = strategy._denoise_spikes(noisy_test_hexels, threshold_logp)

    assert (
        set(actual_denoised["trans1"])
        == set(expected_denoised["trans1"])
    )
    assert (
        set(actual_denoised["trans2"])
        == set(expected_denoised["trans2"])
    )


def test_DBSCANStrategy_AllTrue_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["trans1"] = [
        ExpressionPoint("c", -75, "trans1", -75),
        ExpressionPoint("c", 30, "trans1", -100),
        ExpressionPoint("c", 199, "trans1", -90),
    ]
    expected_denoised["trans2"] = [ExpressionPoint("c", 30, "trans2", -99), ExpressionPoint("c", 130, "trans2", -81)]

    strategy = DBSCANStrategy(
        should_normalise=False,
        should_cluster_only_latency=True,
        eps=25,
    )
    actual_denoised = strategy._denoise_spikes(noisy_test_hexels, threshold_logp)

    assert (
        set(actual_denoised["trans1"])
        == set(expected_denoised["trans1"])
    )
    assert (
        set(actual_denoised["trans2"])
        == set(expected_denoised["trans2"])
    )


def test_DBSCANStrategy_AllDefault_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["trans1"] = [
        ExpressionPoint("c", -100, "trans1", -50),
        ExpressionPoint("c", -75, "trans1", -75),
        ExpressionPoint("c", 30, "trans1", -100),
        ExpressionPoint("c", 199, "trans1", -90),
    ]
    expected_denoised["trans2"] = [ExpressionPoint("c", 30, "trans2", -99), ExpressionPoint("c", 130, "trans2", -81)]

    strategy = DBSCANStrategy()
    actual_denoised = strategy._denoise_spikes(noisy_test_hexels, threshold_logp)

    assert (
        set(actual_denoised["trans1"])
        == set(expected_denoised["trans1"])
    )
    assert (
        set(actual_denoised["trans2"])
        == set(expected_denoised["trans2"])
    )


def test_MeanShiftStrategy_AllTrue_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["trans1"] = [
        ExpressionPoint("c", 199, "trans1", -90),
        ExpressionPoint("c", -75, "trans1", -75),
        ExpressionPoint("c", 30, "trans1", -100),
    ]
    expected_denoised["trans2"] = [ExpressionPoint("c", 130, "trans2", -81), ExpressionPoint("c", 30, "trans2", -99)]

    strategy = MeanShiftStrategy(
        bandwidth=0.03, min_bin_freq=2
    )
    actual_denoised = strategy._denoise_spikes(noisy_test_hexels, threshold_logp)

    assert (
        set(actual_denoised["trans1"])
        == set(expected_denoised["trans1"])
    )
    assert (
        set(actual_denoised["trans2"])
        == set(expected_denoised["trans2"])
    )


def test_MeanShiftStrategy_AllDefault_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["trans1"] = [
        ExpressionPoint("c", 199, "trans1", -90),
        ExpressionPoint("c", -75, "trans1", -75),
        ExpressionPoint("c", 30, "trans1", -100),
    ]
    expected_denoised["trans2"] = [ExpressionPoint("c", 130, "trans2", -81), ExpressionPoint("c", 30, "trans2", -99)]

    strategy = MeanShiftStrategy(bandwidth=0.03, min_bin_freq=2)
    actual_denoised = strategy._denoise_spikes(noisy_test_hexels, threshold_logp)

    assert (
        set(actual_denoised["trans1"])
        == set(expected_denoised["trans1"])
    )
    assert (
        set(actual_denoised["trans2"])
        == set(expected_denoised["trans2"])
    )
