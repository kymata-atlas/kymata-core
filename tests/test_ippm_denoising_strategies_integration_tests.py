from copy import deepcopy

from kymata.entities.constants import HEMI_RIGHT
from kymata.entities.expression import ExpressionPoint
from kymata.ippm.denoising_strategies import (
    MaxPoolingStrategy, AdaptiveMaxPoolingStrategy, GMMStrategy, DBSCANStrategy, MeanShiftStrategy)

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

noisy_test_hexels = {
    "trans1":  test_data_trans1,
    "trans2":  test_data_trans2,
}


# NOTE: Max Pooling is set to true in the first integ test only. IF we set it to true, then they all do same thing.
#       Which means we can't see how well it works without the max pooling
#       For sk-learn algorithms, we use the most logical settings. E.g., normalisation rescales dimension, so
#       eps is very hard to pick prior to preprocessing for DBSCAN. Eps plays the role of bin size in DBSCAN.


def test_MaxPoolingStrategy_AllTrue_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["trans1"] = [ExpressionPoint("c", 30, "f", 1e-100)]
    expected_denoised["trans2"] = [ExpressionPoint("c", 30, "f", 1e-99)]

    strategy = MaxPoolingStrategy(
        hemi=HEMI_RIGHT,
        n_timepoints=n_timepoints, n_hexels=n_hexels,
        should_normalise=True,
        should_cluster_only_latency=True,
        should_max_pool=True,
        bin_significance_threshold=2,
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

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
    actual_denoised = strategy.denoise(noisy_test_hexels)

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
        hemi=HEMI_RIGHT,
        n_timepoints=n_timepoints, n_hexels=n_hexels,
        should_normalise=True,
        should_cluster_only_latency=True,
        bin_significance_threshold=2,
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

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
        HEMI_RIGHT,
        n_timepoints=n_timepoints, n_hexels=n_hexels,
        bin_significance_threshold=2, base_bin_size=0.025
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

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
        HEMI_RIGHT,
        n_timepoints=n_timepoints, n_hexels=n_hexels,
        should_normalise=True,
        should_cluster_only_latency=True,
        number_of_clusters_upper_bound=5,
        random_state=random_seed,
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

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

    strategy = GMMStrategy(HEMI_RIGHT,
                           n_timepoints=n_timepoints, n_hexels=n_hexels,
                           number_of_clusters_upper_bound=5, random_state=random_seed, should_evaluate_using_AIC=False)
    actual_denoised = strategy.denoise(noisy_test_hexels)

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
        HEMI_RIGHT,
        n_timepoints=n_timepoints, n_hexels=n_hexels,
        should_normalise=False,
        should_cluster_only_latency=True,
        eps=25,
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

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

    strategy = DBSCANStrategy(HEMI_RIGHT, n_timepoints=n_timepoints, n_hexels=n_hexels)
    actual_denoised = strategy.denoise(noisy_test_hexels)

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
        HEMI_RIGHT,
        n_timepoints=n_timepoints, n_hexels=n_hexels,
        bandwidth=0.03, min_bin_freq=2
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

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

    strategy = MeanShiftStrategy(HEMI_RIGHT,
                                 n_timepoints=n_timepoints, n_hexels=n_hexels,
                                 bandwidth=0.03, min_bin_freq=2)
    actual_denoised = strategy.denoise(noisy_test_hexels)

    assert (
        set(actual_denoised["trans1"])
        == set(expected_denoised["trans1"])
    )
    assert (
        set(actual_denoised["trans2"])
        == set(expected_denoised["trans2"])
    )
