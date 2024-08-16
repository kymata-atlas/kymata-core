from copy import deepcopy
import pandas as pd

from kymata.ippm.data_tools import IPPMHexel
from kymata.entities.constants import HEMI_RIGHT
from kymata.ippm.denoising_strategies import (
    MaxPoolingStrategy,
    AdaptiveMaxPoolingStrategy,
    GMMStrategy,
    DBSCANStrategy,
    MeanShiftStrategy,
)

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
    [211, 1e-55],
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
    [211, 1e-55],
]
significant_test_data_func1_labels = [0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2]
test_df_func1 = pd.DataFrame(significant_test_data_func1, columns=["Latency", "Mag"])

test_data_func2 = [
    [-30, 1e-2],
    [23, 1e-44],
    [26, 1e-59],
    [30, 1e-99],
    [130, 1e-81],
    [131, 1e-23],
    [131, 1e-76],
    [131, 1e-4],
    [200, 1e-2],
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

noisy_test_hexels = {"func1": IPPMHexel("func1"), "func2": IPPMHexel("func2")}
noisy_test_hexels["func1"].right_best_pairings = test_data_func1
noisy_test_hexels["func2"].right_best_pairings = test_data_func2


# NOTE: Max Pooling is set to true in the first integ test only. IF we set it to true, then they all do same thing.
#       Which means we can't see how well it works without the max pooling
#       For sk-learn algorithms, we use the most logical settings. E.g., normalisation rescales dimension, so
#       eps is very hard to pick prior to preprocessing for DBSCAN. Eps plays the role of bin size in DBSCAN.


def test_MaxPoolingStrategy_AllTrue_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["func1"].right_best_pairings = [(30, 1e-100)]
    expected_denoised["func2"].right_best_pairings = [(30, 1e-99)]

    strategy = MaxPoolingStrategy(
        hemi=HEMI_RIGHT,
        should_normalise=True,
        should_cluster_only_latency=True,
        should_max_pool=True,
        bin_significance_threshold=2,
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

    assert (
        actual_denoised["func1"].right_best_pairings
        == expected_denoised["func1"].right_best_pairings
    )
    assert (
        actual_denoised["func2"].right_best_pairings
        == expected_denoised["func2"].right_best_pairings
    )


def test_MaxPoolingStrategy_AllFalse_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["func1"].right_best_pairings = [
        (-100, 1e-50),
        (-75, 1e-75),
        (30, 1e-100),
        (199, 1e-90),
        (211, 1e-55),
    ]
    expected_denoised["func2"].right_best_pairings = [(30, 1e-99), (130, 1e-81)]

    strategy = MaxPoolingStrategy(HEMI_RIGHT, bin_significance_threshold=2)
    actual_denoised = strategy.denoise(noisy_test_hexels)

    assert (
        actual_denoised["func1"].right_best_pairings
        == expected_denoised["func1"].right_best_pairings
    )
    assert (
        actual_denoised["func2"].right_best_pairings
        == expected_denoised["func2"].right_best_pairings
    )


def test_AdaptiveMaxPoolingStrategy_AllTrue_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["func1"].right_best_pairings = [(30, 1e-100)]
    expected_denoised["func2"].right_best_pairings = [(30, 1e-99)]

    strategy = AdaptiveMaxPoolingStrategy(
        hemi=HEMI_RIGHT,
        should_normalise=True,
        should_cluster_only_latency=True,
        bin_significance_threshold=2,
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

    assert (
        actual_denoised["func1"].right_best_pairings
        == expected_denoised["func1"].right_best_pairings
    )
    assert (
        actual_denoised["func2"].right_best_pairings
        == expected_denoised["func2"].right_best_pairings
    )


def test_AdaptiveMaxPoolingStrategy_AllFalse_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["func1"].right_best_pairings = [
        (-75, 1e-75),
        (30, 1e-100),
        (199, 1e-90),
    ]
    expected_denoised["func2"].right_best_pairings = [(30, 1e-99), (130, 1e-81)]

    strategy = AdaptiveMaxPoolingStrategy(
        HEMI_RIGHT, bin_significance_threshold=2, base_bin_size=25
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

    assert (
        actual_denoised["func1"].right_best_pairings
        == expected_denoised["func1"].right_best_pairings
    )
    assert (
        actual_denoised["func2"].right_best_pairings
        == expected_denoised["func2"].right_best_pairings
    )


def test_GMMStrategy_AllTrue_Fit_Successfully():
    random_seed = 40
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["func1"].right_best_pairings = [
        (30, 1e-100),
        (199, 1e-90),
        (-75, 1e-75),
    ]
    expected_denoised["func2"].right_best_pairings = [(26, 1e-59), (130, 1e-81)]

    strategy = GMMStrategy(
        HEMI_RIGHT,
        should_normalise=True,
        should_cluster_only_latency=True,
        random_state=random_seed,
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

    assert (
        actual_denoised["func1"].right_best_pairings
        == expected_denoised["func1"].right_best_pairings
    )
    assert (
        actual_denoised["func2"].right_best_pairings
        == expected_denoised["func2"].right_best_pairings
    )


def test_GMMStrategy_AllFalse_Fit_Successfully():
    random_seed = 40
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["func1"].right_best_pairings = [
        (30, 1e-100),
        (199, 1e-90),
        (-75, 1e-75),
        (176, 1e-50),
    ]
    expected_denoised["func2"].right_best_pairings = [
        (30, 1e-99),
        (131, 1e-76),
        (23, 1e-44),
        (26, 1e-59),
    ]

    strategy = GMMStrategy(HEMI_RIGHT, random_state=random_seed)
    actual_denoised = strategy.denoise(noisy_test_hexels)

    assert (
        actual_denoised["func1"].right_best_pairings
        == expected_denoised["func1"].right_best_pairings
    )
    assert (
        actual_denoised["func2"].right_best_pairings
        == expected_denoised["func2"].right_best_pairings
    )


def test_DBSCANStrategy_AllTrue_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["func1"].right_best_pairings = [
        (-75, 1e-75),
        (30, 1e-100),
        (199, 1e-90),
    ]
    expected_denoised["func2"].right_best_pairings = [(30, 1e-99), (130, 1e-81)]

    strategy = DBSCANStrategy(
        HEMI_RIGHT,
        should_normalise=False,
        should_cluster_only_latency=True,
        eps=25,
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

    assert (
        actual_denoised["func1"].right_best_pairings
        == expected_denoised["func1"].right_best_pairings
    )
    assert (
        actual_denoised["func2"].right_best_pairings
        == expected_denoised["func2"].right_best_pairings
    )


def test_DBSCANStrategy_AllFalse_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["func1"].right_best_pairings = [
        (-100, 1e-50),
        (-75, 1e-75),
        (30, 1e-100),
        (199, 1e-90),
    ]
    expected_denoised["func2"].right_best_pairings = [(30, 1e-99), (130, 1e-81)]

    strategy = DBSCANStrategy(HEMI_RIGHT)
    actual_denoised = strategy.denoise(noisy_test_hexels)

    assert (
        actual_denoised["func1"].right_best_pairings
        == expected_denoised["func1"].right_best_pairings
    )
    assert (
        actual_denoised["func2"].right_best_pairings
        == expected_denoised["func2"].right_best_pairings
    )


def test_MeanShiftStrategy_AllTrue_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["func1"].right_best_pairings = [
        (199, 1e-90),
        (-75, 1e-75),
        (30, 1e-100),
    ]
    expected_denoised["func2"].right_best_pairings = [(130, 1e-81), (30, 1e-99)]

    strategy = MeanShiftStrategy(
        HEMI_RIGHT, should_normalise=False, should_cluster_only_latency=True
    )
    actual_denoised = strategy.denoise(noisy_test_hexels)

    assert (
        actual_denoised["func1"].right_best_pairings
        == expected_denoised["func1"].right_best_pairings
    )
    assert (
        actual_denoised["func2"].right_best_pairings
        == expected_denoised["func2"].right_best_pairings
    )


def test_MeanShiftStrategy_AllFalse_Fit_Successfully():
    expected_denoised = deepcopy(noisy_test_hexels)
    expected_denoised["func1"].right_best_pairings = [
        (199, 1e-90),
        (-75, 1e-75),
        (30, 1e-100),
    ]
    expected_denoised["func2"].right_best_pairings = [(130, 1e-81), (30, 1e-99)]

    strategy = MeanShiftStrategy(HEMI_RIGHT)
    actual_denoised = strategy.denoise(noisy_test_hexels)

    assert (
        actual_denoised["func1"].right_best_pairings
        == expected_denoised["func1"].right_best_pairings
    )
    assert (
        actual_denoised["func2"].right_best_pairings
        == expected_denoised["func2"].right_best_pairings
    )
