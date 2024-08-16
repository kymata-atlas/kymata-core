import pandas as pd

from kymata.ippm.cluster import MaxPoolClusterer, AdaptiveMaxPoolClusterer, GMMClusterer

# no data
cols = ["Latency", "Mag"]
test_case_no_data = []
test_case_no_data_df = pd.DataFrame(test_case_no_data, columns=cols)

# starting with insignificant bin
test_case_start_insignif = [
    [-110, 1e-35],
    [10, 1e-45],
    [15, 1e-80],
    [25, 1e-22],
    [36, 1e-65],
    [70, 1e-12],
    [200, 1e-99],
    [213, 1e-78],
]
test_case_start_insignif_df = pd.DataFrame(test_case_start_insignif, columns=cols)

# ending with insignificant bin
test_case_end_insignif = [
    [10, 1e-33],
    [11, 1e-45],
    [33, 1e-45],
    [47, 1e-33],
    [50, 1e-44],
    [100, 1e-34],
    [111, 1e-77],
    [125, 1e-55],
]
test_case_end_insignif_df = pd.DataFrame(test_case_end_insignif, columns=cols)

# all insignificant
test_case_all_insignif = [[10, 1e-44], [25, 1e-66], [58, 1e-94], [100, 1e-32]]
test_case_all_insignif_df = pd.DataFrame(test_case_all_insignif, columns=cols)

# significant at the start and end
test_case_start_end_signif = [
    [4, 1e-99],
    [19, 1e-32],
    [26, 1e-86],
    [42, 1e-22],
    [50, 1e-39],
    [68, 1e-67],
    [99, 1e-90],
    [100, 1e-100],
    [101, 1e-99],
    [242, 1e-32],
    [249, 1e-55],
]
test_case_start_end_signif_df = pd.DataFrame(test_case_start_end_signif, columns=cols)

# all significant bins
test_case_all_signif = [
    [11, 1e-11],
    [19, 1e-44],
    [23, 1e-50],
    [25, 1e-44],
    [28, 1e-50],
    [50, 1e-70],
    [55, 1e-44],
    [200, 1e-80],
    [210, 1e-99],
    [519, 1e-99],
    [524, 1e-42],
]
test_case_all_signif_df = pd.DataFrame(test_case_all_signif, columns=cols)


def test_MaxPooler_Fit_Successfully_When_NoData():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_case_no_data_df)
    expected_labels = []
    assert mp.labels_ == expected_labels


def test_MaxPooler_Fit_Successfully_When_StartInsignificant():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_case_start_insignif_df)
    expected_labels = [-1, 8, 8, 9, 9, -1, 16, 16]
    assert mp.labels_ == expected_labels


def test_MaxPooler_Fit_Successfully_When_EndInsignificant():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_case_end_insignif_df)
    expected_labels = [8, 8, 9, 9, -1, 12, 12, -1]
    assert mp.labels_ == expected_labels


def test_MaxPooler_Fit_Successfully_When_AllInsignificant():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_case_all_insignif_df)
    expected_labels = [-1, -1, -1, -1]
    assert mp.labels_ == expected_labels


def test_MaxPooler_Fit_Successfully_When_StartEndSignificant():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_case_start_end_signif_df)
    expected_labels = [8, 8, 9, 9, 10, 10, -1, 12, 12, 17, 17]
    assert mp.labels_ == expected_labels


def test_MaxPooler_Fit_Successfully_When_AllSignificant():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_case_all_signif_df)
    expected_labels = [8, 8, 8, 9, 9, 10, 10, 16, 16, 28, 28]
    assert mp.labels_ == expected_labels


# ///////// AdaptiveMaxPooler Integration Tests /////////////


def test_AdaptiveMaxPooler_Fit_Successfully_When_NoData():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_no_data_df)
    expected_labels = []
    assert amp.labels_ == expected_labels


def test_AdaptiveMaxPooler_Fit_Successfully_When_StartInsignificant():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_start_insignif_df)
    expected_labels = [-1, 1, 1, 1, 1, -1, 2, 2]
    assert amp.labels_ == expected_labels


def test_AdaptiveMaxPooler_Fit_Successfully_When_EndInsignificant():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_end_insignif_df)
    expected_labels = [0, 0, 0, 0, -1, 1, 1, -1]
    assert amp.labels_ == expected_labels


def test_AdaptiveMaxPooler_Fit_Successfully_When_AllInsignificant():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_all_insignif_df)
    expected_labels = [-1, -1, -1, -1]
    assert amp.labels_ == expected_labels


def test_AdaptiveMaxPooler_Fit_Successfully_When_StartEndSignificant():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_start_end_signif_df)
    expected_labels = [0, 0, 0, 0, 0, 0, -1, 1, 1, 2, 2]
    assert amp.labels_ == expected_labels


def test_AdaptiveMaxPooler_Fit_Successfully_When_AllSignificant():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_all_signif_df)
    expected_labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
    assert amp.labels_ == expected_labels


# ///////// CustomGMM Integration Tests /////////////


def test_CustomGMM_Fit_Successfully_When_NoData():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_no_data_df)
    expected_labels = []
    assert gmm.labels_ == expected_labels


def test_CustomGMM_Fit_Successfully_When_StartInsignificant():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_start_insignif_df)
    expected_labels = [2, 0, 0, 0, -1, 3, 1, 1]
    assert gmm.labels_ == expected_labels


def test_CustomGMM_Fit_Successfully_When_EndInsignificant():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_end_insignif_df)
    expected_labels = [0, 0, 3, 2, 2, 1, 1, -1]
    assert gmm.labels_ == expected_labels


def test_CustomGMM_Fit_Successfully_When_AllInsignificant():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_all_insignif_df)
    expected_labels = [3, 0, 2, 1]
    assert gmm.labels_ == expected_labels


def test_CustomGMM_Fit_Successfully_When_StartEndSignificant():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_start_end_signif_df)
    expected_labels = [0, 0, 0, 0, 0, -1, 2, 2, 2, 1, 1]
    assert gmm.labels_ == expected_labels


def test_CustomGMM_Fit_Successfully_When_AllSignificant():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_all_signif_df)
    expected_labels = [0, 0, 0, 0, 0, 0, -1, 2, 2, 1, 1]
    assert gmm.labels_ == expected_labels
