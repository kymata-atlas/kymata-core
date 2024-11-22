from kymata.entities.expression import ExpressionPoint
from kymata.ippm.cluster import MaxPoolClusterer, AdaptiveMaxPoolClusterer, GMMClusterer

# no data
test_case_no_data = []

# starting with insignificant bin
test_case_start_insignif = [
    ExpressionPoint("c", -110, "f", -35),
    ExpressionPoint("c", 10, "f", -45),
    ExpressionPoint("c", 15, "f", -80),
    ExpressionPoint("c", 25, "f", -22),
    ExpressionPoint("c", 36, "f", -65),
    ExpressionPoint("c", 70, "f", -12),
    ExpressionPoint("c", 200, "f", -99),
    ExpressionPoint("c", 213, "f", -78),
]

# ending with insignificant bin
test_case_end_insignif = [
    ExpressionPoint("c", 10, "f", -33),
    ExpressionPoint("c", 11, "f", -45),
    ExpressionPoint("c", 33, "f", -45),
    ExpressionPoint("c", 47, "f", -33),
    ExpressionPoint("c", 50, "f", -44),
    ExpressionPoint("c", 100, "f", -34),
    ExpressionPoint("c", 111, "f", -77),
    ExpressionPoint("c", 125, "f", -55),
]

# all insignificant
test_case_all_insignif = [
    ExpressionPoint("c", 10, "f", -44),
    ExpressionPoint("c", 25, "f", -66),
    ExpressionPoint("c", 58, "f", -94),
    ExpressionPoint("c", 100, "f", -32),
]

# significant at the start and end
test_case_start_end_signif = [
    ExpressionPoint("c", 4, "f", -99),
    ExpressionPoint("c", 19, "f", -32),
    ExpressionPoint("c", 26, "f", -86),
    ExpressionPoint("c", 42, "f", -22),
    ExpressionPoint("c", 50, "f", -39),
    ExpressionPoint("c", 68, "f", -67),
    ExpressionPoint("c", 99, "f", -90),
    ExpressionPoint("c", 100, "f", -100),
    ExpressionPoint("c", 101, "f", -99),
    ExpressionPoint("c", 242, "f", -32),
    ExpressionPoint("c", 249, "f", -55),
]

# all significant bins
test_case_all_signif = [
    ExpressionPoint("c", 11, "f", -11),
    ExpressionPoint("c", 19, "f", -44),
    ExpressionPoint("c", 23, "f", -50),
    ExpressionPoint("c", 25, "f", -44),
    ExpressionPoint("c", 28, "f", -50),
    ExpressionPoint("c", 50, "f", -70),
    ExpressionPoint("c", 55, "f", -44),
    ExpressionPoint("c", 200, "f", -80),
    ExpressionPoint("c", 210, "f", -99),
    ExpressionPoint("c", 519, "f", -99),
    ExpressionPoint("c", 524, "f", -42),
]


def test_MaxPooler_Fit_Successfully_When_NoData():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_case_no_data)
    expected_labels = []
    assert mp.labels_ == expected_labels


def test_MaxPooler_Fit_Successfully_When_StartInsignificant():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp.fit(test_case_start_insignif)
    expected_labels = [-1, 8, 8, 9, 9, -1, 16, 16]
    assert mp.labels_ == expected_labels


def test_MaxPooler_Fit_Successfully_When_EndInsignificant():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_case_end_insignif)
    expected_labels = [8, 8, 9, 9, -1, 12, 12, -1]
    assert mp.labels_ == expected_labels


def test_MaxPooler_Fit_Successfully_When_AllInsignificant():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_case_all_insignif)
    expected_labels = [-1, -1, -1, -1]
    assert mp.labels_ == expected_labels


def test_MaxPooler_Fit_Successfully_When_StartEndSignificant():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_case_start_end_signif)
    expected_labels = [8, 8, 9, 9, 10, 10, -1, 12, 12, 17, 17]
    assert mp.labels_ == expected_labels


def test_MaxPooler_Fit_Successfully_When_AllSignificant():
    mp = MaxPoolClusterer(label_significance_threshold=2)
    mp = mp.fit(test_case_all_signif)
    expected_labels = [8, 8, 8, 9, 9, 10, 10, 16, 16, 28, 28]
    assert mp.labels_ == expected_labels


# ///////// AdaptiveMaxPooler Integration Tests /////////////


def test_AdaptiveMaxPooler_Fit_Successfully_When_NoData():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_no_data)
    expected_labels = []
    assert amp.labels_ == expected_labels


def test_AdaptiveMaxPooler_Fit_Successfully_When_StartInsignificant():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_start_insignif)
    expected_labels = [-1, 1, 1, 1, 1, -1, 2, 2]
    assert amp.labels_ == expected_labels


def test_AdaptiveMaxPooler_Fit_Successfully_When_EndInsignificant():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_end_insignif)
    expected_labels = [0, 0, 0, 0, -1, 1, 1, -1]
    assert amp.labels_ == expected_labels


def test_AdaptiveMaxPooler_Fit_Successfully_When_AllInsignificant():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_all_insignif)
    expected_labels = [-1, -1, -1, -1]
    assert amp.labels_ == expected_labels


def test_AdaptiveMaxPooler_Fit_Successfully_When_StartEndSignificant():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_start_end_signif)
    expected_labels = [0, 0, 0, 0, 0, 0, -1, 1, 1, 2, 2]
    assert amp.labels_ == expected_labels


def test_AdaptiveMaxPooler_Fit_Successfully_When_AllSignificant():
    amp = AdaptiveMaxPoolClusterer(label_significance_threshold=2, base_label_size=25)
    amp = amp.fit(test_case_all_signif)
    expected_labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
    assert amp.labels_ == expected_labels


# ///////// CustomGMM Integration Tests /////////////


def test_CustomGMM_Fit_Successfully_When_NoData():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_no_data)
    expected_labels = []
    assert gmm.labels_ == expected_labels


def test_CustomGMM_Fit_Successfully_When_StartInsignificant():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_start_insignif)
    expected_labels = [2, 0, 0, 0, 0, 3, 1, 1]
    assert list(gmm.labels_) == expected_labels


def test_CustomGMM_Fit_Successfully_When_EndInsignificant():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_end_insignif)
    expected_labels = [0, 0, 3, 2, 2, 1, 1, 1]
    assert list(gmm.labels_) == expected_labels


def test_CustomGMM_Fit_Successfully_When_AllInsignificant():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_all_insignif)
    expected_labels = [3, 0, 2, 1]
    assert list(gmm.labels_) == expected_labels


def test_CustomGMM_Fit_Successfully_When_StartEndSignificant():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_start_end_signif)
    expected_labels = [0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1]
    assert list(gmm.labels_) == expected_labels


def test_CustomGMM_Fit_Successfully_When_AllSignificant():
    random_seed = 40
    gmm = GMMClusterer(random_state=random_seed)
    gmm = gmm.fit(test_case_all_signif)
    expected_labels = [0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1]
    assert list(gmm.labels_) == expected_labels
