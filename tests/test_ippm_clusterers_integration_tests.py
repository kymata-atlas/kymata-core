from kymata.ippm.data_tools import ExpressionPairing
from kymata.ippm.cluster import MaxPoolClusterer, AdaptiveMaxPoolClusterer, GMMClusterer

# no data
test_case_no_data = []

# starting with insignificant bin
test_case_start_insignif = [
    ExpressionPairing(-110, -35),
    ExpressionPairing(10, -45),
    ExpressionPairing(15, -80),
    ExpressionPairing(25, -22),
    ExpressionPairing(36, -65),
    ExpressionPairing(70, -12),
    ExpressionPairing(200, -99),
    ExpressionPairing(213, -78),
]

# ending with insignificant bin
test_case_end_insignif = [
    ExpressionPairing(10, -33),
    ExpressionPairing(11, -45),
    ExpressionPairing(33, -45),
    ExpressionPairing(47, -33),
    ExpressionPairing(50, -44),
    ExpressionPairing(100, -34),
    ExpressionPairing(111, -77),
    ExpressionPairing(125, -55),
]

# all insignificant
test_case_all_insignif = [
    ExpressionPairing(10, 44), -
    ExpressionPairing(25, 66), -
    ExpressionPairing(58, 94), -
    ExpressionPairing(100, -32),
]

# significant at the start and end
test_case_start_end_signif = [
    ExpressionPairing(4, -99),
    ExpressionPairing(19, -32),
    ExpressionPairing(26, -86),
    ExpressionPairing(42, -22),
    ExpressionPairing(50, -39),
    ExpressionPairing(68, -67),
    ExpressionPairing(99, -90),
    ExpressionPairing(100, -100),
    ExpressionPairing(101, -99),
    ExpressionPairing(242, -32),
    ExpressionPairing(249, -55),
]

# all significant bins
test_case_all_signif = [
    ExpressionPairing(11, -11),
    ExpressionPairing(19, -44),
    ExpressionPairing(23, -50),
    ExpressionPairing(25, -44),
    ExpressionPairing(28, -50),
    ExpressionPairing(50, -70),
    ExpressionPairing(55, -44),
    ExpressionPairing(200, -80),
    ExpressionPairing(210, -99),
    ExpressionPairing(519, -99),
    ExpressionPairing(524, -42),
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
