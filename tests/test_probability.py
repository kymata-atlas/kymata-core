from math import isclose

from kymata.math.probability import bonferroni_correct, sidak_correct


def test_bonferroni_1_comparison():
    original_alpha = 0.05
    corrected_alpha = bonferroni_correct(alpha_p=original_alpha, n_comparisons=1)
    assert isclose(corrected_alpha, 0.05)


def test_bonferroni_2_comparisons():
    original_alpha = 0.05
    corrected_alpha = bonferroni_correct(alpha_p=original_alpha, n_comparisons=2)
    assert isclose(corrected_alpha, 0.025)


def test_bonferroni_10_comparisons():
    original_alpha = 0.05
    corrected_alpha = bonferroni_correct(alpha_p=original_alpha, n_comparisons=10)
    assert isclose(corrected_alpha, 0.005)


def test_bonferroni_500_comparisons():
    original_alpha = 0.05
    corrected_alpha = bonferroni_correct(alpha_p=original_alpha, n_comparisons=500)
    assert isclose(corrected_alpha, 0.0001)


def test_sidak_1_comparison():
    original_alpha = 0.05
    corrected_alpha = sidak_correct(alpha_p=original_alpha, n_comparisons=1)
    assert isclose(corrected_alpha, 0.05)


def test_sidak_2_comparisons():
    original_alpha = 0.05
    corrected_alpha = sidak_correct(alpha_p=original_alpha, n_comparisons=2)
    assert isclose(corrected_alpha, 0.0253206, abs_tol=0.0000001)


def test_sidak_10_comparisons():
    original_alpha = 0.05
    corrected_alpha = sidak_correct(alpha_p=original_alpha, n_comparisons=10)
    assert isclose(corrected_alpha, 0.0051162, abs_tol=0.0000001)


def test_sidak_500_comparisons():
    original_alpha = 0.05
    corrected_alpha = sidak_correct(alpha_p=original_alpha, n_comparisons=500)
    assert isclose(corrected_alpha, 0.0001026, abs_tol=0.0000001)
