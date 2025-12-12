from kymata.math.combinatorics import generate_derangement


def test_derangement_length():
    n = 10
    derangement = generate_derangement(n)
    assert len(derangement) == n


def test_is_derangement():
    n = 10
    derangement = generate_derangement(n)
    for i in range(n):
        assert not derangement[i] == i, f"Element at index {i} is not a derangement"


def test_larger_derangement():
    n = 1000
    derangement = generate_derangement(n)
    assert len(derangement) == n
    for i in range(n):
        assert not derangement[i] == i, f"Element at index {i} is not a derangement"


def test_repeatability():
    n = 10
    derangement1 = generate_derangement(n)
    derangement2 = generate_derangement(n)
    assert len(derangement1) == len(derangement2)
    assert (
        not derangement1.tolist() == derangement2.tolist()
    ), "Derangements should not be identical"
