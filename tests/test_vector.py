import pytest
from numpy import array, float16, errstate, array_equal

from kymata.math.vector import normalize, index_in


@pytest.fixture
def small_vector():
    return array(
        [
            0.0001237,
            0.0001237,
            0.0001237,
            0.0001237,
            0.0001237,
            0.0001237,
            0.0001237,
            0.0001237,
            0.0001236,
            0.0001237,
        ],
        dtype=float16,
    )


def test_normalize_small_vector(small_vector):
    with errstate(divide="raise"):
        normalize(small_vector)


def test_index_in_github():
    """Tests that the example from GitHub (https://stackoverflow.com/a/8251757) checks out."""
    v1 = array([3, 5, 7, 1, 9, 8, 6, 6])
    v2 = array([2, 1, 5, 10, 100, 6])
    result = index_in(v1, v2)
    expected = array([-1, 2, -1, 1, -1, -1, 5, 5])

    assert array_equal(result, expected)
