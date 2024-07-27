import pytest
from numpy import array, float16, errstate

from kymata.math.vector import normalize


@pytest.fixture
def small_vector():
    return array([0.0001237, 0.0001237, 0.0001237, 0.0001237, 0.0001237, 0.0001237,
                  0.0001237, 0.0001237, 0.0001237, 0.0001237], dtype=float16)


def test_normalize_small_vector(small_vector):
    with errstate(divide='raise'):
        normalize(small_vector)
