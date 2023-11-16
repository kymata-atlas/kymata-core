import pytest
from numpy.random import rand
from sparse import COO
from numpy import array_equal

from kymata.entities.sparse_data import expand_dims


def test_expand_dims_in_middle():

    matrix = COO.from_numpy(rand(3, 3))

    expanded_matrix = expand_dims(matrix, 1)

    assert expanded_matrix.shape == (3, 1, 3)
    assert array_equal(
        matrix.todense(),
        expanded_matrix.todense().squeeze()
    )


def test_expand_dims_at_end():

    matrix = COO.from_numpy(rand(3, 3))

    expanded_matrix = expand_dims(matrix, -1)

    assert expanded_matrix.shape == (3, 3, 1)
    assert array_equal(
        matrix.todense(),
        expanded_matrix.todense().squeeze()
    )


def test_expand_dims_not_quite_too_big():

    matrix = COO.from_numpy(rand(3, 3, 1))

    expanded_matrix = expand_dims(matrix, 2)

    assert expanded_matrix.shape == (3, 3, 1)
    assert array_equal(
        matrix.todense(),
        expanded_matrix.todense()
    )


def test_expand_dims_too_big():

    matrix = COO.from_numpy(rand(3, 3, 3))

    with pytest.raises(ValueError):
        expanded_matrix = expand_dims(matrix, -1)


def test_expand_dims_much_too_big():

    matrix = COO.from_numpy(rand(3, 4, 5, 6, 7))

    with pytest.raises(ValueError):
        expanded_matrix = expand_dims(matrix, -1)
