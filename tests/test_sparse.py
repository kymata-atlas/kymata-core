import pytest
from numpy.random import rand
from sparse import COO
from numpy import array_equal

from kymata.entities.sparse_data import expand_dims, all_nonfill_close


def test_expand_dims_in_middle():
    matrix = COO.from_numpy(rand(3, 3))

    expanded_matrix = expand_dims(matrix, 1)

    assert expanded_matrix.shape == (3, 1, 3)
    assert array_equal(matrix.todense(), expanded_matrix.todense().squeeze())


def test_expand_dims_at_end():
    matrix = COO.from_numpy(rand(3, 3))

    expanded_matrix = expand_dims(matrix, -1)

    assert expanded_matrix.shape == (3, 3, 1)
    assert array_equal(matrix.todense(), expanded_matrix.todense().squeeze())


def test_expand_dims_not_quite_too_big():
    matrix = COO.from_numpy(rand(3, 3, 1))

    expanded_matrix = expand_dims(matrix, 2)

    assert expanded_matrix.shape == (3, 3, 1)
    assert array_equal(matrix.todense(), expanded_matrix.todense())


def test_expand_dims_too_big():
    matrix = COO.from_numpy(rand(3, 3, 3))

    with pytest.raises(ValueError):
        expand_dims(matrix, -1)


def test_expand_dims_much_too_big():
    matrix = COO.from_numpy(rand(3, 4, 5, 6, 7))

    with pytest.raises(ValueError):
        expand_dims(matrix, -1)


def test_all_nonfill_close_mismatched_coords():
    coords_left = [
        [0, 0, 0, 1, 1],
        [0, 1, 2, 0, 3],
        [0, 3, 2, 0, 1],
    ]
    coords_right = [
        [0, 0, 0, 1, 2],
        [0, 1, 2, 0, 3],
        [0, 3, 2, 0, 1],
    ]
    data = [1, 2, 3, 4, 5]
    shape = (3, 4, 5)
    left = COO(coords_left, data, shape)
    right = COO(coords_right, data, shape)

    assert not all_nonfill_close(left, right)


def test_all_nonfill_close_unequal_data():
    coords = [
        [0, 0, 0, 1, 1],
        [0, 1, 2, 0, 3],
        [0, 3, 2, 0, 1],
    ]
    data_left = [1.0, 2.0, 3.0, 4.0, 5.0]
    data_right = [1.0, 2.0, 3.0, 4.0, 5.0001]  # close but not close enough
    shape = (3, 4, 5)
    left = COO(coords, data_left, shape)
    right = COO(coords, data_right, shape)

    assert not all_nonfill_close(left, right)


def test_all_nonfill_close_close_data():
    coords = [
        [0, 0, 0, 1, 1],
        [0, 1, 2, 0, 3],
        [0, 3, 2, 0, 1],
    ]
    data_left = [1.0, 2.0, 3.0, 4.0, 5.0]
    data_right = [1.0, 2.0, 3.0, 4.0, 5.000000000000000001]  # close enough
    shape = (3, 4, 5)
    left = COO(coords, data_left, shape)
    right = COO(coords, data_right, shape)

    assert not all_nonfill_close(left, right)


def test_all_nonfill_close_equal():
    coords = [
        [0, 0, 0, 1, 1],
        [0, 1, 2, 0, 3],
        [0, 3, 2, 0, 1],
    ]
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    shape = (3, 4, 5)
    from copy import copy
    left = COO(copy(coords), copy(data), shape)
    right = COO(copy(coords), copy(data), shape)

    assert all_nonfill_close(left, right)
