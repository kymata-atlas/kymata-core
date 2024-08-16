import pytest

from kymata.entities.iterables import all_equal


@pytest.fixture
def all_0s():
    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


@pytest.fixture
def all_0s_and_a_1():
    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]


@pytest.fixture
def all_nones():
    return [None, None, None, None, None, None, None, None]


def test_all_0s_are_equal(all_0s):
    assert all_equal(all_0s)


def test_0s_and_a_1_are_not_equal(all_0s_and_a_1):
    assert not all_equal(all_0s_and_a_1)


def test_all_nones_are_equal(all_nones):
    assert all_equal(all_nones)


def test_one_thing_is_equal():
    assert all_equal("a")


def test_nothing_is_equal():
    assert all_equal([])


def test_list_of_arrays_is_equal():
    from numpy import array

    assert all_equal(
        [
            array([0, 0]),
            array([0, 0]),
            array([0, 0]),
            array([0, 0]),
        ]
    )
