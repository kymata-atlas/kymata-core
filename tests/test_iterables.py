import pytest

from kymata.entities.iterables import all_equal, interleave


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


def test_interleave_equal_length_two_lists():
    list_1 = [1, 2, 3, 4, 5]
    list_2 = list("abcde")
    interleaved = interleave(list_1, list_2)

    assert interleaved == [1, "a", 2, "b", 3, "c", 4, "d", 5, "e"]


def test_interleave_unequal_length_two_lists_first_shorter():
    list_1 = [1, 2, 3]
    list_2 = list("abcde")
    interleaved = interleave(list_1, list_2)

    assert interleaved == [1, "a", 2, "b", 3, "c"]


def test_interleave_unequal_length_two_lists_second_shorter():
    list_1 = [1, 2, 3, 4, 5]
    list_2 = list("abc")
    interleaved = interleave(list_1, list_2)

    assert interleaved == [1, "a", 2, "b", 3, "c"]


def test_interleave_equal_length_three_lists():
    list_1 = [1, 2, 3, 4, 5]
    list_2 = list("abcde")
    list_3 = list("ABCDE")
    interleaved = interleave(list_1, list_2, list_3)

    assert interleaved == [1, "a", "A", 2, "b", "B", 3, "c", "C", 4, "d", "D", 5, "e", "E"]
