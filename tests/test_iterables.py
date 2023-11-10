from kymata.entities.iterables import all_equal


def test_all_0s_are_equal():
    all_zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert all_equal(all_zeros)


def test_0s_and_a_1_are_not_equal():
    all_zeros_and_a_one = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    assert not all_equal(all_zeros_and_a_one)


def test_all_nones_are_equal():
    all_nones = [None, None, None, None, None, None, None, None]
    assert all_equal(all_nones)


def test_one_thing_is_equal():
    assert all_equal("a")


def test_nothing_is_equal():
    assert all_equal([])


def test_list_of_arrays_is_equal():
    from numpy import array
    assert all_equal([
        array([0, 0]),
        array([0, 0]),
        array([0, 0]),
        array([0, 0]),
    ])
