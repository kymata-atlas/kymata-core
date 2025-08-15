from numpy import array, array_equal

from kymata.entities.transform import Transform, shift_by


def test_resample_to_same_rate():
    t = Transform("test", array(range(100)), sample_rate=1000)
    t_resampled = t.resampled(t.sample_rate)
    assert t.sample_rate == t_resampled.sample_rate
    assert len(t.values) == len(t_resampled.values)


def test_integer_subsample():
    t = Transform("test", array(range(100)), sample_rate=1000)
    new_sample_rate = 500
    downsample_ratio = t.sample_rate / new_sample_rate
    t_resampled = t.resampled(new_sample_rate)
    assert t_resampled.sample_rate == new_sample_rate
    assert len(t.values) / downsample_ratio == len(t_resampled.values)
    assert array_equal(t_resampled.values, array(range(0, 100, 2)))


def test_noninteger_downsample():
    from math import floor

    t = Transform("test", array(range(10)), 5)
    t_downsampled = t.resampled(2)
    ratio = t_downsampled.sample_rate / t.sample_rate
    assert len(t_downsampled.values) == floor(len(t.values) * ratio)
    assert array_equal(t_downsampled.values, array([0, 2, 5, 7]))


def test_upsample_integer_ratio():
    t = Transform("test", array(range(5)), sample_rate=1)
    t_upsampled = t.resampled(2)
    ratio = t_upsampled.sample_rate / t.sample_rate
    assert ratio.is_integer()
    assert len(t.values) * ratio == len(t_upsampled.values)
    assert array_equal(t_upsampled.values, array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]))


def test_upsample_noninteger_ratio():
    from math import floor

    t = Transform("test", array(range(5)), sample_rate=2)
    t_upsampled = t.resampled(3)
    ratio = t_upsampled.sample_rate / t.sample_rate
    assert len(t_upsampled.values) == floor(len(t.values) * ratio)
    assert array_equal(t_upsampled.values, array([0, 0, 1, 2, 2, 3, 4]))


def test_shift_by_0():
    arr = array([1, 2, 3, 4, 5, 6, 7, 8])
    assert array_equal(shift_by(arr, 4), array([0, 0, 0, 0, 1, 2, 3, 4]))


def test_shift_by_positive():
    arr = array([1, 2, 3, 4, 5, 6, 7, 8])
    assert array_equal(shift_by(arr, 0), arr)


def test_shift_by_negative():
    arr = array([1, 2, 3, 4, 5, 6, 7, 8])
    assert array_equal(shift_by(arr, -4), array([5, 6, 7, 8, 0, 0, 0, 0]))
