from numpy import array, array_equal

from kymata.entities.functions import Function


def test_resample_to_same_rate():
    f = Function("test", array(range(100)), sample_rate=1000)
    f_resampled = f.resampled(f.sample_rate)
    assert f.sample_rate == f_resampled.sample_rate
    assert len(f.values) == len(f_resampled.values)


def test_integer_subsample():
    f = Function("test", array(range(100)), sample_rate=1000)
    new_sample_rate = 500
    downsample_ratio = f.sample_rate / new_sample_rate
    f_resampled = f.resampled(new_sample_rate)
    assert f_resampled.sample_rate == new_sample_rate
    assert len(f.values) / downsample_ratio == len(f_resampled.values)


def test_noninteger_downsample():
    from math import floor

    f = Function("test", array(range(10)), 5)
    f_downsampled = f.resampled(2)
    ratio = f_downsampled.sample_rate / f.sample_rate
    assert len(f_downsampled.values) == floor(len(f.values) * ratio)
    assert array_equal(f_downsampled.values, array([0, 2, 5, 7]))


def test_upsample_integer_ratio():
    f = Function("test", array(range(5)), sample_rate=1)
    f_upsampled = f.resampled(2)
    ratio = f_upsampled.sample_rate / f.sample_rate
    assert ratio.is_integer()
    assert len(f.values) * ratio == len(f_upsampled.values)
    assert array_equal(f_upsampled.values, array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]))


def test_upsample_noninteger_ratio():
    from math import floor

    f = Function("test", array(range(5)), sample_rate=2)
    f_upsampled = f.resampled(3)
    ratio = f_upsampled.sample_rate / f.sample_rate
    assert len(f_upsampled.values) == floor(len(f.values) * ratio)
    assert array_equal(f_upsampled.values, array([0, 0, 1, 2, 2, 3, 4]))
