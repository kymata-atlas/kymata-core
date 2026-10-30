import pytest
from numpy import ones, array, array_equal, zeros_like, isclose

from kymata.entities.contours import Contour, ContourSet, shift_by



def test_resample_to_same_rate():
    t = Contour("test", array(range(100)), sample_rate_hz=1000)
    t_resampled = t.resampled(t.sample_rate_hz)
    assert t.sample_rate_hz == t_resampled.sample_rate_hz
    assert len(t.values) == len(t_resampled.values)


def test_integer_subsample():
    t = Contour("test", array(range(100)), sample_rate_hz=1000)
    new_sample_rate = 500
    downsample_ratio = t.sample_rate_hz / new_sample_rate
    t_resampled = t.resampled(new_sample_rate)
    assert t_resampled.sample_rate_hz == new_sample_rate
    assert len(t.values) / downsample_ratio == len(t_resampled.values)
    assert array_equal(t_resampled.values, array(range(0, 100, 2)))


def test_noninteger_downsample():
    from math import floor

    t = Contour("test", array(range(10)), 5)
    t_downsampled = t.resampled(2)
    ratio = t_downsampled.sample_rate_hz / t.sample_rate_hz
    assert len(t_downsampled.values) == floor(len(t.values) * ratio)
    assert array_equal(t_downsampled.values, array([0, 2, 5, 7]))


def test_timestep():
    t = Contour("test", array(range(10)), 500)
    assert isclose(t.time_step, 0.002)



def test_upsample_integer_ratio():
    t = Contour("test", array(range(5)), sample_rate_hz=1)
    t_upsampled = t.resampled(2)
    ratio = t_upsampled.sample_rate_hz / t.sample_rate_hz
    assert ratio.is_integer()
    assert len(t.values) * ratio == len(t_upsampled.values)
    assert array_equal(t_upsampled.values, array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]))


def test_upsample_noninteger_ratio():
    from math import floor

    t = Contour("test", array(range(5)), sample_rate_hz=2)
    t_upsampled = t.resampled(3)
    ratio = t_upsampled.sample_rate_hz / t.sample_rate_hz
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


def test_shift_by_too_much():
    arr = array([1, 2, 3, 4, 5, 6, 7, 8])
    assert array_equal(shift_by(arr, 100), zeros_like(arr))


def test_duration_seconds():
    c = ContourSet(
        {"TRANS1": ones((400,)),
         "TRANS2": ones((400,))},
        sample_rate_hz=200,
    )
    assert c.duration_seconds == 2.0


def test_validation_wrong_length():
    with pytest.raises(ValueError):
        ContourSet(
            {"TRANS1": ones((400,)),
             "TRANS2": ones((401,))},
        )


def test_validation_not_a_vector():
    with pytest.raises(ValueError):
        ContourSet(
            {"TRANS1": ones((400,2)),
             "TRANS2": ones((400,2))},
        )


def test_add_contour_exists():
    c = ContourSet(
        {"TRANS1": ones((400,)),
         "TRANS2": ones((400,))},
    )
    with pytest.raises(ValueError):
        c.add_contour("TRANS1", ones((400,)))


def test_add_remove_contour():
    c = ContourSet(
        {"TRANS1": ones((400,)),
         "TRANS2": ones((400,))},
    )
    c.add_contour("TRANS3", ones((400,)))

    assert "TRANS1" in c.names
    assert "TRANS2" in c.names
    assert "TRANS3" in c.names

    c.remove_contour("TRANS2")

    assert "TRANS1" in c.names
    assert "TRANS2" not in c.names
    assert "TRANS3" in c.names


def test_add_contour_wrong_length():
    c = ContourSet(
        {"TRANS1": ones((400,)),
         "TRANS2": ones((400,))},
    )
    with pytest.raises(ValueError):
        c.add_contour("TRANS3", ones((401,)))
