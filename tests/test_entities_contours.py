import pytest
from numpy import ones

from kymata.entities.contours import Contours


def test_duration_seconds():
    c = Contours(
        {"TRANS1": ones((400,)),
         "TRANS2": ones((400,))},
        sample_rate_hz=200,
    )
    assert c.duration_seconds == 2.0


def test_validation_wrong_length():
    with pytest.raises(ValueError):
        Contours(
            {"TRANS1": ones((400,)),
             "TRANS2": ones((401,))},
        )


def test_validation_not_a_vector():
    with pytest.raises(ValueError):
        Contours(
            {"TRANS1": ones((400,2)),
             "TRANS2": ones((400,2))},
        )


def test_add_contour_exists():
    c = Contours(
        {"TRANS1": ones((400,)),
         "TRANS2": ones((400,))},
    )
    with pytest.raises(ValueError):
        c.add_contour("TRANS1", ones((400,)))


def test_add_remove_contour():
    c = Contours(
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
    c = Contours(
        {"TRANS1": ones((400,)),
         "TRANS2": ones((400,))},
    )
    with pytest.raises(ValueError):
        c.add_contour("TRANS3", ones((401,)))
