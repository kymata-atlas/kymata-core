import pytest
from numpy import ones

from kymata.entities.contours import Contours


def test_validation_wrong_length():
    with pytest.raises(ValueError):
        Contours(
            {"TRANS1": ones((400,)),
             "TRANS2": ones((401,))},
        )


def test_add_contour_exists():
    c = Contours(
        {"TRANS1": ones((400,)),
         "TRANS2": ones((400,))},
    )
    with pytest.raises(ValueError):
        c.add_contour("TRANS1", ones((400,)))


def test_add_contour_wrong_length():
    c = Contours(
        {"TRANS1": ones((400,)),
         "TRANS2": ones((400,))},
    )
    with pytest.raises(ValueError):
        c.add_contour("TRANS3", ones((401,)))
