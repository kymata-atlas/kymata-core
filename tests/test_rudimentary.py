import pytest

from kymata.entities.rudimentary import get_coerce


def test_dict_get_coerce():
    d = {"a": "123", "b": 4}

    assert get_coerce(d, key="a", default=0,  coerce=int) == 123
    assert get_coerce(d, key="b", default="", coerce=str) == "4"
    assert get_coerce(d, key="c", default=42, coerce=int) == 42

    with pytest.raises(ValueError):
        get_coerce({"x": "abc"}, "x", 0, int)
