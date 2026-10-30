# some tests written with genai
import numpy as np
import pytest
from numpy import ones

from kymata.entities.contours import ContourSet
from kymata.io.contours import save_contours, load_contours


@pytest.fixture
def example_contours():
    return ContourSet(
        {"TRANS1": ones((400,)),
         "TRANS2": 2*ones((1, 400))},
        author="Cai Wingfield <cai.wingfield@kymata.org>",
        description="Test description",
        sample_rate_hz=500,
    )


def test_io_contours_save_load_roundtrip(example_contours: ContourSet, tmp_path):

    file_path = tmp_path / "test.contours"

    save_contours(example_contours, file_path)

    loaded = load_contours(file_path)

    assert loaded.names == example_contours.names
    assert loaded.sample_rate_hz == example_contours.sample_rate_hz
    assert loaded.description == example_contours.description
    assert loaded.author == example_contours.author

    for name in example_contours.names:
        np.testing.assert_array_equal(loaded[name].values, example_contours[name].values)


def test_io_contours_replace_sample_rate(example_contours: ContourSet, tmp_path):

    file_path = tmp_path / "test.contours"

    save_contours(example_contours, file_path)

    loaded = load_contours(file_path, replace_sample_rate=10)

    assert loaded.names == example_contours.names
    assert loaded.sample_rate_hz == 10
    assert loaded.description == example_contours.description
    assert loaded.author == example_contours.author

    for name in example_contours.names:
        np.testing.assert_array_equal(loaded[name].values, example_contours[name].values)


def test_save_contours_does_not_overwrite(example_contours, tmp_path):
    """Test that existing files are not overwritten by default."""
    file_path = tmp_path / "test.mysuffix"

    save_contours(example_contours, file_path)

    with pytest.raises(FileExistsError):
        save_contours(example_contours, file_path)


def test_load_contours_npz(tmp_path):
    """Test loading a regular NPZ file without metadata."""
    file_path = tmp_path / "test.npz"

    data = {
        "contour_a": np.array([1.0, 2.0, 3.0]),
        "contour_b": np.array([4.0, 5.0, 6.0]),
    }
    np.savez_compressed(file_path, **data)

    loaded = load_contours(file_path)

    assert loaded.names == list(data.keys())

    for name, expected in data.items():
        np.testing.assert_array_equal(loaded[name].values, expected)
