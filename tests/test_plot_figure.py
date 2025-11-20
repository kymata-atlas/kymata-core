import pytest
from matplotlib import pyplot

from kymata.plot.figure import resize_fig_keep_aspect


def test_resize_fig_keep_aspect():
    """Test that resize_fig_keep_aspect keeps the aspect ratio of a figure when resizing."""
    fig: pyplot.Figure = pyplot.figure(figsize=(6, 4))
    original_width, original_height = fig.get_size_inches()
    original_aspect = original_height / original_width

    new_width = 9
    resize_fig_keep_aspect(fig, new_width)

    width, height = fig.get_size_inches()

    assert width == new_width

    new_aspect = height / width
    assert new_aspect == pytest.approx(original_aspect, rel=1e-6)
