import pytest
from matplotlib import pyplot

from kymata.plot.figure import resize_fig_keep_aspect


def test_resize_fig_keep_aspect():
    # Create a test figure
    fig: pyplot.Figure = pyplot.figure(figsize=(6, 4))  # width=6, height=4
    original_width, original_height = fig.get_size_inches()
    original_aspect = original_height / original_width

    # Apply resizing
    new_width = 9
    resize_fig_keep_aspect(fig, new_width)

    # Get new size
    width, height = fig.get_size_inches()

    # Check width was updated
    assert width == new_width

    # Check aspect ratio preserved (float comparison)
    new_aspect = height / width
    assert new_aspect == pytest.approx(original_aspect, rel=1e-6)
