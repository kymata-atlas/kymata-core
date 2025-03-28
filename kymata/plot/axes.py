from matplotlib import pyplot


def hide_axes(axes: pyplot.Axes):
    """Hide all axes markings from a pyplot.Axes."""
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.axis("off")
