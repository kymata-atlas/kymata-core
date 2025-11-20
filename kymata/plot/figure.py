from matplotlib import pyplot

from kymata.entities.expression import ExpressionSet


def hide_axes(axes: pyplot.Axes):
    """Hide all axes markings from a pyplot.Axes."""
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.axis("off")


def xlims_from_expressionset(es: ExpressionSet, padding: float = 0.05) -> tuple[float, float]:
    """
    Get an appropriate set of xlims from an ExpressionSet.

    Args:
        es (ExpressionSet):
        padding (float): The amount of padding to add either side of the IPPM plot, in seconds. Default is 0.05 (50ms).

    Returns:
        tuple[float, float]: xmin, xmax
    """
    return (
        es.latencies.min() - padding,
        es.latencies.max() + padding,
    )


def resize_fig_keep_aspect(fig: pyplot.Figure, width: float) -> None:
    """
    Resize a figure to the specified width, but maintain the aspect ratio.

    Args:
        fig (matplotlib.pyplot.Figure):
        width (float):
    """
    old_width, old_height = fig.get_size_inches()
    aspect = old_height / old_width
    fig.set_size_inches(width, width * aspect)
