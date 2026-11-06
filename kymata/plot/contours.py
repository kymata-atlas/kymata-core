from typing import Optional, Callable

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from kymata.entities.contours import PredictedContourSet, PredictedContour
from kymata.plot.color import fallback_default_colors


def plot_contours(
        contours: PredictedContourSet | PredictedContour,
        time_window_s: tuple[float, float] | None = (0.0, 1.0),
        /,
        color: Optional[dict[str, str]] = None,
        figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """
    Plots a set of contours.

    Args:
        contours (PredictedContourSet): The contour(s) to plot.
        time_window_s ((float, float), optional): Time window to plot (in seconds). If None, plot the whole thing.
            Default is the first 1 second.
        color (dict[str, str], optional): Dictionary mapping contour names to colors. Defaults to random colors.
        figsize:

    Returns:
        matplotlib.pyplot.Figure
    """
    # Wrap if necessary
    if isinstance(contours, PredictedContour):
        contours = PredictedContourSet(
            contour_dict={contours.name: contours.values},
            sample_rate_hz=contours.sample_rate_hz,
            description=contours.name,
        )

    if color is None:
        color = dict()

    if time_window_s is None:
        time_window_s = (0.0, contours.duration_seconds)

    tmin_s, tmax_s = time_window_s
    if tmin_s >= tmax_s:
        raise ValueError("Invalid time range: must be strictly increasing")
    if tmax_s > contours.duration_seconds:
        raise ValueError(f"Time range exceeds contours duration ({contours.duration_seconds:.2}s)")

    tmin_idx = int(tmin_s * contours.sample_rate_hz)
    tmax_idx = int(tmax_s * contours.sample_rate_hz)

    fallback_default_colors(color, contours.names)

    n_contours = len(contours)

    fig, axs = pyplot.subplots(n_contours, 1, figsize=figsize, sharex=True)

    contour: PredictedContour
    ax: Axes
    for contour_name, ax, ax_idx in zip(contours.names, axs, range(len(contours))):
        contour = contours[contour_name]
        plot_values = contour.values[tmin_idx:tmax_idx]
        ax.plot(plot_values, color=color[contour_name])
        ax.xaxis.set_major_formatter(FuncFormatter(_formatter_miliseconds(contours.sample_rate_hz, tmin_idx)))

        ax.set_title(contour_name, fontsize="small", loc="left")

        # Add x-axis label to bottom-most axis
        if ax_idx == len(contours) - 1:
            ax.set_xlabel("Time")

    fig.suptitle(f"{contours.description or 'Contours'}")
    fig.tight_layout()

    return fig


def _formatter_miliseconds(sample_rate: float, tmin_samples: int) -> Callable[[float, float], str]:
    """Produces an axis formatter which converts samples into formatted time."""
    def inner_formatter(sample, pos):
        hours: int = 0
        mins: int = 0
        secs: float = (tmin_samples + sample) / sample_rate
        if secs >= 60:
            mins = int(secs / 60)
            secs %= 60
        if mins >= 60:
            hours = mins // 60
            mins %= 60

        string = f"{hours}:" if hours > 0 else ""
        if mins > 0:
            string += f"{mins:02}:"
        string += f"{secs:02.3}"

        return string
    return inner_formatter
