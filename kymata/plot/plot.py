from itertools import cycle
from pathlib import Path
from statistics import NormalDist
from typing import Optional, Sequence, Dict, NamedTuple

import numpy as np
from matplotlib import pyplot, colors
from matplotlib.lines import Line2D
from pandas import DataFrame
from seaborn import color_palette

from kymata.entities.expression import HexelExpressionSet, ExpressionSet, SensorExpressionSet, DIM_SENSOR, DIM_FUNCTION, \
    p_to_logp
from kymata.plot.layouts import get_meg_sensor_xy, eeg_sensors

# log scale: 10 ** -this will be the ytick interval and also the resolution to which the ylims will be rounded
_MAJOR_TICK_SIZE = 50


def _plot_function_expression_on_axes(ax: pyplot.Axes, function_data: DataFrame, color, sidak_corrected_alpha: float, filled: bool):
    """
    Returns:
        x_min, x_max, y_min, y_max
            logp values for axis limits
            Note: *_min and *_max values are np.Inf and -np.Inf respectively if x or y is empty
                  (so they can be added to min() and max() without altering the result).
    """
    x = function_data["latency"].values * 1000  # Convert to milliseconds
    y = function_data["value"].values
    c = np.where(np.array(y) <= sidak_corrected_alpha, color, "black")
    ax.vlines(x=x, ymin=1, ymax=y, color=c)
    ax.scatter(x, y, facecolors=c if filled else 'none', s=20, edgecolors=c)

    x_min = x.min() if len(x) > 0 else np.Inf
    x_max = x.max() if len(x) > 0 else -np.Inf
    # Careful, the y value is inverted, with y==1 on the origin and y<1 away from the origin.
    # "y_min" here is real absolute min value in the data (closest to zero)
    y_min = y.min() if len(y) > 0 else np.Inf
    y_max = y.max() if len(y) > 0 else -np.Inf

    return x_min, x_max, y_min, y_max


class AxisAssignment(NamedTuple):
    axis_name: str
    axis_channels: list


_left_right_sensors: tuple[AxisAssignment, AxisAssignment] = (
    AxisAssignment(axis_name="left",
                   axis_channels=[
                       sensor
                       for sensor, (x, y) in get_meg_sensor_xy().items()
                       if x <= 0
                   ] + eeg_sensors()),  # TODO: these EEGs aren't split appropriately
    AxisAssignment(axis_name="right",
                   axis_channels=[
                       sensor
                       for sensor, (x, y) in get_meg_sensor_xy().items()
                       if x >= 0
                   ] + eeg_sensors()),
)


def expression_plot(
        expression_set: ExpressionSet,
        show_only: Optional[str | Sequence[str]] = None,
        paired_axes: bool = True,
        # Statistical kwargs
        alpha: float = 1 - NormalDist(mu=0, sigma=1).cdf(5),  # 5-sigma
        # Style kwargs
        color: Optional[str | Dict[str, str] | list[str]] = None,
        ylim: Optional[float] = None,
        xlims: Optional[tuple[Optional[float], Optional[float]]] = None,
        hidden_functions_in_legend: bool = True,
        # I/O args
        save_to: Optional[Path] = None,
        overwrite: bool = True,
):
    """
    Generates an expression plot

    paired_axes: When True, show the expression plot split left/right.
                 When False, all points shown on the same axis.

    color: colour name, function_name → colour name, or list of colour names
    xlims: None or tuple. None to use default values, or either entry of the tuple as None to use default for that value.
    """

    # Default arg values
    if show_only is None:
        # Plot all
        show_only = expression_set.functions
    elif isinstance(show_only, str):
        show_only = [show_only]
    not_shown = [f for f in expression_set.functions if f not in show_only]

    if color is None:
        color = dict()
    elif isinstance(color, str):
        # Single string specified: use all that colour
        color = {f: color for f in show_only}
    elif isinstance(color, str):
        # List specified, then pair up in order
        assert len(color) == len(str)
        color = {f: c for f, c in zip(show_only, color)}

    # Default colours
    cycol = cycle(color_palette("Set1"))
    for function in show_only:
        if function not in color:
            color[function] = colors.to_hex(next(cycol))

    if isinstance(expression_set, HexelExpressionSet):
        channels = expression_set.hexels
    elif isinstance(expression_set, SensorExpressionSet):
        channels = expression_set.sensors
    else:
        raise NotImplementedError()

    best_functions = expression_set.best_functions()

    if paired_axes:
        if isinstance(expression_set, HexelExpressionSet):
            axes_names = ("left hemisphere", "right hemisphere")
            assert isinstance(best_functions, tuple)
        elif isinstance(expression_set, SensorExpressionSet):
            axes_names = ("left", "right")
            # Same functions passed, filtering done at channel level
            best_functions = (best_functions, best_functions)
        else:
            raise NotImplementedError()
    else:
        if isinstance(expression_set, HexelExpressionSet):
            # TODO: if we ever want to, we could implement collapsing sensor expression plots
            raise NotImplementedError("HexelExpressionSets have preset hemisphere assignments")
        elif isinstance(expression_set, SensorExpressionSet):
            axes_names = ("", )
            # Wrap into list
            best_functions = (best_functions, )
        else:
            raise NotImplementedError()

    sidak_corrected_alpha = 1 - (
        (1 - alpha)
        ** (1 / (2
                 * len(expression_set.latencies)
                 * len(channels)
                 * len(show_only))))

    sidak_corrected_alpha = p_to_logp(sidak_corrected_alpha)

    fig, axes = pyplot.subplots(nrows=2 if paired_axes else 1, ncols=1, figsize=(12, 7))
    if isinstance(axes, pyplot.Axes): axes = (axes, )  # Wrap if necessary
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(right=0.84, left=0.08)

    custom_handles = []
    custom_labels = []
    data_x_min, data_x_max = np.Inf, -np.Inf
    data_y_min             = np.Inf
    for function in show_only:

        custom_handles.extend([Line2D([], [], marker='.', color=color[function], linestyle='None')])
        custom_labels.append(function)

        # We have a special case with paired sensor data, in that some sensors need to appear
        # on both sides of the midline.
        if paired_axes and isinstance(expression_set, SensorExpressionSet):
            assign_left_right_channels = _left_right_sensors
            # Some points will be plotted on one axis, filled, some on both, empty
            top_chans = set(assign_left_right_channels[0].axis_channels)
            bottom_chans = set(assign_left_right_channels[1].axis_channels)
            # Symmetric difference
            both_chans = top_chans & bottom_chans
            top_chans -= both_chans
            bottom_chans -= both_chans
            chans = (top_chans, bottom_chans)
            for ax, best_funs_this_ax, chans_this_ax in zip(axes, best_functions, chans):
                # Plot filled
                x_min, x_max, y_min, _y_max, = _plot_function_expression_on_axes(
                    function_data=best_funs_this_ax[(best_funs_this_ax[DIM_FUNCTION] == function)
                                                    & (best_funs_this_ax[DIM_SENSOR].isin(chans_this_ax))],
                    color=color[function],
                    ax=ax, sidak_corrected_alpha=sidak_corrected_alpha, filled=True)
                data_x_min = min(data_x_min, x_min)
                data_x_max = max(data_x_max, x_max)
                data_y_min = min(data_y_min, y_min)
                # Plot empty
                x_min, x_max, y_min, _y_max, = _plot_function_expression_on_axes(
                    function_data=best_funs_this_ax[(best_funs_this_ax[DIM_FUNCTION] == function)
                                                    & (best_funs_this_ax[DIM_SENSOR].isin(both_chans))],
                    color=color[function],
                    ax=ax, sidak_corrected_alpha=sidak_corrected_alpha, filled=False)
                data_x_min = min(data_x_min, x_min)
                data_x_max = max(data_x_max, x_max)
                data_y_min = min(data_y_min, y_min)

        # With non-sensor data, or non-paired axes, we can treat these cases together
        else:
            # As normal, plot appropriate filled points in each axis
            for ax, best_funs_this_ax in zip(axes, best_functions):
                x_min, x_max, y_min, _y_max, = _plot_function_expression_on_axes(
                    function_data=best_funs_this_ax[best_funs_this_ax[DIM_FUNCTION] == function],
                    color=color[function],
                    ax=ax, sidak_corrected_alpha=sidak_corrected_alpha, filled=True)
                data_x_min = min(data_x_min, x_min)
                data_x_max = max(data_x_max, x_max)
                data_y_min = min(data_y_min, y_min)

    # format shared axis qualities
    for ax in axes:
        xlims = _get_best_xlims(xlims, data_x_min, data_x_max)
        ylim = _get_best_ylim(ylim, data_y_min)
        ax.set_xlim(*xlims)
        ax.set_ylim((0, ylim))
        ax.axvline(x=0, color='k', linestyle='dotted')
        ax.axhline(y=sidak_corrected_alpha, color='k', linestyle='dotted')
        ax.text(-100, sidak_corrected_alpha, 'α*',
                bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        ax.text(600, sidak_corrected_alpha, 'α*',
                bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        ax.set_yticks(_get_yticks(ylim))
        # Format yaxis as p-values (e.g. 10^{-50} instead of -50)
        pval_labels = [
            f"$10^{{{int(t)}}}$" if t != 0 else "1"  # Instead of 10^0
            for t in ax.get_yticks()
        ]
        ax.set_yticklabels(pval_labels)

    # format one-off axis qualities
    if paired_axes:
        top_ax, bottom_ax = axes
        top_ax.set_xticklabels([])
        bottom_ax.invert_yaxis()
    else:
        top_ax = bottom_ax = axes[0]
    top_ax.set_title('Function Expression')
    bottom_ax.set_xlabel('Latency (ms) relative to onset of the environment')
    # TODO: hard-coded?
    bottom_ax.xaxis.set_ticks(np.arange(-200, 800 + 1, 100))
    if paired_axes:
        top_ax.text(s=axes_names[0],
                    x=-180, y=ylim * 0.95,
                    style='italic', verticalalignment='center')
        bottom_ax.text(s=axes_names[1],
                       x=-180, y=ylim * 0.95,
                       style='italic', verticalalignment='center')
    fig.supylabel(f'p-value (with α at 5-sigma, Šidák corrected)', x=0, y=0.5)
    bottom_ax.text(s='   onset of environment   ',
                   x=0, y=0 if paired_axes else ylim/2,  # vertically centred
                   color='white', fontsize='x-small',
                   bbox={'facecolor': 'grey', 'edgecolor': 'none'}, verticalalignment='center',
                   horizontalalignment='center', rotation='vertical')

    # Legend for plotted function
    split_legend_at_n_functions = 15
    legend_n_col = 2 if len(custom_handles) > split_legend_at_n_functions else 2
    if hidden_functions_in_legend and len(not_shown) > 0:
        if len(not_shown) > split_legend_at_n_functions:
            legend_n_col = 2
        # Plot dummy legend for other functions which are included in model selection but not plotted
        leg = bottom_ax.legend(labels=not_shown, fontsize="x-small", alignment="left",
                               title="Non-plotted functions",
                               ncol=legend_n_col,
                               bbox_to_anchor=(1.02, -0.02), loc="lower left",
                               # Hide lines for non-plotted functions
                               handlelength=0, handletextpad=0)
        for lh in leg.legend_handles:
            lh.set_alpha(0)
    top_ax.legend(handles=custom_handles, labels=custom_labels, fontsize='x-small', alignment="left",
                  title="Plotted functions",
                  ncol=legend_n_col,
                  loc="upper left", bbox_to_anchor=(1.02, 1.02))

    if save_to is not None:
        pyplot.rcParams['savefig.dpi'] = 300
        save_to = Path(save_to)

        if overwrite or not save_to.exists():
            pyplot.savefig(Path(save_to), bbox_inches='tight')
        raise FileExistsError(save_to)

    pyplot.show()
    pyplot.close()


def _get_best_xlims(xlims, data_x_min, data_x_max):
    default_xlims = (-200, 800)
    if xlims is None:
        xlims = (None, None)
    xmin, xmax = xlims
    if xmin is None:
        xmin = min(default_xlims[0], data_x_min)
    if xmax is None:
        xmax = max(default_xlims[1], data_x_max)
    xlims = (xmin, xmax)
    return xlims


def _get_best_ylim(ylim: float | None, data_y_min):
    if ylim is not None:
        return ylim
    default_y_min = -1 * _MAJOR_TICK_SIZE
    ylim = min(default_y_min, data_y_min)
    # Round to nearest major tick
    major_tick = np.floor(ylim / _MAJOR_TICK_SIZE) * _MAJOR_TICK_SIZE
    ylim = major_tick
    return ylim


def _get_yticks(ylim):
    n_major_ticks = int(ylim / _MAJOR_TICK_SIZE) * -1
    last_major_tick = -1 * n_major_ticks * _MAJOR_TICK_SIZE
    return np.linspace(start=0, stop=last_major_tick, num=n_major_ticks + 1)
