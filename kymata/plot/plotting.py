from pathlib import Path
from itertools import cycle
from typing import Optional, Sequence, Dict
from statistics import NormalDist

from matplotlib import pyplot, colors
from matplotlib.lines import Line2D
import numpy as np
from pandas import DataFrame
from seaborn import color_palette

from kymata.entities.expression import HexelExpressionSet


# 10 ** -this will be the ytick interval and also the resolution to which the ylims will be rounded
_MAJOR_TICK_SIZE = 50


def expression_plot(
        expression_set: HexelExpressionSet,
        show_only: Optional[str | Sequence[str]] = None,
        # Statistical kwargs
        alpha: float = 1 - NormalDist(mu=0, sigma=1).cdf(5),  # 5-sigma
        # Style kwargs
        color: Optional[str | Dict[str, str] | list[str]] = None,
        ylim: Optional[float] = None,
        xlims: Optional[tuple[Optional[float], Optional[float]]] = None,
        hidden_functions_in_legend: bool = True,
        # I/O args
        save_to: Optional[Path] = None,
):
    """
    Generates an expression plot

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

    sidak_corrected_alpha = 1 - (
        (1 - alpha)
        ** (1 / (2
                 * len(expression_set.latencies)
                 * len(expression_set.hexels)
                 * len(show_only))))

    best_functions_lh: DataFrame
    best_functions_rh: DataFrame
    best_functions_lh, best_functions_rh = expression_set.best_functions()

    fig, (left_hem_expression_plot, right_hem_expression_plot) = pyplot.subplots(nrows=2, ncols=1, figsize=(12, 7))
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(right=0.84, left=0.08)

    custom_handles = []
    custom_labels = []
    data_x_min, data_x_max = np.Inf, -np.Inf
    # Careful, the y value is inverted, with y==1 on the origin and y<1 away from the origin.
    # "y_min" here is real absolute min value in the data (closest to zero)
    data_y_min = np.Inf
    for function in show_only:

        custom_handles.extend([Line2D([], [], marker='.', color=color[function], linestyle='None')])
        custom_labels.append(function)

        # left
        data_left = best_functions_lh[best_functions_lh["function"] == function]
        x_left = data_left["latency"].values * 1000
        y_left = data_left["value"].values
        left_color = np.where(np.array(y_left) <= sidak_corrected_alpha, color[function], 'black')
        left_hem_expression_plot.vlines(x=x_left, ymin=1, ymax=y_left, color=left_color)
        left_hem_expression_plot.scatter(x_left, y_left, color=left_color, s=20)

        # right
        data_right = best_functions_rh[best_functions_rh["function"] == function]
        x_right = data_right["latency"].values * 1000
        y_right = data_right["value"].values
        right_color = np.where(np.array(y_right) <= sidak_corrected_alpha, color[function], 'black')
        right_hem_expression_plot.vlines(x=x_right, ymin=1, ymax=y_right, color=right_color)
        right_hem_expression_plot.scatter(x_right, y_right, color=right_color, s=20)

        data_x_min = min(data_x_min,
                         x_left.min() if len(x_left) > 0 else np.Inf,
                         x_right.min() if len(x_right) > 0 else np.Inf)
        data_x_max = max(data_x_max,
                         x_left.max() if len(x_left) > 0 else -np.Inf,
                         x_right.max() if len(x_right) > 0 else- np.Inf)
        data_y_min = min(data_y_min,
                         y_left.min() if len(y_left) > 0 else np.Inf,
                         y_right.min() if len(y_right) > 0 else np.Inf)

    # format shared axis qualities

    for plot in [right_hem_expression_plot, left_hem_expression_plot]:
        plot.set_yscale('log')
        xlims = _get_best_xlims(xlims, data_x_min, data_x_max)
        ylim = _get_best_ylim(ylim, data_y_min)
        plot.set_xlim(*xlims)
        plot.set_ylim((1, ylim))
        plot.axvline(x=0, color='k', linestyle='dotted')
        plot.axhline(y=sidak_corrected_alpha, color='k', linestyle='dotted')
        plot.text(-100, sidak_corrected_alpha, 'α*',
                  bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        plot.text(600, sidak_corrected_alpha, 'α*',
                  bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        plot.set_yticks(_get_yticks(ylim))

    # format one-off axis qualities
    left_hem_expression_plot.set_title('Function Expression')
    left_hem_expression_plot.set_xticklabels([])
    right_hem_expression_plot.set_xlabel('Latency (ms) relative to onset of the environment')
    # TODO: hard-coded?
    right_hem_expression_plot.xaxis.set_ticks(np.arange(-200, 800 + 1, 100))
    right_hem_expression_plot.invert_yaxis()
    left_hem_expression_plot.text(-180, ylim * 10000000, 'left hemisphere', style='italic',
                                  verticalalignment='center')
    right_hem_expression_plot.text(-180, ylim * 10000000, 'right hemisphere', style='italic',
                                   verticalalignment='center')
    y_axis_label = f'p-value (with α at 5-sigma, Šidák corrected)'
    left_hem_expression_plot.text(-275, 1, y_axis_label, verticalalignment='center', rotation='vertical')
    right_hem_expression_plot.text(0, 1, '   onset of environment   ', color='white', fontsize='x-small',
                                   bbox={'facecolor': 'grey', 'edgecolor': 'none'}, verticalalignment='center',
                                   horizontalalignment='center', rotation='vertical')
    # Legend for plotted functions
    split_legend_at_n_functions = 15
    legend_n_col = 2 if len(custom_handles) > split_legend_at_n_functions else 2
    if hidden_functions_in_legend and len(not_shown) > 0:
        if len(not_shown) > split_legend_at_n_functions:
            legend_n_col = 2
        # Plot dummy legend for other functions which are included in model selection but not plotted
        leg = right_hem_expression_plot.legend(labels=not_shown, fontsize="x-small", alignment="left",
                                               title="Non-plotted functions",
                                               ncol=legend_n_col,
                                               bbox_to_anchor=(1.02, -0.02), loc="lower left",
                                               # Hide lines for non-plotted functions
                                               handlelength=0, handletextpad=0)
        for lh in leg.legend_handles:
            lh.set_alpha(0)
    left_hem_expression_plot.legend(handles=custom_handles, labels=custom_labels, fontsize='x-small', alignment="left",
                                    title="Plotted functions",
                                    ncol=legend_n_col,
                                    loc="upper left", bbox_to_anchor=(1.02, 1.02))

    if save_to is not None:
        pyplot.rcParams['savefig.dpi'] = 300
        pyplot.savefig(Path(save_to), bbox_inches='tight')

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
    default_y_min = 10 ** (-1 * _MAJOR_TICK_SIZE)
    ylim = min(default_y_min, data_y_min)
    # Round to nearest major tick
    major_tick = np.floor(np.log10(ylim) / _MAJOR_TICK_SIZE) * _MAJOR_TICK_SIZE
    ylim = 10 ** major_tick
    return ylim


def _get_yticks(ylim):
    n_major_ticks = int(np.log10(ylim) / _MAJOR_TICK_SIZE) * -1
    last_major_tick = 10 ** (-1 * n_major_ticks * _MAJOR_TICK_SIZE)
    return np.geomspace(start=1, stop=last_major_tick, num=n_major_ticks + 1)


if __name__ == '__main__':
    from kymata.datasets.sample import KymataMirror2023Q3Dataset

    # create new expression set object and add to it
    expression_data_kymata_mirror = KymataMirror2023Q3Dataset().to_expressionset()

    expression_plot(expression_data_kymata_mirror, show_only=expression_data_kymata_mirror.functions[1:], save_to=Path("/Users/cai/Desktop/temp.png"))
