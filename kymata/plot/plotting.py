from pathlib import Path
from itertools import cycle
from typing import Optional, Sequence, Dict, Tuple
from statistics import NormalDist

from matplotlib import pyplot, colors
from matplotlib.lines import Line2D
import numpy as np
from pandas import DataFrame
from seaborn import color_palette

from kymata.entities.expression import ExpressionSet


def plot_expression_plot(
        expression_set: ExpressionSet,
        # TODO: order of this arg gives front-to-back z-order
        include_functions: Optional[Sequence[str]] = None,
        # Statistical kwargs
        alpha: float = 1 - NormalDist(mu=0, sigma=1).cdf(5),  # 5-sigma
        # Style kwargs
        color: Optional[Dict[str, str]] = None,  # function_name → colour name
        ylim: Optional[Tuple[float, float]] = None,
        # I/O args
        save_to: Optional[Path] = None,
):
    """Generates an expression plot"""

    # Default arg values
    if include_functions is None:
        # Plot all
        include_functions = expression_set.functions
    if color is None:
        color = dict()
    if ylim is None:
        ylim = 10 ** -100

    # Default colours
    cycol = cycle(color_palette("Set1"))
    for function in include_functions:
        if function not in color:
            color[function] = colors.to_hex(next(cycol))

    bonferroni_corrected_alpha = 1 - (
        (1 - alpha)
        ** (1 / (2
                 * len(expression_set.latencies)
                 * len(expression_set.hexels))))

    best_functions_lh: DataFrame
    best_functions_rh: DataFrame
    best_functions_lh, best_functions_rh = expression_set.best_functions()

    fig, (left_hem_expression_plot, right_hem_expression_plot) = pyplot.subplots(nrows=2, ncols=1, figsize=(12, 7))
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(right=0.84, left=0.08)

    custom_handles = []
    custom_labels = []
    for function in include_functions:

        custom_handles.extend([Line2D([], [], marker='.', color=color[function], linestyle='None')])
        custom_labels.append(function)

        # left
        data_left = best_functions_lh[best_functions_lh["function"] == function]
        x_left = data_left["latency"].values * 1000
        y_left = data_left["value"].values
        left_color = np.where(np.array(y_left) <= bonferroni_corrected_alpha, color[function], 'black')
        left_hem_expression_plot.vlines(x=x_left, ymin=1, ymax=y_left, color=left_color)
        left_hem_expression_plot.scatter(x_left, y_left, color=left_color, s=20)

        # right
        data_right = best_functions_rh[best_functions_rh["function"] == function]
        x_right = data_right["latency"].values * 1000
        y_right = data_right["value"].values
        right_color = np.where(np.array(y_right) <= bonferroni_corrected_alpha, color[function], 'black')
        right_hem_expression_plot.vlines(x=x_right, ymin=1, ymax=y_right, color=left_color)
        right_hem_expression_plot.scatter(x_right, y_right, color=right_color, s=20)

    # format shared axis qualities

    for plot in [right_hem_expression_plot, left_hem_expression_plot]:
        plot.set_yscale('log')
        # TODO: hard-coded?
        plot.set_xlim(-200, 800)
        plot.set_ylim((1, ylim))
        plot.axvline(x=0, color='k', linestyle='dotted')
        plot.axhline(y=bonferroni_corrected_alpha, color='k', linestyle='dotted')
        plot.text(-100, bonferroni_corrected_alpha, 'α*',
                  bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        plot.text(600, bonferroni_corrected_alpha, 'α*',
                  bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        plot.set_yticks([1, 10 ** -50, 10 ** -100])

    # format one-off axis qualities
    left_hem_expression_plot.set_title('Function Expression')
    left_hem_expression_plot.set_xticklabels([])
    right_hem_expression_plot.set_xlabel('Latency (ms) relative to onset of the environment')
    right_hem_expression_plot.xaxis.set_ticks(np.arange(-200, 800 + 1, 100))
    right_hem_expression_plot.invert_yaxis()
    left_hem_expression_plot.text(-180, ylim * 10000000, 'left hemisphere', style='italic',
                                  verticalalignment='center')
    right_hem_expression_plot.text(-180, ylim * 10000000, 'right hemisphere', style='italic',
                                   verticalalignment='center')
    y_axis_label = f'p-value (with α at 5-sigma, Bonferroni corrected)'
    left_hem_expression_plot.text(-275, 1, y_axis_label, verticalalignment='center', rotation='vertical')
    right_hem_expression_plot.text(0, 1, '   onset of environment   ', color='white', fontsize='x-small',
                                   bbox={'facecolor': 'grey', 'edgecolor': 'none'}, verticalalignment='center',
                                   horizontalalignment='center', rotation='vertical')
    left_hem_expression_plot.legend(handles=custom_handles, labels=custom_labels, fontsize='x-small',
                                    bbox_to_anchor=(1.2, 1))

    if save_to is not None:
        pyplot.rcParams['savefig.dpi'] = 300
        pyplot.savefig(Path(save_to))

    pyplot.show()
    pyplot.close()


def lognuniform(low=0, high=1, size=None, base=np.e):
    return np.random.uniform(low, high, size) / 1000000000000
