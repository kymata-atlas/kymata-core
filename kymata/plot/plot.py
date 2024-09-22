from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from statistics import NormalDist
from typing import Optional, Sequence, NamedTuple, Any, Type, Literal
from warnings import warn

import numpy as np
from matplotlib import pyplot
from matplotlib.colors import to_hex, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
from mne import SourceEstimate
from numpy.typing import NDArray
from pandas import DataFrame
from seaborn import color_palette

from kymata.entities.expression import (
    HexelExpressionSet,
    SensorExpressionSet,
    ExpressionSet,
    DIM_FUNCTION, COL_LOGP_VALUE, DIM_LATENCY,
)
from kymata.entities.functions import Function
from kymata.math.p_values import p_to_logp
from kymata.math.rounding import round_down, round_up
from kymata.plot.layouts import (
    get_meg_sensor_xy,
    get_eeg_sensor_xy,
    get_meg_sensors,
    get_eeg_sensors,
)

transparent = (0, 0, 0, 0)

# log scale: 10 ** -this will be the ytick interval and also the resolution to which the ylims will be rounded
_MAJOR_TICK_SIZE = 50


class _AxName:
    """Canonical names for the axes."""

    top = "top"
    bottom = "bottom"
    main = "main"
    minimap_top = "minimap top"
    minimap_bottom = "minimap bottom"
    minimap_main = "minimap main"


@dataclass
class _MosaicSpec:
    mosaic: list[list[str]]
    width_ratios: list[float] | None
    fig_size: tuple[float, float]
    expression_yaxis_label_xpos: float
    subplots_adjust_kwargs: dict[str, float] = None

    def __post_init__(self):
        if self.subplots_adjust_kwargs is None:
            self.subplots_adjust_kwargs = dict()

    def to_subplots(self) -> tuple[pyplot.Figure, dict[str, pyplot.Axes]]:
        return pyplot.subplot_mosaic(
            self.mosaic, width_ratios=self.width_ratios, figsize=self.fig_size
        )


def _minimap_mosaic(
    paired_axes: bool,
    show_minimap: bool,
    expression_set_type: Type[ExpressionSet],
    fig_size: tuple[float, float],
) -> _MosaicSpec:
    # Set defaults:
    if show_minimap:
        width_ratios = [1, 3]
        subplots_adjust = {
            "hspace": 0,
            "wspace": 0.25,
            "left": 0.02,
            "right": 0.8,
        }
        # Place next to the expression plot yaxis
        yaxis_label_xpos = width_ratios[0] / (width_ratios[1] + width_ratios[0]) - 0.04
    else:
        width_ratios = None
        subplots_adjust = {
            "hspace": 0,
            "left": 0.08,
            "right": 0.84,
        }
        yaxis_label_xpos = 0.04

    if paired_axes:
        if show_minimap:
            if expression_set_type == HexelExpressionSet:
                spec = [
                    [_AxName.minimap_top, _AxName.top],
                    [_AxName.minimap_bottom, _AxName.bottom],
                ]
            elif expression_set_type == SensorExpressionSet:
                spec = [
                    [_AxName.minimap_main, _AxName.top],
                    [_AxName.minimap_main, _AxName.bottom],
                ]
            else:
                raise NotImplementedError()
        else:
            spec = [
                [_AxName.top],
                [_AxName.bottom],
            ]
    else:
        if show_minimap:
            spec = [
                [_AxName.minimap_main, _AxName.main],
            ]
        else:
            spec = [
                [_AxName.main],
            ]

    return _MosaicSpec(
        mosaic=spec,
        width_ratios=width_ratios,
        fig_size=fig_size,
        subplots_adjust_kwargs=subplots_adjust,
        expression_yaxis_label_xpos=yaxis_label_xpos,
    )


def _hexel_minimap_data(
    expression_set: HexelExpressionSet, alpha_logp: float, show_functions: list[str]
) -> tuple[NDArray, NDArray]:
    """
    Generates data arrays for a minimap visualization of significant hexels in a HexelExpressionSet.

    Args:
        expression_set (HexelExpressionSet): The set of hexel expressions to analyze.
        alpha_logp (float): The logarithm of the p-value threshold for significance. Hexels with values
            below this threshold are considered significant.
        show_functions (list[str]): A list of function names to consider for significance. Only these
            functions will be checked for significant hexels.

    Returns:
        tuple[NDArray, NDArray]: A tuple containing two arrays (one for the left hemisphere and one for
        the right hemisphere). Each array has a length equal to the number of hexels in the respective
        hemisphere, with entries:
            - 0, if no function is ever significant for this hexel.
            - i + 1, where i is the index of the function (from show_functions) that is significant
              for this hexel.

    Notes:
        This function identifies which hexels are significant for the given functions based on a provided
        significance threshold. It returns arrays for both the left and right hemispheres, where each entry
        indicates whether the hexel is significant for any function and, if so, which function it is
        significant for.
    """
    data_left = np.zeros((len(expression_set.hexels_left),))
    data_right = np.zeros((len(expression_set.hexels_right),))
    best_functions_left, best_functions_right = expression_set.best_functions()
    best_functions_left = best_functions_left[best_functions_left[COL_LOGP_VALUE] < alpha_logp]
    best_functions_right = best_functions_right[
        best_functions_right[COL_LOGP_VALUE] < alpha_logp
    ]
    for function_i, function in enumerate(
        expression_set.functions,
        # 1-indexed, as 0 will refer to transparent
        start=1,
    ):
        if function not in show_functions:
            continue
        significant_hexel_names_left = best_functions_left[
            best_functions_left[DIM_FUNCTION] == function
        ][expression_set.channel_coord_name]
        hexel_idxs_left = np.searchsorted(
            expression_set.hexels_left, significant_hexel_names_left.to_numpy()
        )
        data_left[hexel_idxs_left] = function_i

        significant_hexel_names_right = best_functions_right[
            best_functions_right[DIM_FUNCTION] == function
        ][expression_set.channel_coord_name]
        hexel_idxs_right = np.searchsorted(
            expression_set.hexels_right, significant_hexel_names_right.to_numpy()
        )
        data_right[hexel_idxs_right] = function_i

    return data_left, data_right


def _plot_function_expression_on_axes(
    ax: pyplot.Axes,
    function_data: DataFrame,
    color,
    sidak_corrected_alpha: float,
    filled: bool,
):
    """
    Returns:
        x_min, x_max, y_min, y_max
            logp values for axis limits
            Note: *_min and *_max values are np.Inf and -np.Inf respectively if x or y is empty
                  (so they can be added to min() and max() without altering the result).
    """
    x = function_data[DIM_LATENCY].values * 1000  # Convert to milliseconds
    y = function_data[COL_LOGP_VALUE].values
    c = np.where(np.array(y) <= sidak_corrected_alpha, color, "black")
    ax.vlines(x=x, ymin=1, ymax=y, color=c)
    ax.scatter(x, y, facecolors=c if filled else "none", s=20, edgecolors=c)

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


sensor_left_right_assignment: tuple[AxisAssignment, AxisAssignment] = (
    AxisAssignment(
        axis_name="left",
        axis_channels=[
            sensor for sensor, (x, y) in get_meg_sensor_xy().items() if x <= 0
        ]
        + [sensor for sensor, (x, y) in get_eeg_sensor_xy().items() if x <= 0],
    ),
    AxisAssignment(
        axis_name="right",
        axis_channels=[
            sensor for sensor, (x, y) in get_meg_sensor_xy().items() if x >= 0
        ]
        + [sensor for sensor, (x, y) in get_eeg_sensor_xy().items() if x >= 0],
    ),
)


def _plot_minimap_sensor(
    expression_set: ExpressionSet,
    minimap_axis: pyplot.Axes,
    colors: dict[str, str],
    alpha_logp: float,
):
    raise NotImplementedError("Minimap not yet implemented for sensor data")


def _plot_minimap_hexel(
    expression_set: HexelExpressionSet,
    show_functions: list[str],
    lh_minimap_axis: pyplot.Axes,
    rh_minimap_axis: pyplot.Axes,
    view: str,
    surface: str,
    colors: dict[str, Any],
    alpha_logp: float,
):
    # Ensure we have the FSAverage dataset downloaded
    from kymata.datasets.fsaverage import FSAverageDataset

    fsaverage = FSAverageDataset(download=True)
    os.environ["SUBJECTS_DIR"] = str(fsaverage.path)

    # Functions which aren't being shown need to be padded into the colormap, for indexing purposes, but should show up
    # as transparent
    colors = colors.copy()
    for function in expression_set.functions:
        if function not in show_functions:
            colors[function] = transparent

    # segment at index 0 will map to transparent
    # segment at index i will map to function of index i-1
    colormap = LinearSegmentedColormap.from_list(
        "custom",
        # Insert transparent for index 0
        colors=[transparent] + [colors[f] for f in expression_set.functions],
        # +1 for the transparency
        N=len(expression_set.functions) + 1,
    )
    data_left, data_right = _hexel_minimap_data(
        expression_set, alpha_logp=alpha_logp, show_functions=show_functions
    )
    stc = SourceEstimate(
        data=np.concatenate([data_left, data_right]),
        vertices=[expression_set.hexels_left, expression_set.hexels_right],
        tmin=0,
        tstep=1,
    )
    warn(
        "Plotting on the fsaverage brain. Ensure that hexel numbers match those of the fsaverage brain."
    )
    plot_kwargs = dict(
        subject="fsaverage",
        surface=surface,
        views=view,
        colormap=colormap,
        smoothing_steps=2,
        background="white",
        spacing="ico5",
        brain_kwargs={
            "offscreen": True,  # offscreen here appears to not actually do anything in mne
        },
        time_viewer=False,
        colorbar=False,
        transparent=False,
        clim=dict(
            kind="value",
            lims=[0, len(expression_set.functions) / 2, len(expression_set.functions)],
        ),
    )
    # Plot left view
    lh_brain = stc.plot(hemi="lh", **plot_kwargs)
    lh_brain_fig = pyplot.gcf()
    lh_minimap_axis.imshow(lh_brain.screenshot())
    hide_axes(lh_minimap_axis)
    pyplot.close(lh_brain_fig)
    # Plot right view
    rh_brain = stc.plot(hemi="rh", **plot_kwargs)
    rh_brain_fig = pyplot.gcf()
    rh_minimap_axis.imshow(rh_brain.screenshot())
    hide_axes(rh_minimap_axis)
    pyplot.close(rh_brain_fig)


def hide_axes(axes: pyplot.Axes):
    """Hide all axes markings from a pyplot.Axes."""
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.axis("off")


def expression_plot(
    expression_set: ExpressionSet,
    show_only: Optional[str | Sequence[str]] = None,
    paired_axes: bool = True,
    # Statistical kwargs
    alpha: float = 1 - NormalDist(mu=0, sigma=1).cdf(5),  # 5-sigma
    # Style kwargs
    color: Optional[str | dict[str, str] | list[str]] = None,
    ylim: Optional[float] = None,
    xlims: Optional[tuple[float | None, float | None]] = None,
    hidden_functions_in_legend: bool = True,
    title: str = "Function expression",
    fig_size: tuple[float, float] = (12, 7),
    # Display options
    minimap: bool = False,
    minimap_view: str = "lateral",
    minimap_surface: str = "inflated",
    show_only_sensors: Optional[Literal["eeg", "meg"]] = None,
    # I/O args
    save_to: Optional[Path] = None,
    overwrite: bool = True,
    show_legend: bool = True,
    legend_display: dict[str, str] | None = None,
) -> pyplot.Figure:
    """
    Generates a plot of function expressions over time with optional display customizations.

    Args:
        expression_set (ExpressionSet): The set of expressions to plot, containing functions and associated data.
        show_only (Optional[str | Sequence[str]], optional): A string or a sequence of strings specifying which
            functions to plot.
            If None, all functions in the expression_set will be plotted. Default is None.
        paired_axes (bool, optional): When True, shows the expression plot split into left and right axes.
            When False, all points are shown on the same axis. Default is True.
        alpha (float, optional): Significance level for statistical tests, defaulting to a 5-sigma threshold.
        color (Optional[str | dict[str, str] | list[str]], optional): Color settings for the plot. Can be a single
            color, a dictionary mapping function names to colors, or a list of colors. Default is None.
        ylim (Optional[float], optional): The y-axis limit (p-value). Use log10 of the desired value — e.g. if the
            desired limit is 10^-100, supply ylim=-100. If None, it will be determined automatically. Default is None.
        xlims (tuple[Optional[float], Optional[float]], optional): The x-axis limits as a tuple (in ms). None to use
            default values, or set either entry to None to use the default for that value. Default is (-100, 800).
        hidden_functions_in_legend (bool, optional): If True, includes non-plotted functions in the legend.
            Default is True.
        title (str, optional): Title over the top axis in the figure. Default is "Function expression".
        fig_size (tuple[float, float], optional): Figure size in inches. Default is (12, 7).
        minimap (bool, optional): If True, displays a minimap of the expression data. Default is False.
        minimap_view (str, optional): The view type for the minimap, either "lateral" or other specified views.
            Valid options are:
            `"lateral"`: From the left or right side such that the lateral (outside) surface of the given hemisphere is
                         visible.
            `"medial"`: From the left or right side such that the medial (inside) surface of the given hemisphere is
                        visible (at least when in split or single-hemi mode).
            `"rostral"`: From the front.
            `"caudal"`: From the rear.
            `"dorsal"`: From above, with the front of the brain pointing up.
            `"ventral"`: From below, with the front of the brain pointing up.
            `"frontal"`: From the front and slightly lateral, with the brain slightly tilted forward (yielding a view
                         from slightly above).
            `"parietal"`: From the rear and slightly lateral, with the brain slightly tilted backward (yielding a view
                          from slightly above).
            `"axial"`: From above with the brain pointing up (same as 'dorsal').
            `"sagittal"`: From the right side.
            `"coronal"`: From the rear.
            Default is `lateral`.
        minimap_surface (str, optional): The surface type for the minimap, such as "inflated". Default is "inflated".
        show_only_sensors (str, optional): Show only one type of sensors. "meg" for MEG sensors, "eeg" for EEG sensors.
            None to show all sensors. Supplying this value with something other than a SensorExpressionSet causes will
            throw an exception. Default is None.
        save_to (Optional[Path], optional): Path to save the generated plot. If None, the plot is not saved.
            Default is None.
        overwrite (bool, optional): If True, overwrite the existing file if it exists. Default is True.
        show_legend (bool, optional): If True, displays the legend. Default is True.
        legend_display (dict[str, str] | None, optional): Allows grouping of multiple functions under the same legend
            item. Provide a dictionary mapping true function names to display names. None applies no grouping.
            Default is None.

    Returns:
        pyplot.Figure: The matplotlib figure object containing the generated plot.

    Raises:
        FileExistsError: If the file already exists at save_to and overwrite is set to False.

    Notes:
        The function plots the expression data with options to customize the appearance and statistical
        significance thresholds. It supports different data types (e.g., HexelExpressionSet, SensorExpressionSet)
        and can handle paired axes for left/right hemisphere data.
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
    elif isinstance(color, list):
        # List specified, then pair up in order
        assert len(color) == len(show_only)
        color = {f: c for f, c in zip(show_only, color)}

    # Default colours
    cycol = cycle(color_palette("Set1"))
    for function in show_only:
        if function not in color:
            color[function] = to_hex(next(cycol))

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
            raise NotImplementedError(
                "HexelExpressionSets have preset hemisphere assignments"
            )
        elif isinstance(expression_set, SensorExpressionSet):
            axes_names = ("",)
            # Wrap into list
            best_functions = (best_functions,)
        else:
            raise NotImplementedError()

    if isinstance(expression_set, HexelExpressionSet):
        n_channels = len(expression_set.hexels_left) + len(expression_set.hexels_right)
    elif isinstance(expression_set, SensorExpressionSet):
        n_channels = len(expression_set.sensors)
    else:
        raise NotImplementedError()

    chosen_channels = _restrict_channels(
        expression_set, best_functions, show_only_sensors
    )

    sidak_corrected_alpha = 1 - (
        (1 - alpha)
        ** (1 / (2 * len(expression_set.latencies) * n_channels * len(show_only)))
    )

    sidak_corrected_alpha = p_to_logp(sidak_corrected_alpha)

    def _custom_label(function_name):
        if legend_display is not None:
            if function_name in legend_display.keys():
                return legend_display[function_name]
        return function_name

    mosaic = _minimap_mosaic(
        paired_axes=paired_axes,
        show_minimap=minimap,
        expression_set_type=type(expression_set),
        fig_size=fig_size,
    )

    fig: pyplot.Figure
    axes: dict[str, pyplot.Axes]
    fig, axes = mosaic.to_subplots()

    expression_axes_list: list[pyplot.Axes]
    if paired_axes:
        expression_axes_list = [
            axes[_AxName.top],
            axes[_AxName.bottom],
        ]  # For iterating over in a predictable order
    else:
        expression_axes_list = [axes[_AxName.main]]

    fig.subplots_adjust(**mosaic.subplots_adjust_kwargs)

    # Turn off autoscaling to make plotting faster, and since we manually set the axes scale later
    pyplot.autoscale(False)

    custom_handles = []
    custom_labels = []
    data_x_min, data_x_max = np.Inf, -np.Inf
    data_y_min = np.Inf
    for function in show_only:
        custom_label = _custom_label(function)
        if custom_label not in custom_labels:
            custom_handles.extend(
                [Line2D([], [], marker=".", color=color[function], linestyle="None")]
            )
            custom_labels.append(custom_label)

        # We have a special case with paired sensor data, in that some sensors need to appear
        # on both sides of the midline.
        if paired_axes and isinstance(expression_set, SensorExpressionSet):
            assign_left_right_channels = sensor_left_right_assignment
            # Some points will be plotted on one axis, filled, some on both, empty
            top_chans = (
                set(assign_left_right_channels[0].axis_channels) & chosen_channels
            )
            bottom_chans = (
                set(assign_left_right_channels[1].axis_channels) & chosen_channels
            )
            # Symmetric difference
            both_chans = top_chans & bottom_chans
            top_chans -= both_chans
            bottom_chans -= both_chans
            for ax, best_funs_this_ax, chans_this_ax in zip(
                expression_axes_list, best_functions, (top_chans, bottom_chans)
            ):
                # Plot filled
                (
                    x_min,
                    x_max,
                    y_min,
                    _y_max,
                ) = _plot_function_expression_on_axes(
                    function_data=best_funs_this_ax[
                        (best_funs_this_ax[DIM_FUNCTION] == function)
                        & (
                            best_funs_this_ax[expression_set.channel_coord_name].isin(
                                chans_this_ax
                            )
                        )
                    ],
                    color=color[function],
                    ax=ax,
                    sidak_corrected_alpha=sidak_corrected_alpha,
                    filled=True,
                )
                data_x_min = min(data_x_min, x_min)
                data_x_max = max(data_x_max, x_max)
                data_y_min = min(data_y_min, y_min)
                # Plot empty
                (
                    x_min,
                    x_max,
                    y_min,
                    _y_max,
                ) = _plot_function_expression_on_axes(
                    function_data=best_funs_this_ax[
                        (best_funs_this_ax[DIM_FUNCTION] == function)
                        & (
                            best_funs_this_ax[expression_set.channel_coord_name].isin(
                                both_chans
                            )
                        )
                    ],
                    color=color[function],
                    ax=ax,
                    sidak_corrected_alpha=sidak_corrected_alpha,
                    filled=False,
                )
                data_x_min = min(data_x_min, x_min)
                data_x_max = max(data_x_max, x_max)
                data_y_min = min(data_y_min, y_min)

        # With non-sensor data, or non-paired axes, we can treat these cases together
        else:
            # As normal, plot appropriate filled points in each axis
            for ax, best_funs_this_ax in zip(expression_axes_list, best_functions):
                (
                    x_min,
                    x_max,
                    y_min,
                    _y_max,
                ) = _plot_function_expression_on_axes(
                    function_data=best_funs_this_ax[
                        (best_funs_this_ax[DIM_FUNCTION] == function)
                        & (
                            best_funs_this_ax[expression_set.channel_coord_name].isin(
                                chosen_channels
                            )
                        )
                    ],
                    color=color[function],
                    ax=ax,
                    sidak_corrected_alpha=sidak_corrected_alpha,
                    filled=True,
                )
                data_x_min = min(data_x_min, x_min)
                data_x_max = max(data_x_max, x_max)
                data_y_min = min(data_y_min, y_min)

    # Format shared axis qualities
    for ax in expression_axes_list:
        xlims = _get_best_xlims(xlims, data_x_min, data_x_max)
        ylim = _get_best_ylim(ylim, data_y_min)
        ax.set_xlim(*xlims)
        ax.set_ylim((0, ylim))
        ax.axvline(x=0, color="k", linestyle="dotted")
        ax.axhline(y=sidak_corrected_alpha, color="k", linestyle="dotted")
        ax.text(
            -50,
            sidak_corrected_alpha,
            "α*",
            bbox={"facecolor": "white", "edgecolor": "none"},
            verticalalignment="center",
        )
        ax.text(
            600,
            sidak_corrected_alpha,
            "α*",
            bbox={"facecolor": "white", "edgecolor": "none"},
            verticalalignment="center",
        )
        ax.set_yticks(_get_yticks(ylim))
        # Format yaxis as p-values (e.g. 10^{-50} instead of -50)
        pval_labels = [
            f"$10^{{{int(t)}}}$" if t != 0 else "1"  # Instead of 10^0
            for t in ax.get_yticks()
        ]
        ax.set_yticklabels(pval_labels)

    # Plot minimap
    if minimap:
        if isinstance(expression_set, SensorExpressionSet):
            _plot_minimap_sensor(
                expression_set,
                minimap_axis=axes[_AxName.minimap_main],
                colors=color,
                alpha_logp=sidak_corrected_alpha,
            )
        elif isinstance(expression_set, HexelExpressionSet):
            _plot_minimap_hexel(
                expression_set,
                show_functions=show_only,
                lh_minimap_axis=axes[_AxName.minimap_top],
                rh_minimap_axis=axes[_AxName.minimap_bottom],
                view=minimap_view,
                surface=minimap_surface,
                colors=color,
                alpha_logp=sidak_corrected_alpha,
            )
        else:
            raise NotImplementedError()

    # Format one-off axis qualities
    if paired_axes:
        top_ax = axes[_AxName.top]
        bottom_ax = axes[_AxName.bottom]
        top_ax.set_xticklabels([])
        bottom_ax.invert_yaxis()
    else:
        top_ax = bottom_ax = axes[_AxName.main]
    top_ax.set_title(title)
    bottom_ax.set_xlabel("Latency (ms) relative to onset of the environment")
    bottom_ax_xmin, bottom_ax_xmax = bottom_ax.get_xlim()
    bottom_ax.xaxis.set_major_locator(
        FixedLocator(_get_xticks((bottom_ax_xmin, bottom_ax_xmax)))
    )
    if paired_axes:
        top_ax.text(
            s=axes_names[0],
            x=bottom_ax_xmin + 20,
            y=ylim * 0.95,
            style="italic",
            verticalalignment="center",
        )
        bottom_ax.text(
            s=axes_names[1],
            x=bottom_ax_xmin + 20,
            y=ylim * 0.95,
            style="italic",
            verticalalignment="center",
        )
    fig.text(
        x=mosaic.expression_yaxis_label_xpos,
        y=0.5,
        s="p-value (with α at 5-sigma, Šidák corrected)",
        ha="center",
        va="center",
        rotation="vertical",
    )
    if bottom_ax_xmin <= 0 <= bottom_ax_xmax:
        bottom_ax.text(
            s="   onset of environment   ",
            x=0,
            y=0 if paired_axes else ylim / 2,
            color="black",
            fontsize="x-small",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
            verticalalignment="center",
            horizontalalignment="center",
            rotation="vertical",
        )

    # Legend for plotted function
    if show_legend:
        split_legend_at_n_functions = 15
        legend_n_col = 2 if len(custom_handles) > split_legend_at_n_functions else 2
        if hidden_functions_in_legend and len(not_shown) > 0:
            if len(not_shown) > split_legend_at_n_functions:
                legend_n_col = 2
            # Plot dummy legend for other functions which are included in model selection but not plotted
            custom_labels_not_shown = []
            dummy_patches = []
            for hidden_function in not_shown:
                custom_label = _custom_label(hidden_function)
                if custom_label not in custom_labels_not_shown:
                    custom_labels_not_shown.append(custom_label)
                    dummy_patches.append(Patch(color=None, label=custom_label))
            leg = bottom_ax.legend(
                labels=custom_labels_not_shown,
                fontsize="x-small",
                alignment="left",
                title="Non-plotted functions",
                ncol=legend_n_col,
                bbox_to_anchor=(1.02, -0.02),
                loc="lower left",
                handles=dummy_patches,
            )
            for lh in leg.legend_handles:
                lh.set_alpha(0)
        top_ax.legend(
            handles=custom_handles,
            labels=custom_labels,
            fontsize="x-small",
            alignment="left",
            title="Plotted functions",
            ncol=legend_n_col,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.02),
        )

    if save_to is not None:
        pyplot.rcParams["savefig.dpi"] = 300
        save_to = Path(save_to)

        if overwrite or not save_to.exists():
            pyplot.savefig(Path(save_to), bbox_inches="tight")
        else:
            raise FileExistsError(save_to)

    return fig


def _restrict_channels(
    expression_set: ExpressionSet,
    best_functions: tuple[DataFrame, ...],
    show_only_sensors: str | None,
):
    """Restrict to specific sensor type if requested."""
    if show_only_sensors is not None:
        if isinstance(expression_set, SensorExpressionSet):
            if show_only_sensors == "meg":
                chosen_channels = get_meg_sensors()
            elif show_only_sensors == "eeg":
                chosen_channels = get_eeg_sensors()
            else:
                raise NotImplementedError()
        else:
            raise ValueError("`show_only_sensors` only valid with sensor data.")
    else:
        if isinstance(expression_set, SensorExpressionSet):
            # All sensors
            chosen_channels = {
                sensor
                for best_funs_each_ax in best_functions
                for sensor in best_funs_each_ax[expression_set.channel_coord_name]
            }
        elif isinstance(expression_set, HexelExpressionSet):
            # All hexels
            chosen_channels = {
                sensor
                for best_funs_each_ax in best_functions
                for sensor in best_funs_each_ax[expression_set.channel_coord_name]
            }
        else:
            raise NotImplementedError()
    return chosen_channels


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


def _get_xticks(xlims: tuple[float, float]):
    xmin, xmax = xlims
    # Round to the nearest 100
    step = 100
    xmin = round_up(xmin, step)
    xmax = round_down(xmax, step)
    return np.arange(xmin, xmax + 1, step)


def _get_yticks(ylim):
    n_major_ticks = int(ylim / _MAJOR_TICK_SIZE) * -1
    last_major_tick = -1 * n_major_ticks * _MAJOR_TICK_SIZE
    return np.linspace(start=0, stop=last_major_tick, num=n_major_ticks + 1)


def plot_top_five_channels_of_gridsearch(
    latencies: NDArray,
    corrs: NDArray,
    function: Function,
    n_samples_per_split: int,
    n_reps: int,
    n_splits: int,
    auto_corrs: NDArray,
    log_pvalues: any,
    # I/O args
    save_to: Optional[Path] = None,
    overwrite: bool = True,
):
    """
    Generates correlation and p-value plots showing the top five channels of the gridsearch.

    Args:
        latencies (NDArray[any]): Array of latency values (e.g., time points in milliseconds) for the x-axis of the plots.
        corrs (NDArray[any]): Correlation coefficients array with shape (n_channels, n_conditions, n_splits, n_time_steps).
        function (Function): The function object whose name attribute will be used in the plot title.
        n_samples_per_split (int): Number of samples per split used in the grid search.
        n_reps (int): Number of repetitions in the grid search.
        n_splits (int): Number of splits in the grid search.
        auto_corrs (NDArray[any]): Auto-correlation values array used for plotting the function auto-correlation.
        log_pvalues (any): Array of log-transformed p-values for each channel and time point.
        save_to (Optional[Path], optional): Path to save the generated plot. If None, the plot is not saved. Default is None.
        overwrite (bool, optional): If True, overwrite the existing file if it exists. Default is True.

    Raises:
        FileExistsError: If the file already exists at save_to and overwrite is set to False.

    Notes:
        The function generates two subplots:

        - The first subplot shows the correlation coefficients over latencies for the top five channels.
        - The second subplot shows the corresponding p-values for these channels.
    """

    figure, axis = pyplot.subplots(1, 2, figsize=(15, 7))
    figure.suptitle(
        f"{function.name}: Plotting corrs and pvalues for top five channels"
    )

    maxs = np.min(log_pvalues, axis=1)
    n_amaxs = 5
    amaxs = np.argpartition(maxs, -n_amaxs)[-n_amaxs:]
    amax = np.argmin(log_pvalues) // (n_samples_per_split // 2)
    amaxs = [i for i in amaxs if i != amax]  # + [209]

    axis[0].plot(latencies, np.mean(corrs[amax, 0], axis=-2).T, "r-", label=amax)
    axis[0].plot(latencies, np.mean(corrs[amaxs, 0], axis=-2).T, label=amaxs)
    std_null = (
        np.mean(np.std(corrs[:, 1], axis=-2), axis=0).T * 3 / np.sqrt(n_reps * n_splits)
    )  # 3 pop std.s
    std_real = np.std(corrs[amax, 0], axis=-2).T * 3 / np.sqrt(n_reps * n_splits)
    av_real = np.mean(corrs[amax, 0], axis=-2).T

    axis[0].fill_between(latencies, -std_null, std_null, alpha=0.5, color="grey")
    axis[0].fill_between(
        latencies, av_real - std_real, av_real + std_real, alpha=0.25, color="red"
    )

    peak_lat_ind = np.argmin(log_pvalues) % (n_samples_per_split // 2)
    peak_lat = latencies[peak_lat_ind]
    peak_corr = np.mean(corrs[amax, 0], axis=-2)[peak_lat_ind]
    print(
        f"{function.name}: peak lat: {peak_lat:.1f},   peak corr: {peak_corr:.4f}   [sensor] ind: {amax},   -log(pval): {-log_pvalues[amax][peak_lat_ind]:.4f}"
    )

    auto_corrs = np.mean(auto_corrs, axis=0)
    axis[0].plot(
        latencies,
        np.roll(auto_corrs, peak_lat_ind) * peak_corr / np.max(auto_corrs),
        "k--",
        label="func auto-corr",
    )

    axis[0].axvline(0, color="k")
    axis[0].legend()
    axis[0].set_title("Corr coef.")
    axis[0].set_xlabel("latencies (ms)")
    axis[0].set_ylabel("Corr coef.")

    axis[1].plot(latencies, -log_pvalues[amax].T, "r-", label=amax)
    axis[1].plot(latencies, -log_pvalues[amaxs].T, label=amaxs)
    axis[1].axvline(0, color="k")
    axis[1].legend()
    axis[1].set_title("p-values")
    axis[1].set_xlabel("latencies (ms)")
    axis[1].set_ylabel("p-values")

    if save_to is not None:
        pyplot.rcParams["savefig.dpi"] = 300
        save_to = Path(save_to, function.name + "_gridsearch_top_five_channels.png")

        if overwrite or not save_to.exists():
            pyplot.savefig(Path(save_to))
        else:
            raise FileExistsError(save_to)

    pyplot.clf()
    pyplot.close()


def legend_display_dict(functions: list[str], display_name) -> dict[str, str]:
    """
    Creates a dictionary for the `legend_display` parameter of `expression_plot()`.

    This function maps each function name in the provided list to a single display name,
    which can be used to group multiple functions under one legend item in the plot.

    Args:
        functions (list[str]): A list of function names to be grouped under the same display name.
        display_name (str): The display name to be used for all functions in the list.

    Returns:
        dict[str, str]: A dictionary mapping each function name to the provided display name.
    """
    return {function: display_name for function in functions}
