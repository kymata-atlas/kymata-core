from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Optional, Sequence, Type, Literal
from warnings import warn

import numpy as np
from matplotlib import pyplot
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
from seaborn import color_palette

from kymata.entities.expression import ExpressionPoint, HexelExpressionSet, SensorExpressionSet, ExpressionSet
from kymata.io.layouts import SensorLayout, MEGLayout, EEGLayout
from kymata.math.probability import p_to_logp, sidak_correct, p_threshold_for_sigmas
from kymata.math.rounding import round_up, round_down
from kymata.plot.brain import plot_minimap_hexel
from kymata.plot.sensor import get_sensor_left_right_assignment, restrict_sensors_by_type

# log scale: 10 ** -this will be the ytick interval and also the resolution to which the ylims will be rounded
_MAJOR_TICK_SIZE = 50


class _AxName:
    """Canonical names for the axes."""
    expression_top_lh = "expression top lh"
    expression_bottom_rh = "expression bottom rh"
    expression_main = "expression main"
    minimap_lh = "minimap lh"
    minimap_rh = "minimap rh"
    minimap_main = "minimap main"


@dataclass
class _MosaicSpec:
    mosaic: list[list[str]]
    width_ratios: list[float] | None
    height_ratios: list[float] | None
    fig_size: tuple[float, float]
    subplots_adjust_kwargs: dict[str, float] = None

    def __post_init__(self):
        if self.subplots_adjust_kwargs is None:
            self.subplots_adjust_kwargs = dict()

    def to_subplots(self) -> tuple[pyplot.Figure, dict[str, pyplot.Axes]]:
        print(self.mosaic)
        return pyplot.subplot_mosaic(
            self.mosaic, width_ratios=self.width_ratios, height_ratios=self.height_ratios, figsize=self.fig_size
        )


def _minimap_mosaic(
    paired_axes: bool,
    minimap_option: str | None,
    minimap_type: str,
    expression_set_type: Type[ExpressionSet],
    fig_size: tuple[float, float],
) -> _MosaicSpec:

    # Convert keywords to lowercase
    if minimap_option is not None:
        minimap_option = minimap_option.lower()
    minimap_type = minimap_type.lower()

    # Set defaults for other parameters:
    if minimap_option is None:
        # No minimap
        width_ratios = None
        height_ratios = None
        subplots_adjust = {
            "hspace": 0,
            "left": 0.08,
            "right": 0.84,
        }
    elif minimap_option == "standard":
        width_ratios = [1, 3]
        height_ratios = None
        subplots_adjust = {
            "hspace": 0,
            "wspace": 0.25,
            "left": 0.02,
            "right": 0.8,
        }
    elif minimap_option == "large":
        width_ratios = None
        if minimap_type == "volumetric":
            height_ratios = [3, 1, 1]
        else:
            height_ratios = [6, 1, 1]
        subplots_adjust = {
            "hspace": 0,
            "wspace": 0.1,
            "left": 0.08,
            "right": 0.92,
        }
    else:
        raise NotImplementedError()

    if paired_axes:
        if minimap_option is None:
            spec = [
                [_AxName.expression_top_lh],
                [_AxName.expression_bottom_rh],
            ]
        elif minimap_option == "standard":
            if expression_set_type == HexelExpressionSet:
                if minimap_type == "volumetric":
                    # Volumetric minimaps have L, main, and R views
                    spec = [
                        [_AxName.minimap_lh,   _AxName.expression_top_lh],
                        [_AxName.minimap_lh,   _AxName.expression_top_lh],
                        [_AxName.minimap_main, _AxName.expression_top_lh],
                        [_AxName.minimap_main, _AxName.expression_bottom_rh],
                        [_AxName.minimap_rh,   _AxName.expression_bottom_rh],
                        [_AxName.minimap_rh,   _AxName.expression_bottom_rh],
                    ]
                else:
                    # Cortical minimaps have only L and R views
                    spec = [
                        [_AxName.minimap_lh, _AxName.expression_top_lh],
                        [_AxName.minimap_rh, _AxName.expression_bottom_rh],
                    ]
            elif expression_set_type == SensorExpressionSet:
                spec = [
                    [_AxName.minimap_main, _AxName.expression_top_lh],
                    [_AxName.minimap_main, _AxName.expression_bottom_rh],
                ]
            else:
                raise NotImplementedError()
        elif minimap_option == "large":
            if expression_set_type == HexelExpressionSet:
                if minimap_type == "volumetric":
                    # Volumetric minimaps have L, main, and R views
                    spec = [
                        [_AxName.minimap_lh] * 2 + [_AxName.minimap_main] * 2 + [_AxName.minimap_rh] * 2,
                        [_AxName.expression_top_lh] * 6,
                        [_AxName.expression_bottom_rh] * 6
                    ]
                else:
                    # Cortical minimaps have only L and R views
                    spec = [
                        [_AxName.minimap_lh, _AxName.minimap_rh],
                        [_AxName.expression_top_lh] * 2,
                        [_AxName.expression_bottom_rh] * 2
                    ]
            elif expression_set_type == SensorExpressionSet:
                spec = [
                    [_AxName.minimap_main],
                    [_AxName.expression_top_lh],
                    [_AxName.expression_bottom_rh],
                ]
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    else:
        if minimap_option is None:
            spec = [
                [_AxName.expression_main],
            ]
        elif minimap_option == "standard":
            spec = [
                [_AxName.minimap_main, _AxName.expression_main],
            ]
        elif minimap_option == "large":
            spec = [
                [_AxName.minimap_main],
                [_AxName.expression_main],
            ]
        else:
            raise NotImplementedError()

    return _MosaicSpec(
        mosaic=spec,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        fig_size=fig_size,
        subplots_adjust_kwargs=subplots_adjust,
    )


def _plot_transform_expression_on_axes(
    ax: pyplot.Axes,
    transform_data: list[ExpressionPoint],
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
    x = np.array([ep.latency * 1000   # Convert to milliseconds
                  for ep in transform_data])
    y = np.array([ep.logp_value
                  for ep in transform_data])
    c = np.where(y <= sidak_corrected_alpha, color, "black")
    ax.vlines(x=x, ymin=1, ymax=y, color=c)
    ax.scatter(x, y, facecolors=c if filled else "none", s=20, edgecolors=c)

    x_min = x.min() if len(x) > 0 else np.Inf
    x_max = x.max() if len(x) > 0 else -np.Inf
    # Careful, the y value is inverted, with y==1 on the origin and y<1 away from the origin.
    # "y_min" here is real absolute min value in the data (closest to zero)
    y_min = y.min() if len(y) > 0 else np.Inf
    y_max = y.max() if len(y) > 0 else -np.Inf

    return x_min, x_max, y_min, y_max


def expression_plot(
    expression_set: ExpressionSet,
    show_only: Optional[str | Sequence[str]] = None,
    paired_axes: bool = True,
    # Statistical kwargs
    alpha: float = p_threshold_for_sigmas(5),
    # Style kwargs
    color: Optional[str | dict[str, str] | list[str]] = None,
    ylim: Optional[float] = None,
    xlims: Optional[tuple[float | None, float | None]] = None,
    hidden_transforms_in_legend: bool = True,
    title: Optional[str] = None,
    fig_size: tuple[float, float] = (12, 7),
    # Display options
    minimap: str | None = None,
    minimap_type: str = "inflated",
    minimap_view: Optional[str] = None,
    show_only_sensors: Optional[Literal["eeg", "meg"]] = None,
    minimap_latency_range: Optional[tuple[float | None, float | None]] = None,
    plot_top_n: Optional[int] = None,
    # I/O args
    save_to: Optional[Path] = None,
    overwrite: bool = True,
    show_legend: bool = True,
    legend_display: dict[str, str] | None = None,
    # Extra kwargs
    minimap_kwargs: Optional[dict] = None
) -> pyplot.Figure:
    """
    Generates a plot of transform expressions over time with optional display customizations.

    Args:
        expression_set (ExpressionSet): The set of expressions to plot, containing transforms and associated data.
        show_only (Optional[str | Sequence[str]], optional): A string or a sequence of strings specifying which
            transforms to plot.
            If None, all transforms in the expression_set will be plotted. Default is None.
        paired_axes (bool, optional): When True, shows the expression plot split into left and right axes.
            When False, all points are shown on the same axis. Default is True.
        alpha (float, optional): Significance level for statistical tests, defaulting to a 5-sigma threshold.
        color (Optional[str | dict[str, str] | list[str]], optional): Color settings for the plot. Can be a single
            color, a dictionary mapping transform names to colors, or a list of colors. Default is None.
        ylim (Optional[float], optional): The y-axis limit (p-value). Use log10 of the desired value — e.g. if the
            desired limit is 10^-100, supply ylim=-100. If None, it will be determined automatically. Default is None.
        xlims (tuple[Optional[float], Optional[float]], optional): The x-axis limits as a tuple (in ms). None to use
            default values, or set either entry to None to use the default for that value. Default is (-100, 800).
        hidden_transforms_in_legend (bool, optional): If True, includes non-plotted transforms in the legend.
            Default is True.
        title (str, optional): Title over the top axis in the figure. Supply None for no title. Default is None.
        fig_size (tuple[float, float], optional): Figure size in inches. Default is (12, 7).
        minimap (str, optional): If None, no minimap is shown. Other options are:
            `"standard"`: Show small minimal.
            `"large"`: Show a large minimal with smaller expression plot.
            Default is None.
        minimap_type (str, optional): The type of minimap to display. Options include:
            `"inflated"`, or any other valid keyword for a mne surface: This will plot on a 3D cortical mesh.
            `"volumetric"`: plot in a volumetric view à la MRICron.
            Default is "inflated".
        minimap_view (Optional[str]): The view type for the minimap, either "lateral" or other specified views.
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
            Only `"axial"` and `"coronal"` are valid for volumetric view; sagittal views will always be plotted.
            Default is None (lateral for cortical, axial for volumetric).
        show_only_sensors (str, optional): Show only one type of sensors. "meg" for MEG sensors, "eeg" for EEG sensors.
            None to show all sensors. Supplying this value with something other than a SensorExpressionSet causes will
            throw an exception. Default is None.
        minimap_latency_range (Optional[tuple[float | None, float | None]]): Supply `(start_time, stop_time)` to
            restrict the minimap view to only the specified time window, and highlight the time window on the expression
            plot. Both `start_time` and `stop_time` are in seconds. Set `start_time` or `stop_time` to `None` for
            half-open intervals.
        plot_top_n (Optional[int]): If not None, show only the N most significant sources. If None, plot all significant
            sources. Default is None.
        save_to (Optional[Path], optional): Path to save the generated plot. If None, the plot is not saved.
            Default is None.
        overwrite (bool, optional): If True, overwrite the existing file if it exists. Default is True.
        show_legend (bool, optional): If True, displays the legend. Default is True.
        legend_display (dict[str, str] | None, optional): Allows grouping of multiple transforms under the same legend
            item. Provide a dictionary mapping true transform names to display names. None applies no grouping.
            Default is None.
        minimap_kwargs (Optional[dict]): Keyword argument overrides for minimap plotting. Default is None.

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
        show_only = expression_set.transforms
    elif isinstance(show_only, str):
        show_only = [show_only]
    not_shown = [f for f in expression_set.transforms if f not in show_only]

    if color is None:
        color = dict()
    elif isinstance(color, str):
        # Single string specified: use all that colour
        color = {f: color for f in show_only}
    elif isinstance(color, list):
        # List specified, then pair up in order
        assert len(color) == len(show_only)
        color = {f: c for f, c in zip(show_only, color)}

    if minimap_latency_range is None:
        minimap_latency_range = (None, None)
    assert len(minimap_latency_range) == 2

    if minimap_kwargs is None:
        minimap_kwargs = dict()

    # Default colours
    cycol = cycle(color_palette("Set1"))
    for transform in show_only:
        if transform not in color:
            color[transform] = to_hex(next(cycol))

    # Default views
    if minimap_view is None:
        if minimap_type == "volumetric":
            minimap_view = "axial"
        else:
            minimap_view = "lateral"

    if plot_top_n is not None:
        if plot_top_n < 1:
            raise ValueError("`plot_top_n` must be greater than or equal to 1")

    best_transforms = expression_set.best_transforms()

    if paired_axes:
        if isinstance(expression_set, HexelExpressionSet):
            axes_names = ("left hemisphere", "right hemisphere")
            assert isinstance(best_transforms, tuple)
        elif isinstance(expression_set, SensorExpressionSet):
            axes_names = ("left", "right")
            # Same transforms passed, filtering done at channel level
            best_transforms = (best_transforms, best_transforms)
        else:
            raise NotImplementedError()
    else:
        if isinstance(expression_set, HexelExpressionSet):
            raise NotImplementedError(
                "HexelExpressionSets have preset hemisphere assignments"
            )
        elif isinstance(expression_set, SensorExpressionSet):
            axes_names = ("",)
            # Wrap into tuple
            best_transforms = (best_transforms,)
        else:
            raise NotImplementedError()

    if isinstance(expression_set, HexelExpressionSet):
        n_channels = len(expression_set.hexels_left) + len(expression_set.hexels_right)
    elif isinstance(expression_set, SensorExpressionSet):
        n_channels = len(expression_set.sensors)
    else:
        raise NotImplementedError()

    chosen_channels = restrict_sensors_by_type(expression_set, best_transforms, show_only_sensors)

    sidak_corrected_alpha = sidak_correct(alpha, n_comparisons=len(expression_set.latencies) * n_channels * len(show_only))
    sidak_corrected_alpha = p_to_logp(sidak_corrected_alpha)

    def _custom_label(transform_name):
        if legend_display is not None:
            if transform_name in legend_display.keys():
                return legend_display[transform_name]
        return transform_name

    mosaic = _minimap_mosaic(
        paired_axes=paired_axes,
        minimap_option=minimap,
        minimap_type=minimap_type,
        expression_set_type=type(expression_set),
        fig_size=fig_size,
    )

    fig: pyplot.Figure
    axes: dict[str, pyplot.Axes]
    fig, axes = mosaic.to_subplots()

    expression_axes_list: list[pyplot.Axes]
    if paired_axes:
        expression_axes_list = [
            axes[_AxName.expression_top_lh],
            axes[_AxName.expression_bottom_rh],
        ]  # For iterating over in a predictable order
    else:
        expression_axes_list = [axes[_AxName.expression_main]]

    fig.subplots_adjust(**mosaic.subplots_adjust_kwargs)

    # Turn off autoscaling to make plotting faster, and since we manually set the axes scale later
    pyplot.autoscale(False)

    custom_handles = []
    custom_labels = []
    data_x_min, data_x_max = np.Inf, -np.Inf
    data_y_min = np.Inf
    for transform in show_only:
        custom_label = _custom_label(transform)
        if custom_label not in custom_labels:
            custom_handles.extend([Line2D([], [], marker=".", color=color[transform], linestyle="None")])
            custom_labels.append(custom_label)

        # We have a special case with paired sensor data, in that some sensors need to appear
        # on both sides of the midline.
        if paired_axes and isinstance(expression_set, SensorExpressionSet):
            if expression_set.sensor_layout is not None:
                sensor_layout = expression_set.sensor_layout
            else:
                sensor_layout = SensorLayout(meg=MEGLayout.Vectorview, eeg=EEGLayout.Easycap)
                warn(f"SensorExpressionSet had no sensor layout for assignment of sensors to left/right. "
                     f"Attempting to use default value {sensor_layout}, but beware of display issues.")
            assign_left_right_channels = get_sensor_left_right_assignment(sensor_layout)

            # Some points will be plotted on one axis, filled, some on both, empty
            top_chans = set(assign_left_right_channels[0].axis_channels) & chosen_channels
            bottom_chans = set(assign_left_right_channels[1].axis_channels) & chosen_channels
            # Symmetric difference
            both_chans = top_chans & bottom_chans
            top_chans -= both_chans
            bottom_chans -= both_chans
            for ax, best_trans_this_ax, chans_this_ax in zip(
                expression_axes_list, best_transforms, (top_chans, bottom_chans)
            ):
                # Plot filled
                (x_min, x_max, y_min, _y_max) = _plot_transform_expression_on_axes(
                    transform_data=[ep
                                    for ep in best_trans_this_ax
                                    if ep.transform == transform
                                    and ep.channel in chans_this_ax],
                    color=color[transform],
                    ax=ax,
                    sidak_corrected_alpha=sidak_corrected_alpha,
                    filled=True,
                )
                data_x_min = min(data_x_min, x_min)
                data_x_max = max(data_x_max, x_max)
                data_y_min = min(data_y_min, y_min)
                # Plot empty
                (x_min, x_max, y_min, _y_max) = _plot_transform_expression_on_axes(
                    transform_data=[ep
                                    for ep in best_trans_this_ax
                                    if ep.transform == transform
                                    and ep.channel in both_chans],
                    color=color[transform],
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
            for ax, best_trans_this_ax in zip(expression_axes_list, best_transforms):
                (
                    x_min,
                    x_max,
                    y_min,
                    _y_max,
                ) = _plot_transform_expression_on_axes(
                    transform_data=[ep
                                    for ep in best_trans_this_ax
                                    if ep.transform == transform
                                    and ep.channel in chosen_channels],
                    color=color[transform],
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
            xlims[1] - 50,
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

        # Show highlighted range
        if minimap_latency_range != (None, None):
            start, stop = minimap_latency_range
            if start is None:
                start = ax.get_xlim()[0]
            else:
                start = start * 1000  # Convert to ms
            if stop is None:
                stop = ax.get_xlim()[1]
            else:
                stop = stop * 1000  # Convert to ms
            ax.axvspan(xmin=start, xmax=stop, color="grey", alpha=0.2, lw=0, zorder=-10)

    # Plot minimap
    if minimap is not None:
        if isinstance(expression_set, HexelExpressionSet):
            plot_minimap_hexel(
                expression_set,
                show_transforms=show_only,
                lh_minimap_axis=axes[_AxName.minimap_lh],
                rh_minimap_axis=axes[_AxName.minimap_rh],
                main_minimap_axis=axes[_AxName.minimap_main] if _AxName.minimap_main in axes.keys() else None,
                view=minimap_view,
                surface=minimap_type,
                colors=color,
                alpha_logp=sidak_corrected_alpha,
                minimap_latency_range=minimap_latency_range,
                minimap_kwargs=minimap_kwargs,
                top_n=plot_top_n,
            )
        elif isinstance(expression_set, SensorExpressionSet):
            raise NotImplementedError("Minimap not yet implemented for sensor data")
        else:
            raise NotImplementedError()

    # Format one-off axis qualities
    top_expression_ax: pyplot.Axes
    bottom_expression_ax: pyplot.Axes
    if paired_axes:
        top_expression_ax = axes[_AxName.expression_top_lh]
        bottom_expression_ax = axes[_AxName.expression_bottom_rh]
        top_expression_ax.set_xticklabels([])
        bottom_expression_ax.invert_yaxis()
    else:
        top_expression_ax = bottom_expression_ax = axes[_AxName.expression_main]
    top_expression_ax.set_title(title)
    bottom_expression_ax.set_xlabel("Latency (ms) relative to onset of the environment")
    bottom_ax_xmin, bottom_ax_xmax = bottom_expression_ax.get_xlim()
    bottom_expression_ax.xaxis.set_major_locator(
        FixedLocator(_get_xticks((bottom_ax_xmin, bottom_ax_xmax)))
    )

    # Legend for plotted transform
    if minimap == "large":
        main_legend_ax = axes[_AxName.minimap_rh]
    else:
        main_legend_ax = top_expression_ax
    minor_legend_ax = bottom_expression_ax
    legends = []
    if show_legend:
        split_legend_at_n_transforms = 15
        legend_n_col = 2 if len(custom_handles) > split_legend_at_n_transforms else 1
        if hidden_transforms_in_legend and len(not_shown) > 0 and legend_display is None:
            if len(not_shown) > split_legend_at_n_transforms:
                legend_n_col = 2
            # Plot dummy legend for other transforms which are included in model selection but not plotted
            custom_labels_not_shown = []
            dummy_patches = []
            for hidden_transform in not_shown:
                custom_label = _custom_label(hidden_transform)
                if custom_label not in custom_labels_not_shown:
                    custom_labels_not_shown.append(custom_label)
                    dummy_patches.append(Patch(color=None, label=custom_label))
            bottom_legend = minor_legend_ax.legend(
                labels=custom_labels_not_shown,
                fontsize="x-small",
                alignment="left",
                title="Non-plotted transforms",
                ncol=legend_n_col,
                bbox_to_anchor=(1.02, -0.02),
                loc="lower left",
                handles=dummy_patches,
            )
            for lh in bottom_legend.legend_handles:
                lh.set_alpha(0)
            legends.append(bottom_legend)
        main_legend = main_legend_ax.legend(
            handles=custom_handles,
            labels=custom_labels,
            fontsize="x-small",
            alignment="left",
            title="Plotted transforms",
            ncol=legend_n_col,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.02),
        )
        legends.append(main_legend)

    __reposition_axes_for_legends(fig, legends)

    __add_axis_name_annotations(axes_names, top_expression_ax, bottom_expression_ax, fig, paired_axes, ylim, minimap)

    if save_to is not None:
        pyplot.rcParams["savefig.dpi"] = 300
        save_to = Path(save_to)

        if overwrite or not save_to.exists():
            pyplot.savefig(Path(save_to), bbox_inches="tight")
        else:
            raise FileExistsError(save_to)

    return fig


def __reposition_axes_for_legends(fig, legends):
    """Adjust figure width to accommodate legend(s)"""
    buffer = 1.0
    fig_width, _fig_height = fig.get_size_inches()
    max_axes_right_inches = max(ax.get_position().xmax for ax in fig.get_axes()) * fig_width
    max_extent_inches = (max(leg.get_window_extent().xmax / fig.dpi
                             for leg in legends)
                         # Plus a bit extra because matplotlib
                         + buffer
                         if len(legends) > 0
                         else max_axes_right_inches)
    fig.subplots_adjust(right=(max_axes_right_inches / max_extent_inches))


def __add_axis_name_annotations(axes_names: Sequence[str],
                                top_ax: pyplot.Axes, bottom_ax: pyplot.Axes, fig: pyplot.Figure,
                                paired_axes: bool, ylim: float,
                                minimap: str | None):
    bottom_ax_xmin, bottom_ax_xmax = bottom_ax.get_xlim()
    if paired_axes:
        # Label each axis in the pair
        offset_x_abs = 20
        if minimap == "large":
            offset_y_rel = 0.85
        else:
            offset_y_rel = 0.95
        assert len(axes_names) >= 2
        top_ax.text(
            s=axes_names[0],
            x=bottom_ax_xmin + offset_x_abs,
            y=ylim * offset_y_rel,
            style="italic",
            verticalalignment="center",
        )
        bottom_ax.text(
            s=axes_names[1],
            x=bottom_ax_xmin + offset_x_abs,
            y=ylim * offset_y_rel,
            style="italic",
            verticalalignment="center",
        )
    # p-val label
    fig.text(
        x=top_ax.get_position().xmin - 0.06,
        y=top_ax.get_position().ymin,
        s="p-value\n(α at 5-sigma, Šidák corrected)",
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


def legend_display_dict(transforms: list[str], display_name) -> dict[str, str]:
    """
    Creates a dictionary for the `legend_display` parameter of `expression_plot()`.

    This function maps each transform name in the provided list to a single display name,
    which can be used to group multiple transforms under one legend item in the plot.

    Args:
        transforms (list[str]): A list of transform names to be grouped under the same display name.
        display_name (str): The display name to be used for all transforms in the list.

    Returns:
        dict[str, str]: A dictionary mapping each transform name to the provided display name.
    """
    return {transform: display_name for transform in transforms}
