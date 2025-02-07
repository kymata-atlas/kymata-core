from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from statistics import NormalDist
from typing import Optional, Sequence, NamedTuple, Any, Type, Literal, Callable
from warnings import warn

import numpy as np
from matplotlib import pyplot
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
from mne import SourceEstimate
from numpy.typing import NDArray
from pandas import DataFrame
from seaborn import color_palette

from kymata.entities.datatypes import TransformNameDType
from kymata.entities.expression import (
    HexelExpressionSet, SensorExpressionSet, ExpressionSet, DIM_TRANSFORM, COL_LOGP_VALUE, DIM_LATENCY)
from kymata.entities.transform import Transform
from kymata.math.p_values import p_to_logp
from kymata.math.rounding import round_down, round_up
from kymata.plot.color import DiscreteListedColormap
from kymata.plot.layouts import (
    get_meg_sensor_xy, get_eeg_sensor_xy, get_meg_sensors, get_eeg_sensors)

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
    expression_set_type: Type[ExpressionSet],
    fig_size: tuple[float, float],
) -> _MosaicSpec:
    # Set defaults:
    if minimap_option is None:
        width_ratios = None
        height_ratios = None
        subplots_adjust = {
            "hspace": 0,
            "left": 0.08,
            "right": 0.84,
        }
    elif minimap_option.lower() == "standard":
        width_ratios = [1, 3]
        height_ratios = None
        subplots_adjust = {
            "hspace": 0,
            "wspace": 0.25,
            "left": 0.02,
            "right": 0.8,
        }
    elif minimap_option.lower() == "large":
        width_ratios = None
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
                [_AxName.top],
                [_AxName.bottom],
            ]
        elif minimap_option.lower() == "standard":
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
        elif minimap_option.lower() == "large":
            if expression_set_type == HexelExpressionSet:
                spec = [
                    [_AxName.minimap_top, _AxName.minimap_bottom],
                    [_AxName.top,         _AxName.top],
                    [_AxName.bottom,      _AxName.bottom]
                ]
            elif expression_set_type == SensorExpressionSet:
                spec = [
                    [_AxName.minimap_main],
                    [_AxName.top],
                    [_AxName.bottom],
                ]
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    else:
        if minimap_option is None:
            spec = [
                [_AxName.main],
            ]
        elif minimap_option.lower() == "standard":
            spec = [
                [_AxName.minimap_main, _AxName.main],
            ]
        elif minimap_option.lower() == "large":
            spec = [
                [_AxName.minimap_main],
                [_AxName.main],
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


def _hexel_minimap_data(expression_set: HexelExpressionSet,
                        alpha_logp: float,
                        show_transforms: list[str],
                        value_lookup: dict[str, int | float],
                        minimap_latency_range: Optional[tuple[float | None, float | None]] = None,
                        ) -> tuple[NDArray, NDArray]:
    """
    Generates data arrays for a minimap visualization of significant hexels in a HexelExpressionSet.

    Args:
        expression_set (HexelExpressionSet): The set of hexel expressions to analyze.
        alpha_logp (float): The logarithm of the p-value threshold for significance. Hexels with values
            below this threshold are considered significant.
        show_transforms (list[str]): A list of transform names to consider for significance. Only these
            transforms will be checked for significant hexels.
        value_lookup (dict[str, int]): A dictionary mapping transform names to values to set in the data arrays.
        minimap_latency_range: tuple[float | None, float | None]: The latency range to use in the minimap.
            Defaults to None.

    Returns:
        tuple[NDArray, NDArray]: A tuple containing two arrays (one for the left hemisphere and one for
        the right hemisphere). Each array has a length equal to the number of hexels in the respective
        hemisphere, with entries:
            - 0, if no transform is ever significant for this hexel.
            - i+1, where i is the index of the transform (within `show_transforms`) that is significant
              for this hexel.

    Notes:
        This function identifies which hexels are significant for the given transforms based on a provided
        significance threshold. It returns arrays for both the left and right hemispheres, where each entry
        indicates whether the hexel is significant for any transform and, if so, which transform it is
        significant for.
    """
    # Initialise with zeros: transparent everywhere
    data_left = np.zeros((len(expression_set.hexels_left),))
    data_right = np.zeros((len(expression_set.hexels_right),))

    # Get best transforms which survive alpha threshold
    best_transforms_left, best_transforms_right = expression_set.best_transforms()
    best_transforms_left = best_transforms_left[best_transforms_left[COL_LOGP_VALUE] < alpha_logp]
    best_transforms_right = best_transforms_right[best_transforms_right[COL_LOGP_VALUE] < alpha_logp]

    # Apply latency range if appropriate
    if minimap_latency_range != (None, None):
        # Filter the dataframe to keep rows where 'latency' is within the range
        best_transforms_left = best_transforms_left[(best_transforms_left['latency'] >= minimap_latency_range[0]) &
                                                    (best_transforms_left['latency'] <= minimap_latency_range[1])]
        best_transforms_right = best_transforms_right[(best_transforms_right['latency'] >= minimap_latency_range[0]) &
                                                      (best_transforms_right['latency'] <= minimap_latency_range[1])]

    # Apply colour index to each shown transform
    for transform in show_transforms:
        significant_hexel_names_left = best_transforms_left[
                best_transforms_left[DIM_TRANSFORM] == transform
            ][expression_set.channel_coord_name]
        hexel_idxs_left = np.searchsorted(
            expression_set.hexels_left, significant_hexel_names_left.to_numpy()
        )
        data_left[hexel_idxs_left] = value_lookup[transform]

        significant_hexel_names_right = best_transforms_right[
                best_transforms_right[DIM_TRANSFORM] == transform
            ][expression_set.channel_coord_name]
        hexel_idxs_right = np.searchsorted(
            expression_set.hexels_right, significant_hexel_names_right.to_numpy()
        )
        data_right[hexel_idxs_right] = value_lookup[transform]

    return data_left, data_right


def _plot_transform_expression_on_axes(
    ax: pyplot.Axes,
    transform_data: DataFrame,
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
    x = transform_data[DIM_LATENCY].values * 1000  # Convert to milliseconds
    y = transform_data[COL_LOGP_VALUE].values
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
    minimap_latency_range: Optional[tuple[float | None, float | None]] = None,
):
    raise NotImplementedError("Minimap not yet implemented for sensor data")


def _plot_minimap_hexel(
    expression_set: HexelExpressionSet,
    show_transforms: list[str],
    lh_minimap_axis: pyplot.Axes,
    rh_minimap_axis: pyplot.Axes,
    view: str,
    surface: str,
    colors: dict[str, Any],
    alpha_logp: float,
    minimap_kwargs: dict,
    minimap_latency_range: Optional[tuple[float | None, float | None]] = None,
):
    # Ensure we have the FSAverage dataset downloaded
    from kymata.datasets.fsaverage import FSAverageDataset

    fsaverage = FSAverageDataset(download=True)
    os.environ["SUBJECTS_DIR"] = str(fsaverage.path)

    # There is a little circular dependency between the smoothing_steps in plot_kwargs below, which gets created after
    # the colormap, and the colormap, which depends on the smoothing steps. So we short-circuit that here by pulling out
    # the relevant value, if it's there so it doesn't get out of sync
    if "smoothing_steps" in minimap_kwargs:
        smoothing_steps = minimap_kwargs["smoothing_steps"]
    else:
        # Default value
        smoothing_steps = 2

    colormap, colour_idx_lookup, n_colors = _get_colormap_for_cortical_minimap(colors, show_transforms)

    data_left, data_right = _hexel_minimap_data(
        expression_set,
        alpha_logp=alpha_logp,
        show_transforms=show_transforms,
        value_lookup=colour_idx_lookup,
        minimap_latency_range=minimap_latency_range,
    )
    stc = SourceEstimate(
        data=np.concatenate([data_left, data_right]),
        vertices=[expression_set.hexels_left, expression_set.hexels_right],
        tmin=0,
        tstep=1,
    )
    warn("Plotting on the fsaverage brain. Ensure that hexel numbers match those of the fsaverage brain.")
    plot_kwargs = dict(
        subject="fsaverage",
        surface=surface,
        views=view,
        colormap=colormap,
        smoothing_steps=smoothing_steps,
        cortex=dict(colormap="Greys", vmin=-3, vmax=6),
        background="white",
        spacing="ico5",
        time_viewer=False,
        colorbar=False,
        transparent=False,
        clim=dict(
            kind="value",
            lims=[0, n_colors / 2, n_colors],
        ),
    )
    # Override plot kwargs with those passed
    plot_kwargs.update(minimap_kwargs)
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


def _get_colormap_for_cortical_minimap(colors: dict[str, Any],
                                       show_transforms: list[str],
                                       ) -> tuple[Callable, dict[TransformNameDType, int], int]:
    """
    Get a colormap appropriate for displaying transforms on a brain minimap.

    Colormap will have specified colours for the transforms, interleaved with transparency. The transparency
    interleaving is necessary to remove false-colour "halos" appearing around the edge of significant hexel patches when
    smoothing is >1.

    Index point to transform position within `show_transforms` (1-indexed).

    Args:
        colors (dict): A dictionary mapping transform names to colours (in any matplotlib-appropriate format, e.g.
            strings ("red", "#2r4fa6") or rgb(a) tuples ((1.0, 0.0, 0.0, 1.0))
        show_transforms (list[str]): The transforms which will be shown

    Returns: Tuple of the following items
        (
            LinearSegmentedColormap: Colormap with colours for shown functions interleaved with transparency
            dict[TransformNameDType, int]: Dictionary mapping transform names to indices within the colourmap for the
                appropriate colour
            int: the total number of colours in the colourmap, including the initial and interleaved transparency
        )

    """

    base_colours = [transparent] + [colors[f] for f in show_transforms]

    colour_idx_lookup = {
        # Map the colour to its corresponding index in the colourmap
        TransformNameDType(transform): transform_idx
        for transform_idx, transform in enumerate(show_transforms, start=1)
    }

    colormap = DiscreteListedColormap(colors=base_colours, scale01=True)

    return colormap, colour_idx_lookup, max(colour_idx_lookup.values())


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
    hidden_transforms_in_legend: bool = True,
    title: str = None,
    fig_size: tuple[float, float] = (12, 7),
    # Display options
    minimap: str | None = None,
    minimap_view: str = "lateral",
    minimap_surface: str = "inflated",
    show_only_sensors: Optional[Literal["eeg", "meg"]] = None,
    minimap_latency_range: Optional[tuple[float | None, float | None]] = None,
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
        title (str, optional): Title over the top axis in the figure. Default is None.
        fig_size (tuple[float, float], optional): Figure size in inches. Default is (12, 7).
        minimap (str, optional): If None, no minimap is shown. Other options are:
            `"standard"`: Show small minimal.
            `"large"`: Show a large minimal with smaller expression plot.
            Default is None.
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
        minimap_latency_range (Optional[tuple[float | None, float | None]]): Supply `(start_time, stop_time)` to restrict
            minimap view to only the specified time window, and highlight the time window on the expression plot.
            Both `start_time` and `stop_time` are in seconds. Set `start_time` or `stop_time` to `None` for half-open
            intervals.
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
            # Wrap into list
            best_transforms = (best_transforms,)
        else:
            raise NotImplementedError()

    if isinstance(expression_set, HexelExpressionSet):
        n_channels = len(expression_set.hexels_left) + len(expression_set.hexels_right)
    elif isinstance(expression_set, SensorExpressionSet):
        n_channels = len(expression_set.sensors)
    else:
        raise NotImplementedError()

    chosen_channels = _restrict_channels(
        expression_set, best_transforms, show_only_sensors
    )

    sidak_corrected_alpha = 1 - (
        (1 - alpha)
        ** np.longdouble(1 / (2 * len(expression_set.latencies) * n_channels * len(show_only)))
    )

    sidak_corrected_alpha = p_to_logp(sidak_corrected_alpha)

    def _custom_label(transform_name):
        if legend_display is not None:
            if transform_name in legend_display.keys():
                return legend_display[transform_name]
        return transform_name

    mosaic = _minimap_mosaic(
        paired_axes=paired_axes,
        minimap_option=minimap,
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
    for transform in show_only:
        custom_label = _custom_label(transform)
        if custom_label not in custom_labels:
            custom_handles.extend(
                [Line2D([], [], marker=".", color=color[transform], linestyle="None")]
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
                expression_axes_list, best_transforms, (top_chans, bottom_chans)
            ):
                # Plot filled
                (
                    x_min,
                    x_max,
                    y_min,
                    _y_max,
                ) = _plot_transform_expression_on_axes(
                    transform_data=best_funs_this_ax[
                        (best_funs_this_ax[DIM_TRANSFORM] == transform)
                        & (
                            best_funs_this_ax[expression_set.channel_coord_name].isin(
                                chans_this_ax
                            )
                        )
                    ],
                    color=color[transform],
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
                ) = _plot_transform_expression_on_axes(
                    transform_data=best_funs_this_ax[
                        (best_funs_this_ax[DIM_TRANSFORM] == transform)
                        & (
                            best_funs_this_ax[expression_set.channel_coord_name].isin(
                                both_chans
                            )
                        )
                    ],
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
            for ax, best_funs_this_ax in zip(expression_axes_list, best_transforms):
                (
                    x_min,
                    x_max,
                    y_min,
                    _y_max,
                ) = _plot_transform_expression_on_axes(
                    transform_data=best_funs_this_ax[
                        (best_funs_this_ax[DIM_TRANSFORM] == transform)
                        & (
                            best_funs_this_ax[expression_set.channel_coord_name].isin(
                                chosen_channels
                            )
                        )
                    ],
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
        if isinstance(expression_set, SensorExpressionSet):
            _plot_minimap_sensor(
                expression_set,
                minimap_axis=axes[_AxName.minimap_main],
                colors=color,
                alpha_logp=sidak_corrected_alpha,
                minimap_latency_range=minimap_latency_range,
            )
        elif isinstance(expression_set, HexelExpressionSet):
            _plot_minimap_hexel(
                expression_set,
                show_transforms=show_only,
                lh_minimap_axis=axes[_AxName.minimap_top],
                rh_minimap_axis=axes[_AxName.minimap_bottom],
                view=minimap_view,
                surface=minimap_surface,
                colors=color,
                alpha_logp=sidak_corrected_alpha,
                minimap_latency_range=minimap_latency_range,
                minimap_kwargs=minimap_kwargs,
            )
        else:
            raise NotImplementedError()

    # Format one-off axis qualities
    top_ax: pyplot.Axes
    bottom_ax: pyplot.Axes
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

    # Legend for plotted transform
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
            bottom_legend = bottom_ax.legend(
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
        top_legend = top_ax.legend(
            handles=custom_handles,
            labels=custom_labels,
            fontsize="x-small",
            alignment="left",
            title="Plotted transforms",
            ncol=legend_n_col,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.02),
        )
        legends.append(top_legend)

    __reposition_axes_for_legends(fig, legends)

    __add_axis_name_annotations(axes_names, top_ax, bottom_ax, fig, paired_axes, ylim, minimap)

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
    max_axes_right_inches = max(ax.get_position().xmax for ax in fig.get_axes())
    max_extent_inches = (max(leg.get_window_extent().xmax / fig.dpi
                             for leg in legends)
                         # Plus a bit extra because matplotlib
                         + buffer
                         if len(legends) > 0
                         else max_axes_right_inches)
    fig.subplots_adjust(right=(fig.get_figwidth()
                               * max_axes_right_inches
                               / max_extent_inches))


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


def _restrict_channels(
    expression_set: ExpressionSet,
    best_transforms: tuple[DataFrame, ...],
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
                for best_funs_each_ax in best_transforms
                for sensor in best_funs_each_ax[expression_set.channel_coord_name]
            }
        elif isinstance(expression_set, HexelExpressionSet):
            # All hexels
            chosen_channels = {
                sensor
                for best_funs_each_ax in best_transforms
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
    transform: Transform,
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
        transform (Transform): The transform object whose name attribute will be used in the plot title.
        n_samples_per_split (int): Number of samples per split used in the grid search.
        n_reps (int): Number of repetitions in the grid search.
        n_splits (int): Number of splits in the grid search.
        auto_corrs (NDArray[any]): Auto-correlation values array used for plotting the transform auto-correlation.
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
        f"{transform.name}: Plotting corrs and pvalues for top five channels"
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
        f"{transform.name}: peak lat: {peak_lat:.1f},   peak corr: {peak_corr:.4f}   [sensor] ind: {amax},   -log(pval): {-log_pvalues[amax][peak_lat_ind]:.4f}"
    )

    auto_corrs = np.mean(auto_corrs, axis=0)
    axis[0].plot(
        latencies,
        np.roll(auto_corrs, peak_lat_ind) * peak_corr / np.max(auto_corrs),
        "k--",
        label="trans auto-corr",
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
        save_to = Path(save_to, transform.name + "_gridsearch_top_five_channels.png")

        if overwrite or not save_to.exists():
            pyplot.savefig(Path(save_to))
        else:
            raise FileExistsError(save_to)

    pyplot.clf()
    pyplot.close()


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
