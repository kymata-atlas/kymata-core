import os
from pathlib import Path
from typing import Optional, Any
from warnings import warn

import numpy as np
from matplotlib import pyplot
from matplotlib.colors import Colormap, ListedColormap
from mne import SourceEstimate, read_source_spaces
from numpy.typing import NDArray

from kymata.entities.datatypes import TransformNameDType
from kymata.entities.expression import HexelExpressionSet, ExpressionPoint
from kymata.plot.axes import hide_axes
from kymata.plot.color import transparent
from kymata.plot.compositing import rasterize_as_array
from kymata.plot.mne import plot_bem


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
        minimap_latency_range: (Optional[tuple[float | None, float | None]]): The latency range to use in the minimap.
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
    best_transforms_left: list[ExpressionPoint]
    best_transforms_right: list[ExpressionPoint]
    best_transforms_left, best_transforms_right = expression_set.best_transforms()
    best_transforms_left  = [ep for ep in best_transforms_left  if ep.logp_value < alpha_logp]
    best_transforms_right = [ep for ep in best_transforms_right if ep.logp_value < alpha_logp]
    if minimap_latency_range is not None:
        # Filter expression points to keep those where latency is in range
        minimap_start, minimap_end = minimap_latency_range
        if minimap_start is not None:
            best_transforms_left  = [ep for ep in best_transforms_left  if ep.latency >= minimap_start]
            best_transforms_right = [ep for ep in best_transforms_right if ep.latency >= minimap_start]
        if minimap_end is not None:
            best_transforms_left  = [ep for ep in best_transforms_left  if ep.latency <= minimap_end]
            best_transforms_right = [ep for ep in best_transforms_right if ep.latency <= minimap_end]

    for trans_i, transform in enumerate(
        expression_set.transforms,
        # 1-indexed, as 0 will refer to transparent
        start=1,
    ):
        if transform not in show_transforms:
            continue
        significant_hexel_names_left = [ep.channel for ep in best_transforms_left if ep.transform == transform]
        hexel_idxs_left = np.searchsorted(
            expression_set.hexels_left, np.array(significant_hexel_names_left)
        )
        data_left[hexel_idxs_left] = value_lookup[transform]

        significant_hexel_names_right = [ep.channel for ep in best_transforms_right if ep.transform == transform]
        hexel_idxs_right = np.searchsorted(
            expression_set.hexels_right, np.array(significant_hexel_names_right)
        )
        data_right[hexel_idxs_right] = value_lookup[transform]

    return data_left, data_right


def plot_minimap_hexel(
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
    from kymata.datasets.fsaverage import FSAverageDataset

    # Ensure we have the FSAverage dataset downloaded
    fsaverage = FSAverageDataset(download=True)
    os.environ["SUBJECTS_DIR"] = str(fsaverage.path)

    colormap, colour_idx_lookup = _get_colormap_for_cortical_minimap(colors, show_transforms)

    data_left, data_right = _hexel_minimap_data(expression_set,
                                                alpha_logp=alpha_logp,
                                                show_transforms=show_transforms,
                                                value_lookup=colour_idx_lookup,
                                                minimap_latency_range=minimap_latency_range)
    stc = SourceEstimate(
        data=np.concatenate([data_left, data_right]),
        vertices=[expression_set.hexels_left, expression_set.hexels_right],
        tmin=0,
        tstep=1,
    )

    if surface == "volumetric":
        __plot_minimap_hexel_volumetric(
            stc=stc,
            src_loc=fsaverage.path / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif",
            view=view,
            lh_minimap_axis=lh_minimap_axis,
            rh_minimap_axis=rh_minimap_axis,
            colormap=colormap,
            minimap_kwargs=minimap_kwargs,
        )
    else:
        __plot_minimap_hexel_surface(
            stc=stc,
            view=view,
            surface=surface,
            lh_minimap_axis=lh_minimap_axis,
            rh_minimap_axis=rh_minimap_axis,
            colormap=colormap,
            minimap_kwargs=minimap_kwargs,
        )


def __plot_minimap_hexel_volumetric(
        stc: SourceEstimate,
        src_loc: Path,
        view: str,
        lh_minimap_axis: pyplot.Axes,
        rh_minimap_axis: pyplot.Axes,
        colormap: Colormap,
        minimap_kwargs: dict,
):
    # Select nice-looking slice
    if view == "coronal":
        slice = [100]
    elif view == "sagittal":
        slice = [140]
    elif view == "axial":
        slice = [140]
    else:
        raise ValueError(f"{view} not a valid view. Please select one of coronal, sagittal and axial.")

    src = read_source_spaces(src_loc)

    # Reset src's `'inuse'` with LH and RH vertices
    src[0]["inuse"] = np.zeros_like(src[0]["inuse"])
    src[1]["inuse"] = np.zeros_like(src[1]["inuse"])
    src[0]["inuse"][np.where(stc.lh_data.squeeze() == 1.0)] = 1
    src[1]["inuse"][np.where(stc.rh_data.squeeze() == 1.0)] = 1

    fig = plot_bem(
        subject="fsaverage",
        src=src,
        orientation=view,
        slices=slice,
        show=False,
    )

    lh_minimap_axis.imshow(rasterize_as_array(fig))

    hide_axes(lh_minimap_axis)
    hide_axes(rh_minimap_axis)
    pyplot.close(fig)


def __plot_minimap_hexel_surface(
        stc: SourceEstimate,
        view: str,
        surface: str,
        lh_minimap_axis: pyplot.Axes,
        rh_minimap_axis: pyplot.Axes,
        colormap: Colormap,
        minimap_kwargs: dict,
):
    warn("Plotting on the fsaverage brain. Ensure that hexel numbers match those of the fsaverage brain.")
    plot_kwargs = dict(
        subject="fsaverage",
        surface=surface,
        views=view,
        colormap=colormap,
        smoothing_steps=minimap_kwargs.pop("smoothing_steps", "nearest"),
        cortex=dict(colormap="Greys", vmin=-3, vmax=6),
        background="white",
        spacing="ico5",
        time_viewer=False,
        colorbar=False,
        transparent=False,
        clim=dict(
            kind="value",
            lims=(0.0, 0.5, 1.0),
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
                                       ) -> tuple[Colormap,
                                                  dict[TransformNameDType, float]]:
    """
    Get a colormap appropriate for displaying transforms on a brain minimap.

    Args:
        colors (dict): A dictionary mapping transform names to colours (in any matplotlib-appropriate format, e.g.,
            strings ("red", "#2r4fa6") or rgb(a) tuples ((1.0, 0.0, 0.0, 1.0)).
        show_transforms (list[str]): The transforms which will be shown.

    Returns: Tuple of the following items
        (
            Colormap: Colormap with colours for shown functions interleaved with transparency
            dict[TransformNameDType, float]: Dictionary mapping transform names to values within the colourmap for the
                appropriate color
        )
    """

    # Create a list of unique colors
    unique_colors = sorted(colors.values())

    # Create the colormap with unique colors
    colormap = ListedColormap([transparent] + unique_colors)

    # Map each transform to its color in the colormap
    colormap_value_lookup = {
        # Map the colour to its corresponding value index in the colourmap.
        # You might think you could just use the integer index, but you can't because of the way mne plots brains using
        # a LUT.
        # + 1 for transparency
        TransformNameDType(transform): float((unique_colors.index(colors[transform]) + 1) / len(unique_colors))
        for transform in show_transforms
    }

    return colormap, colormap_value_lookup
