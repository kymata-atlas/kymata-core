"""
Functions copied from mne to add functionality.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from mne import read_surface
from mne._fiff.constants import FIFF
from mne._freesurfer import _read_mri_info, _reorient_image, _mri_orientation

from mne.source_space import SourceSpaces, read_source_spaces
from mne.source_space._source_space import _ensure_src
from mne.transforms import apply_trans, _frame_to_str
from mne.utils import get_subjects_dir, _validate_type, _check_option
from mne.viz.misc import _check_mri, _get_bem_plotting_surfaces
from mne.viz.utils import _prepare_trellis, _figure_agg, plt_show


# Copied from mne.viz.misc.plot_bem to add functionality such as colour selection and multiple transforms
def plot_bem_with_source_values(
    subject,
    subjects_dir=None,
    orientation="coronal",
    slices=None,
    brain_surfaces=None,
    src=None,
    show=True,
    slices_as_subplots=True,
    show_indices=True,
    mri="T1.mgz",
    show_orientation=False,
    draw_surfaces=False,
    colormap=None,
):
    """Plot BEM contours on anatomical MRI slices.

    Parameters
    ----------
    subject : str
        The FreeSurfer subject name.
    subjects_dir : path-like | None
        The path to the directory containing the FreeSurfer subjects reconstructions.
        If None, defaults to the `SUBJECTS_DIR` environment variable.
    orientation : str
        'coronal' or 'axial' or 'sagittal'.
    slices : list of int | None
        The indices of the MRI slices to plot. If ``None``, automatically
        pick 12 equally-spaced slices.
    brain_surfaces : str | list of str | None
        One or more brain surface to plot (optional). Entries should correspond
        to files in the subject's ``surf`` directory (e.g. ``"white"``).
    src : SourceSpaces | path-like | None
        SourceSpaces instance or path to a source space to plot individual
        sources as scatter-plot. Sources will be shown on exactly one slice
        (whichever slice is closest to each source in the given orientation
        plane). Path can be absolute or relative to the subject's ``bem``
        folder.
    show : bool
        Show figure if True.
    slices_as_subplots : bool
        Whether to add all slices as subplots to a single figure, or to
        create a new figure for each slice. If ``False``, return list of open figs.
    show_indices : bool
        Show slice indices if True.
    mri : str
        The name of the MRI to use. Can be a standard FreeSurfer MRI such as
        ``'T1.mgz'``, or a full path to a custom MRI file.
    show_orientation : bool | str
        Show the orientation (L/R, P/A, I/S) of the data slices.
        True (default) will only show it on the outside most edges of the
        figure, False will never show labels, and "always" will label each
        plot.
    draw_surfaces : bool
        Draw the surface contours. Default is False.
    colormap : matplotlib.colors.Colormap | None
        Colormap to use when colouring nonzero values. If None (the default), use a default solid colour.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.

    See Also
    --------
    mne.viz.plot_alignment

    Notes
    -----
    Images are plotted in MRI voxel coordinates.

    If ``src`` is not None, for a given slice index, all source points are
    shown that are halfway between the previous slice and the given slice,
    and halfway between the given slice and the next slice.
    For large slice decimations, this can
    make some source points appear outside the BEM contour, which is shown
    for the given slice index. For example, in the case where the single
    midpoint slice is used ``slices=[128]``, all source points will be shown
    on top of the midpoint MRI slice with the BEM boundary drawn for that
    slice.
    """

    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    mri_fname = _check_mri(mri, subject, subjects_dir)

    # Get the BEM surface filenames
    bem_path = subjects_dir / subject / "bem"

    if not bem_path.is_dir():
        raise OSError(f'Subject bem directory "{bem_path}" does not exist')

    surfaces = _get_bem_plotting_surfaces(bem_path)
    if brain_surfaces is not None:
        if isinstance(brain_surfaces, str):
            brain_surfaces = (brain_surfaces,)
        for surf_name in brain_surfaces:
            for hemi in ("lh", "rh"):
                surf_fname = subjects_dir / subject / "surf" / f"{hemi}.{surf_name}"
                if surf_fname.exists():
                    surfaces.append((surf_fname, "#00DD00"))
                else:
                    raise OSError(f"Surface {surf_fname} does not exist.")

    if isinstance(src, str | Path | os.PathLike):
        src = Path(src)
        if not src.exists():
            # convert to Path until get_subjects_dir returns a Path object
            src_ = Path(subjects_dir) / subject / "bem" / src
            if not src_.exists():
                raise OSError(f"{src} does not exist")
            src = src_
        src = read_source_spaces(src)
    elif src is not None and not isinstance(src, SourceSpaces):
        raise TypeError(
            "src needs to be None, path-like or SourceSpaces instance, "
            f"not {repr(src)}"
        )

    if len(surfaces) == 0:
        raise OSError(
            "No surface files found. Surface files must end with "
            "inner_skull.surf, outer_skull.surf or outer_skin.surf"
        )

    # Plot the contours
    fig = _plot_mri_contours(
        mri_fname=mri_fname,
        surfaces=surfaces,
        src=src,
        orientation=orientation,
        slices=slices,
        show=show,
        show_indices=show_indices,
        show_orientation=show_orientation,
        slices_as_subplots=slices_as_subplots,
        draw_surfaces=draw_surfaces,
        colormap=colormap,
    )
    return fig


# Copied from mne.viz.misc._plot_mri_contours to add functionality such as
def _plot_mri_contours(
    *,
    mri_fname,
    surfaces,
    src,
    orientation="coronal",
    slices=None,
    show=True,
    show_indices=False,
    show_orientation=False,
    width=512,
    slices_as_subplots=True,
    draw_surfaces=False,
    colormap=None,
):
    """Plot BEM contours on anatomical MRI slices.

    Parameters
    ----------
    slices_as_subplots : bool
        Whether to add all slices as subplots to a single figure, or to
        create a new figure for each slice. If ``False``, return list of open figs.

    Returns
    -------
    matplotlib.figure.Figure | list of matplotlib.figure.Figure
        The plotted slices.
    """

    # For ease of plotting, we will do everything in voxel coordinates.
    _validate_type(show_orientation, (bool, str), "show_orientation")
    if isinstance(show_orientation, str):
        _check_option(
            "show_orientation", show_orientation, ("always",), extra="when str"
        )
    _check_option("orientation", orientation, ("coronal", "axial", "sagittal"))

    # Load the T1 data
    _, _, _, _, _, nim = _read_mri_info(mri_fname, units="mm", return_img=True)

    data, rasvox_mri_t = _reorient_image(nim)
    mri_rasvox_t = np.linalg.inv(rasvox_mri_t)
    axis, x, y = _mri_orientation(orientation)

    n_slices = data.shape[axis]

    # if no slices were specified, pick some equally-spaced ones automatically
    if slices is None:
        slices = np.round(np.linspace(start=0, stop=n_slices - 1, num=14)).astype(int)

        # omit first and last one (not much brain visible there anyway…)
        slices = slices[1:-1]

    slices = np.atleast_1d(slices).copy()
    slices[slices < 0] += n_slices  # allow negative indexing
    if (
        not np.array_equal(np.sort(slices), slices)
        or slices.ndim != 1
        or slices.size < 1
        or slices[0] < 0
        or slices[-1] >= n_slices
        or slices.dtype.kind not in "iu"
    ):
        raise ValueError(
            "slices must be a sorted 1D array of int with unique "
            "elements, at least one element, and no elements "
            f"greater than {n_slices - 1:d}, got {slices}"
        )

    # create of list of surfaces
    surfs = list()
    for file_name, color in surfaces:
        surf = dict()
        surf["rr"], surf["tris"] = read_surface(file_name)
        # move surface to voxel coordinate system
        surf["rr"] = apply_trans(mri_rasvox_t, surf["rr"])
        surfs.append((surf, color))

    sources = list()
    source_values = list()
    if src is not None:
        _ensure_src(src, extra=" or None")
        # Eventually we can relax this by allowing ``trans`` if need be
        if src[0]["coord_frame"] != FIFF.FIFFV_COORD_MRI:
            raise ValueError(
                "Source space must be in MRI coordinates, got "
                f'{_frame_to_str[src[0]["coord_frame"]]}'
            )
        for src_ in src:
            points = src_["rr"][src_["inuse"].astype(bool)]
            sources.append(apply_trans(mri_rasvox_t, points * 1e3))
            source_values.append(src_["val"][src_["inuse"].astype(bool)])
        sources = np.concatenate(sources, axis=0)
        source_values = np.concatenate(source_values, axis=0)

    # get the figure dimensions right
    if slices_as_subplots:
        n_col = 4
        fig, axs, _, _ = _prepare_trellis(len(slices), n_col)
        fig.set_facecolor("k")
        dpi = fig.get_dpi()
        n_axes = len(axs)
    else:
        n_col = n_axes = 1
        dpi = 96
        # 2x standard MRI resolution is probably good enough for the
        # traces
        w = width / dpi
        figsize = (w, w / data.shape[x] * data.shape[y])

    bounds = np.concatenate(
        [[-np.inf], slices[:-1] + np.diff(slices) / 2.0, [np.inf]]
    )  # float
    slicer = [slice(None)] * 3
    ori_labels = dict(R="LR", A="PA", S="IS")
    xlabels, ylabels = ori_labels["RAS"[x]], ori_labels["RAS"[y]]
    path_effects = [patheffects.withStroke(linewidth=4, foreground="k", alpha=0.75)]
    figs = []
    for ai, (sl, lower, upper) in enumerate(zip(slices, bounds[:-1], bounds[1:])):
        if slices_as_subplots:
            ax = axs[ai]
        else:
            # No need for constrained layout here because we make our axes fill the
            # entire figure
            fig = _figure_agg(figsize=figsize, dpi=dpi, facecolor="k")
            ax = fig.add_axes([0, 0, 1, 1], frame_on=False, facecolor="k")

        # adjust the orientations for good view
        slicer[axis] = sl
        dat = data[tuple(slicer)].T

        # First plot the anatomical data
        ax.imshow(dat, cmap=plt.cm.gray, origin="lower")
        ax.set_autoscale_on(False)
        ax.axis("off")
        ax.set_aspect("equal")  # XXX eventually could deal with zooms

        # and then plot the contours on top
        if draw_surfaces:
            for surf, color in surfs:
                with warnings.catch_warnings(record=True):  # ignore contour warn
                    warnings.simplefilter("ignore")
                    ax.tricontour(
                        surf["rr"][:, x],
                        surf["rr"][:, y],
                        surf["tris"],
                        surf["rr"][:, axis],
                        levels=[sl],
                        colors=color,
                        linewidths=1.0,
                        zorder=1,
                    )

        if len(sources):
            in_slice = (sources[:, axis] >= lower) & (sources[:, axis] < upper)
            ax.scatter(
                sources[in_slice, x],
                sources[in_slice, y],
                marker=".",
                color="#FF00FF" if colormap is None else colormap(source_values[in_slice]),
                s=1,
                zorder=2,
            )
        if show_indices:
            ax.text(
                dat.shape[1] // 8 + 0.5,
                0.5,
                str(sl),
                color="w",
                fontsize="x-small",
                va="bottom",
                ha="left",
            )
        # label the axes
        kwargs = dict(
            color="#66CCEE",
            fontsize="medium",
            path_effects=path_effects,
            family="monospace",
            clip_on=False,
            zorder=5,
            weight="bold",
        )
        always = show_orientation == "always"
        if show_orientation:
            if ai % n_col == 0 or always:  # left
                ax.text(
                    0, dat.shape[0] / 2.0, xlabels[0], va="center", ha="left", **kwargs
                )
            if ai % n_col == n_col - 1 or ai == n_axes - 1 or always:  # right
                ax.text(
                    dat.shape[1] - 1,
                    dat.shape[0] / 2.0,
                    xlabels[1],
                    va="center",
                    ha="right",
                    **kwargs,
                )
            if ai >= n_axes - n_col or always:  # bottom
                ax.text(
                    dat.shape[1] / 2.0,
                    0,
                    ylabels[0],
                    ha="center",
                    va="bottom",
                    **kwargs,
                )
            if ai < n_col or n_col == 1 or always:  # top
                ax.text(
                    dat.shape[1] / 2.0,
                    dat.shape[0] - 1,
                    ylabels[1],
                    ha="center",
                    va="top",
                    **kwargs,
                )

        if not slices_as_subplots:
            figs.append(fig)

    if slices_as_subplots:
        plt_show(show, fig=fig)
        return fig
    else:
        for f in figs:
            plt_show(show, fig=f)
        return figs
