import os
from pathlib import Path
from typing import OrderedDict

import numpy as np
from numpy.typing import NDArray

from kymata.entities.contours import PredictedContourSet, ContoursSource
from kymata.io.file import PathType
from kymata.system.git import get_current_git_commit_hash


_contours_suffix = ".ctr"
_contours_format_version = "0.2"


class _CArrKeys:
    """Keys for storing metadata in the contour array."""
    prefix      = "_kymata-"

    sample_rate = "_kymata-contours-sample_rate"
    description = "_kymata-contours-description"
    author      = "_kymata-contours-author"
    order       = "_kymata-contours-order"

    source_repository = "_kymata-contours-source_repository"
    source_commit     = "_kymata-contours-source_commit"

    version     = "_kymata-contours-format-version"


def load_contours(
        from_path: PathType,
        replace_sample_rate: float | None = None,
) -> PredictedContourSet:
    """
    Loads contours from a file.

    Ideally this would load a .contours file saved with ``save_contours``, but other formats are supported:

    - .contours file
      - This is a compressed numpy .npz file containing the arrays for the different contours, and some optional
        metadata.
    - .npz file
      - A .npz file will be interpreted by loading each named array into a separate contour. This will succeed when all
        arrays are 1-dimensional and haver the same length.

    Args:
        from_path: Path to the contours file.
        replace_sample_rate: Optional. Replace the sample rate listed in the file with the specified one. This can be
            useful if there is no sample rate listed in the file. Supply None (the default) to not replace the sample
            rate.

    Returns:

    """
    from_path = Path(from_path)

    if from_path.suffix == _contours_suffix or from_path.suffix == ".npz":
        with np.load(from_path) as nz_data:
            sample_rate   = nz_data.get(_CArrKeys.sample_rate,       None)
            description   = nz_data.get(_CArrKeys.description,       None)
            author        = nz_data.get(_CArrKeys.author,            None)
            order         = nz_data.get(_CArrKeys.order,             None)
            source_repo   = nz_data.get(_CArrKeys.source_repository, None)
            source_commit = nz_data.get(_CArrKeys.source_commit,     None)
            if source_repo is None and source_commit is None:
                source = None
            else:
                source = ContoursSource(repository=source_repo, commit=source_commit)

            if order is None:
                # Just use the non-reserved names in whatever order they arrive
                order = [key for key in nz_data.keys() if not key.startswith(_CArrKeys.prefix)]
                preserving_order = True
            else:
                preserving_order = False

            data_dict = OrderedDict() if preserving_order else dict()

            for name in order:
                data_dict[name] = nz_data[name]

        if replace_sample_rate is not None:
            sample_rate = replace_sample_rate

        return PredictedContourSet(
            contour_dict=data_dict,
            sample_rate_hz=sample_rate,
            description=description,
            author=author,
            source=source,
        )

    else:
        raise NotImplementedError(f"Not supported for filename {from_path!s}")


def _data_dict(contours: PredictedContourSet) -> dict[str, str | NDArray]:
    """Converts a Contours object into a data dict suitable for saving."""
    d = dict()
    for name in contours.names:
        d[name] = contours[name].values

    d[_CArrKeys.sample_rate] = contours.sample_rate_hz
    if contours.description is not None:
        d[_CArrKeys.description] = contours.description
    if contours.author is not None:
        d[_CArrKeys.author] = contours.author
    d[_CArrKeys.order] = contours.names
    if contours.source is not None:
        if contours.source.repository is not None:
            d[_CArrKeys.source_repository] = contours.source.repository
        if contours.source.commit is not None:
            d[_CArrKeys.source_commit] = contours.source.commit

    d[_CArrKeys.version] = _contours_format_version

    return d


def _fill_default_source_metadata(contours: PredictedContourSet) -> None:
    """
    Updates a PredictedContourSet's source metadata in place with default values based on the current checkout.

    Existing values are left untouched: only None values are updated.
    """
    source = contours.source if contours.source is not None else ContoursSource(None, None)
    # If not supplied, attempt to get from the current checkout.
    if source.repository is None:
        # Path of *this* file in the checkout
        try:
            source.repository = str(Path(__file__).resolve())
        except NameError:
            # If running inside a Jupyter notebook, this can fail because there is no way to get the path to a running
            # notebook [1].
            # [1]: https://stackoverflow.com/a/52119628/2883198
            #
            # The best we can do is this:
            source.repository = os.path.abspath('')
    if source.commit is None:
        source.commit = get_current_git_commit_hash()
    contours.source = source


def save_contours(contours: PredictedContourSet, to_file: PathType, overwrite: bool = False) -> None:
    """
    Write the contours to a file.

    Args:
        contours:
        to_file:
        overwrite:
    """
    _fill_default_source_metadata(contours)

    to_file = Path(to_file)
    supplied_suffix = to_file.suffix
    if not supplied_suffix:
        supplied_suffix = _contours_suffix
    if to_file.exists() and not overwrite:
        raise FileExistsError(str(to_file))
    np.savez_compressed(
        file=to_file.with_suffix(".npz"),
        allow_pickle=False,
        **_data_dict(contours),
    )
    # Save with the suffix supplied
    to_file.with_suffix(".npz").rename(to_file.with_suffix(supplied_suffix))
