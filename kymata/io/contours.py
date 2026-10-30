from pathlib import Path
from typing import OrderedDict

import numpy as np
from numpy.typing import NDArray

from kymata.entities.contours import Contours
from kymata.io.file import PathType


_contours_suffix = ".contours"
_contours_format_version = "0.1"


class _CArrKeys:
    """Keys for storing metadata in the contour array."""
    prefix      = "_kymata-"

    sample_rate = "_kymata-contours-sample_rate"
    description = "_kymata-contours-description"
    author      = "_kymata-contours-author"
    order       = "_kymata-contours-order"

    version     = "_kymata-contours-format-version"


def load_contours(
        from_path: PathType,
        replace_sample_rate: float | None = None,
) -> Contours:
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
            sample_rate = nz_data.get(_CArrKeys.sample_rate, None)
            description = nz_data.get(_CArrKeys.description, None)
            author      = nz_data.get(_CArrKeys.author,      None)
            order       = nz_data.get(_CArrKeys.order,       None)

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

        return Contours(
            contour_dict=data_dict,
            sample_rate_hz=sample_rate,
            description=description,
            author=author,
        )

    else:
        raise NotImplementedError()


def _data_dict(contours: Contours) -> dict[str, str | NDArray]:
    """Converts a Contours object into a data dict suitable for saving."""
    d = dict()
    for name in contours.names:
        d[name] = contours[name]

    d[_CArrKeys.sample_rate] = contours.sample_rate_hz
    if contours.description is not None:
        d[_CArrKeys.description] = contours.description
    if contours.author is not None:
        d[_CArrKeys.author] = contours.author
    d[_CArrKeys.order] = contours.names

    d[_CArrKeys.version] = _contours_format_version

    return d


def save_contours(contours: Contours, to_file: PathType, overwrite: bool = False) -> None:
    """
    Write the contours to a file.

    Args:
        contours:
        to_file:
        overwrite:
    """
    to_file = Path(to_file)
    if not to_file.suffix:
        to_file = to_file.with_suffix(_contours_suffix)
    if to_file.exists() and not overwrite:
        raise FileExistsError(str(to_file))
    np.savez_compressed(
        file=to_file,
        allow_pickle=False,
        **_data_dict(contours),
    )
