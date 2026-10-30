from logging import getLogger
from pathlib import Path
from typing import Optional

import numpy as np

from kymata.entities.contours import Contour
from kymata.io.compatibility.neurogram import load_neurogram_mat, load_neurogram_npz
from kymata.io.compatibility.stimulisig import load_stimulisig_mat
from kymata.io.contours import load_contours
from kymata.io.file import PathType


_logger = getLogger(__file__)


def load_transform_legacy(
    transform_path_without_suffix: PathType,
    trans_name: str,
    sample_rate: float,
    replace_nans: Optional[str] = None,
    n_derivatives: int = 0,
    bruce_neurons: tuple = (0, 10),
) -> Contour:
    transform_path_without_suffix = Path(transform_path_without_suffix)

    if "neurogram" in trans_name:
        _logger.info("USING BRUCE MODEL")
        if transform_path_without_suffix.with_suffix(".npz").exists():
            contours = load_neurogram_npz(
                location=transform_path_without_suffix.with_suffix(".npz"),
                transform_name=trans_name,
                bruce_neurons=bruce_neurons,
            )
        elif transform_path_without_suffix.with_suffix(".mat").exists():
            contours = load_neurogram_mat(
                location=transform_path_without_suffix.with_suffix(".mat"),
                transform_name=trans_name,
                bruce_neurons=bruce_neurons,
            )
        else:
            raise FileNotFoundError(f"{transform_path_without_suffix}: neither .mat nor .npz found.")

    elif transform_path_without_suffix.with_suffix(".npz").exists():
        contours = load_contours(transform_path_without_suffix.with_suffix(".npz"))
    elif transform_path_without_suffix.with_suffix(".mat").exists():
        contours = load_stimulisig_mat(transform_path_without_suffix.with_suffix(".mat"))
    else:
        raise FileNotFoundError(f"{transform_path_without_suffix}: neither .mat nor .npz found.")

    trans = contours[trans_name].values

    n_nan = np.count_nonzero(np.isnan(trans))
    if n_nan > 0:
        nan_message = (
            f"Transform contained {n_nan:,} NaNs, ({100*n_nan/np.prod(trans.shape):.2f}%)"
        )
        if replace_nans is None:
            raise ValueError(
                f"{nan_message}. "
                "Consider replacing small numbers of unavoidable NaNs with zeros or the transform mean "
                "value."
            )
        elif replace_nans == "zero":
            _logger.warning(f"{nan_message}. Replacing with zeros.")
            trans[np.isnan(trans)] = 0
        elif replace_nans == "mean":
            _logger.warning(f"{nan_message}. Replacing with the transform mean.")
            trans[np.isnan(trans)] = np.nanmean(trans)
        else:
            raise NotImplementedError()

    assert not np.isnan(trans).any()

    for _ in range(n_derivatives):
        trans = np.convolve(trans, [-1, 1], "same")  # derivative

    return Contour(
        name=trans_name,
        values=trans.flatten().squeeze().astype(np.float32),
        sample_rate_hz=sample_rate,
    )
