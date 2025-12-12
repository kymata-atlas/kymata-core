from logging import getLogger
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from kymata.entities.transform import Transform
from kymata.io.file import PathType


_logger = getLogger(__file__)


def load_transform(
    transform_path_without_suffix: PathType,
    trans_name: str,
    sample_rate: float,
    replace_nans: Optional[str] = None,
    n_derivatives: int = 0,
    bruce_neurons: tuple = (0, 10),
) -> Transform:
    transform_path_without_suffix = Path(transform_path_without_suffix)
    trans: NDArray
    if "neurogram" in trans_name:
        _logger.info("USING BRUCE MODEL")
        if transform_path_without_suffix.with_suffix(".npz").exists():
            trans_dict = np.load(transform_path_without_suffix.with_suffix(".npz"))
            trans = trans_dict[trans_name]
            trans = np.mean(trans[bruce_neurons[0] : bruce_neurons[1]], axis=0)
        else:
            mat = loadmat(transform_path_without_suffix.with_suffix(".mat"))[
                "neurogramResults"
            ]
            trans = np.array(mat[trans_name][0][0])
            # trans = np.mean(func[bruce_neurons[0]:bruce_neurons[1]], axis=0)
            tt = np.array(mat["t_" + trans_name[-2:]][0][0])
            n_chans = trans.shape[0]
            new_mr_arr = np.zeros((n_chans, 400_000))
            new_ft_arr = np.zeros((n_chans, 400_000))
            if trans_name == "neurogram_mr":
                for i in range(n_chans):
                    new_mr_arr[i] = np.interp(
                        np.linspace(0, 400, 400_000 + 1)[:-1], tt[0], trans[i]
                    )
            elif trans_name == "neurogram_ft":
                trans = np.cumsum(trans * (tt[0, 1] - tt[0, 0]), axis=-1)
                for i in range(n_chans):
                    trans_interp = np.interp(
                        np.linspace(0, 400, 400_000 + 1), tt[0], trans[i]
                    )
                    new_ft_arr[i] = np.diff(trans_interp, axis=-1)

            trans_dict = {"neurogram_mr": new_mr_arr, "neurogram_ft": new_ft_arr}
            np.savez(str(transform_path_without_suffix.with_suffix(".npz")), **trans_dict)

            trans = trans_dict[trans_name]
            trans = np.mean(trans[bruce_neurons[0] : bruce_neurons[1]], axis=0)

    else:
        if not transform_path_without_suffix.with_suffix(".npz").exists():
            if transform_path_without_suffix.with_suffix(".mat").exists():
                convert_stimulisig_on_disk_mat_to_npz(transform_path_without_suffix)
            else:
                raise FileNotFoundError(
                    f"{transform_path_without_suffix}: neither .mat nor .npz found."
                )

        assert transform_path_without_suffix.with_suffix(".npz").exists()

        trans_dict = np.load(str(transform_path_without_suffix.with_suffix(".npz")))
        trans = np.array(trans_dict[trans_name], dtype=float)

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

    return Transform(
        name=trans_name,
        values=trans.flatten().squeeze().astype(np.float32),
        sample_rate=sample_rate,
    )


def convert_stimulisig_on_disk_mat_to_npz(transform_path_without_suffix):
    """
    Converts a (legacy) <stimulisig>.mat file to a (current) Numpy <stimulisig>.npz file on disk. Supply a path without
    a suffix, and the file will be created in the same place with the same name but with a different suffix.
    """
    mat = loadmat(str(transform_path_without_suffix.with_suffix(".mat")))["stimulisig"]
    _logger.info("Saving .mat as .npz ...")
    trans_dict = {}
    for key in mat.dtype.names:
        if key == "name":
            continue
        trans_dict[key] = np.array(mat[key][0, 0], dtype=np.float16)
        trans_dict[key].reshape((1, -1))  # Unwrap if it's a split matlab stimulisig
    np.savez(str(transform_path_without_suffix.with_suffix(".npz")), **trans_dict)
