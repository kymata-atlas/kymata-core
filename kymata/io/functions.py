from logging import getLogger
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from kymata.entities.functions import Function
from kymata.io.file import PathType


_logger = getLogger(__file__)


def load_function(
    function_path_without_suffix: PathType,
    func_name: str,
    replace_nans: Optional[str] = None,
    n_derivatives: int = 0,
    bruce_neurons: tuple = (0, 10),
) -> Function:
    function_path_without_suffix = Path(function_path_without_suffix)
    func: NDArray
    if "neurogram" in func_name:
        _logger.info("USING BRUCE MODEL")
        if function_path_without_suffix.with_suffix(".npz").exists():
            func_dict = np.load(function_path_without_suffix.with_suffix(".npz"))
            func = func_dict[func_name]
            func = np.mean(func[bruce_neurons[0] : bruce_neurons[1]], axis=0)
        else:
            mat = loadmat(function_path_without_suffix.with_suffix(".mat"))[
                "neurogramResults"
            ]
            func = np.array(mat[func_name][0][0])
            # func = np.mean(func[bruce_neurons[0]:bruce_neurons[1]], axis=0)
            tt = np.array(mat["t_" + func_name[-2:]][0][0])
            n_chans = func.shape[0]
            new_mr_arr = np.zeros((n_chans, 400_000))
            new_ft_arr = np.zeros((n_chans, 400_000))
            if func_name == "neurogram_mr":
                for i in range(n_chans):
                    new_mr_arr[i] = np.interp(
                        np.linspace(0, 400, 400_000 + 1)[:-1], tt[0], func[i]
                    )
            elif func_name == "neurogram_ft":
                func = np.cumsum(func * (tt[0, 1] - tt[0, 0]), axis=-1)
                for i in range(n_chans):
                    func_interp = np.interp(
                        np.linspace(0, 400, 400_000 + 1), tt[0], func[i]
                    )
                    new_ft_arr[i] = np.diff(func_interp, axis=-1)

            func_dict = {"neurogram_mr": new_mr_arr, "neurogram_ft": new_ft_arr}
            np.savez(str(function_path_without_suffix.with_suffix(".npz")), **func_dict)

            func = func_dict[func_name]
            func = np.mean(func[bruce_neurons[0] : bruce_neurons[1]], axis=0)

    else:
        if not function_path_without_suffix.with_suffix(".npz").exists():
            if function_path_without_suffix.with_suffix(".mat").exists():
                convert_stimulisig_on_disk_mat_to_npz(function_path_without_suffix)
            else:
                raise FileNotFoundError(
                    f"{function_path_without_suffix}: neither .mat nor .npz found."
                )

        assert function_path_without_suffix.with_suffix(".npz").exists()

        func_dict = np.load(str(function_path_without_suffix.with_suffix(".npz")))
        func = np.array(func_dict[func_name], dtype=float)

    n_nan = np.count_nonzero(np.isnan(func))
    if n_nan > 0:
        nan_message = (
            f"Function contained {n_nan:,} NaNs, ({100*n_nan/np.prod(func.shape):.2f}%)"
        )
        if replace_nans is None:
            raise ValueError(
                f"{nan_message}. "
                "Consider replacing small numbers of unavoidable NaNs with zeros or the function mean "
                "value."
            )
        elif replace_nans == "zero":
            _logger.warning(f"{nan_message}. Replacing with zeros.")
            func[np.isnan(func)] = 0
        elif replace_nans == "mean":
            _logger.warning(f"{nan_message}. Replacing with the function mean.")
            func[np.isnan(func)] = np.nanmean(func)
        else:
            raise NotImplementedError()

    assert not np.isnan(func).any()

    for _ in range(n_derivatives):
        func = np.convolve(func, [-1, 1], "same")  # derivative

    return Function(
        name=func_name,
        values=func.flatten().squeeze(),
        sample_rate=1000,
    )


def convert_stimulisig_on_disk_mat_to_npz(function_path_without_suffix):
    """
    Converts a (legacy) <stimulisig>.mat file to a (current) Numpy <stimulisig>.npz file on disk. Supply a path without
    a suffix, and the file will be created in the same place with the same name but with a different suffix.
    """
    mat = loadmat(str(function_path_without_suffix.with_suffix(".mat")))["stimulisig"]
    _logger.info("Saving .mat as .npz ...")
    func_dict = {}
    for key in mat.dtype.names:
        if key == "name":
            continue
        func_dict[key] = np.array(mat[key][0, 0], dtype=np.float16)
        func_dict[key].reshape((1, -1))  # Unwrap if it's a split matlab stimulisig
    np.savez(str(function_path_without_suffix.with_suffix(".npz")), **func_dict)
