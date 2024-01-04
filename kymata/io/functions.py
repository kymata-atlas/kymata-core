from pathlib import Path

from numpy import array, float16, convolve
from numpy.typing import NDArray
from h5py import File
from scipy.io import loadmat

from kymata.entities.functions import Function
from kymata.io.file import path_type


def load_function(function_path_without_suffix: path_type, func_name: str, n_derivatives: int = 0) -> Function:
    function_path_without_suffix = Path(function_path_without_suffix)
    func: NDArray
    if function_path_without_suffix.with_suffix(".h5").exists():
        with File(function_path_without_suffix.with_suffix(".h5"), "r") as f:
            func = array(f[func_name])
    else:
        mat = loadmat(str(function_path_without_suffix.with_suffix(".mat")))['stimulisig']
        with File(function_path_without_suffix.with_suffix(".h5"), 'w') as f:
            for key in mat.dtype.names:
                if key == 'name':
                    continue
                f.create_dataset(key, data=array(mat[key][0, 0], dtype=float16))
        func = array(mat[func_name][0][0])

    if func_name in ('STL', 'IL', 'LTL'):
        func = func.T

    for _ in range(n_derivatives):
        func = convolve(func, [-1, 1], 'same')  # derivative

    return Function(
        name=func_name,
        values=func.flatten().squeeze(),
        sample_rate=1000,
    )
