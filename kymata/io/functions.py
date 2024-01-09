from pathlib import Path

from numpy import array, float16, convolve, mean
import numpy as np
from numpy.typing import NDArray
from h5py import File
from scipy.io import loadmat

from kymata.entities.functions import Function
from kymata.io.file import path_type


def load_function(function_path_without_suffix: path_type, func_name: str, n_derivatives: int = 0, bruce_neurons: tuple = (0, 10)) -> Function:
    function_path_without_suffix = Path(function_path_without_suffix)
    func: NDArray
    if 'neurogram' in func_name:
        print('USING BRUCE MODEL')
        mat = loadmat(function_path_without_suffix.with_suffix('.mat'))['neurogramResults']
        func = np.array(mat[func_name][0][0])
        func = np.mean(func[bruce_neurons[0]:bruce_neurons[1]], axis=0)
        tt = np.array(mat['t_'+func_name[-2:]][0][0])

        T_end = tt[0, -1]
        if func_name == 'neurogram_mr':
            func = np.interp(np.linspace(0, 400, 400_000 + 1)[:-1], tt[0], func, )
        elif func_name == 'neurogram_ft':
            func = np.cumsum(func * (tt[0, 1] - tt[0, 0]), axis=-1)
            func = np.interp(np.linspace(0, 400, 400_000 + 1)[:], tt[0], func)
            func = np.diff(func, axis=-1)

    else:
        if function_path_without_suffix.with_suffix(".h5").exists():
            with File(function_path_without_suffix.with_suffix(".h5"), "r") as f:
                func = np.array(f[func_name])
        else:
            mat = loadmat(str(function_path_without_suffix.with_suffix(".mat")))['stimulisig']
            with File(function_path_without_suffix.with_suffix(".h5"), 'w') as f:
                for key in mat.dtype.names:
                    if key == 'name':
                        continue
                    f.create_dataset(key, data=np.array(mat[key][0, 0], dtype=np.float16))
            func = np.array(mat[func_name][0][0])

        if func_name in ('STL', 'IL', 'LTL'):
            func = func.T

    for _ in range(n_derivatives):
        func = np.convolve(func, [-1, 1], 'same')  # derivative

    return Function(
        name=func_name,
        values=func.flatten().squeeze(),
        sample_rate=1000,
    )
