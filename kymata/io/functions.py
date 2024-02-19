from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from kymata.entities.functions import Function
from kymata.io.file import path_type


def load_function(function_path_without_suffix: path_type, func_name: str, n_derivatives: int = 0, bruce_neurons: tuple = (0, 10)) -> Function:
    function_path_without_suffix = Path(function_path_without_suffix)
    func: NDArray
    if 'neurogram' in func_name:
        print('USING BRUCE MODEL')
        if function_path_without_suffix.with_suffix(".npz").exists():
            func_dict = np.load(function_path_without_suffix.with_suffix(".npz"))
            func = func_dict[func_name]
            func = np.mean(func[bruce_neurons[0]:bruce_neurons[1]], axis=0)
        else:
            mat = loadmat(function_path_without_suffix.with_suffix('.mat'))['neurogramResults']
            func = np.array(mat[func_name][0][0])
            # func = np.mean(func[bruce_neurons[0]:bruce_neurons[1]], axis=0)
            tt = np.array(mat['t_'+func_name[-2:]][0][0])
            n_chans = func.shape[0]
            new_mr_arr = np.zeros((n_chans, 400_000))
            new_ft_arr = np.zeros((n_chans, 400_000))
            if func_name == 'neurogram_mr':
                for i in range(n_chans):
                    new_mr_arr[i] = np.interp(np.linspace(0, 400, 400_000 + 1)[:-1], tt[0], func[i])
            elif func_name == 'neurogram_ft':
                func = np.cumsum(func * (tt[0, 1] - tt[0, 0]), axis=-1)
                for i in range(n_chans):
                    func_interp = np.interp(np.linspace(0, 400, 400_000 + 1), tt[0], func[i])
                    new_ft_arr[i] = np.diff(func_interp, axis=-1)

            func_dict = {'neurogram_mr': new_mr_arr,
                         'neurogram_ft': new_ft_arr}
            np.savez(str(function_path_without_suffix.with_suffix(".npz")), **func_dict)

            func = func_dict[func_name]
            func = np.mean(func[bruce_neurons[0]:bruce_neurons[1]], axis=0)

    else:
        if function_path_without_suffix.with_suffix(".npz").exists():
            func_dict = np.load(str(function_path_without_suffix.with_suffix(".npz")))
            func = np.array(func_dict[func_name])
        else:
            mat = loadmat(str(function_path_without_suffix.with_suffix(".mat")))['stimulisig']
            func_dict = {}
            for key in mat.dtype.names:
                if key == 'name':
                    continue
                func_dict[key] = np.array(mat[key][0, 0], dtype=np.float16)
            np.savez(str(function_path_without_suffix.with_suffix(".npz")), **func_dict)
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
