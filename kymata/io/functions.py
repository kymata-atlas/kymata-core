from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from kymata.entities.functions import Function
from kymata.io.file import path_type


def load_function(function_path_without_suffix: path_type, func_name: str, n_derivatives: int = 0, n_hamming: int = 0, bruce_neurons: tuple = (5, 10), nn_neuron: int = 201) -> Function:
    function_path_without_suffix = Path(function_path_without_suffix)
    func: NDArray
    if 'neurogram' in func_name:
        if function_path_without_suffix.with_suffix(".npz").exists():
            func_dict = np.load(function_path_without_suffix.with_suffix(".npz"))
            func = func_dict[func_name]
            func = np.mean(func[bruce_neurons[0]:bruce_neurons[1]], axis=0)
            func_name += f'_{str(bruce_neurons[0])}-{str(bruce_neurons[1])}'
        else:
            mat = loadmat(function_path_without_suffix.with_suffix('.mat'))['neurogramResults']
            func_mr = np.array(mat['neurogram_mr'][0][0])
            func_ft = np.array(mat['neurogram_ft'][0][0])
            # func = np.mean(func[bruce_neurons[0]:bruce_neurons[1]], axis=0)
            tt_mr = np.array(mat['t_mr'][0][0])
            tt_ft = np.array(mat['t_ft'][0][0])
            n_chans_mr = func_mr.shape[0]
            n_chans_ft = func_ft.shape[0]
            new_mr_arr = np.zeros((n_chans_mr, 400_000))
            new_ft_arr = np.zeros((n_chans_ft, 400_000))
            # T_end = tt[0, -1]

            for i in range(n_chans_mr):
                new_mr_arr[i] = np.interp(np.linspace(0, 400, 400_000 + 1)[:-1], tt_mr[0], func_mr[i])

            func_ft = np.cumsum(func_ft * (tt_ft[0, 1] - tt_ft[0, 0]), axis=-1)
            for i in range(n_chans_ft):
                func_interp = np.interp(np.linspace(0, 400, 400_000 + 1), tt_ft[0], func_ft[i])
                new_ft_arr[i] = np.diff(func_interp, axis=-1)

            func_dict = {'neurogram_mr': new_mr_arr,
                         'neurogram_ft': new_ft_arr}
            np.savez(str(function_path_without_suffix.with_suffix(".npz")), **func_dict)

            func = func_dict[func_name]
            func = np.mean(func[bruce_neurons[0]:bruce_neurons[1]], axis=0)

    elif 'asr_models' in str(function_path_without_suffix):
        if 'whisper_all_no_reshape' in str(function_path_without_suffix):
            func_dict = np.load(function_path_without_suffix.with_suffix(".npz"))
            func = func_dict[func_name]

            import ipdb;ipdb.set_trace()

            T_max = 401
            s_num = T_max * 1000
            if 'conv' in func_name or func.shape[0] != 1:
                place_holder = np.zeros((func.shape[1], s_num))
            else:
                place_holder = np.zeros((func.shape[2], s_num))
            for j in range(place_holder.shape[0]):
                if 'conv' in func_name:
                    place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, func.shape[2]), func[0, j, :])
                elif func.shape[0] != 1:
                    place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, func.shape[0]), func[:, j])
                else:
                    place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, func.shape[1]), func[0, :, j])
            
            if nn_neuron in ('avr', 'ave', 'mean', 'all'):
                func = np.mean(place_holder[:, :400_000], axis=0) #func[nn_neuron]
            else:
                func = place_holder[nn_neuron, :400_000]
            func_name += f'_{str(nn_neuron)}'

        else:
            func_dict = np.load(function_path_without_suffix.with_suffix(".npz"))
            func = func_dict[func_name]
            if nn_neuron in ('avr', 'ave', 'mean', 'all'):
                func = np.mean(func[:, :400_000], axis=0) #func[nn_neuron]
            else:
                func = func[nn_neuron, :400_000]
            func_name += f'_{str(nn_neuron)}'

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
        elif func_name in ('d_STL', 'd_IL', 'd_LTL'):
            func = np.convolve(func_dict[func_name[2:]].T.flatten(), [-1, 1], 'same')

    for _ in range(n_derivatives):
        func = np.convolve(func, [-1, 1], 'same')  # derivative
        func_name = f'd{n_derivatives}_' + func_name

    if n_hamming > 1:
        func = np.convolve(func, np.hamming(n_hamming), 'same')  # hamming conv (effectively gaussian smoothing)
        func_name = f'ham{n_hamming}_' + func_name

    return Function(
        name=func_name,
        values=func.flatten().squeeze(),
        sample_rate=1000,
    )

def load_function_pre(function_path_without_suffix: path_type, func_name: str):
    function_path_without_suffix = Path(function_path_without_suffix)
    func: NDArray
    func_dict = np.load(function_path_without_suffix.with_suffix(".npz"))
    func = func_dict[func_name]
    return func