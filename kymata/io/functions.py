from pathlib import Path
import os

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from kymata.entities.functions import Function
from kymata.io.file import PathType


def load_function(function_path_without_suffix: PathType, func_name: str, n_derivatives: int = 0, n_hamming: int = 0, bruce_neurons: tuple = (5, 10), nn_neuron: int = 201, mfa: bool = False) -> Function:
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
            if 'logmel' in str(function_path_without_suffix):
                func = np.load(function_path_without_suffix.with_suffix(".npy"))
            else:
                func_dict = np.load(function_path_without_suffix.with_suffix(".npz"))
                func = func_dict[func_name]
            T_max = 401
            s_num = T_max * 1000
            if 'conv' in func_name or func.shape[0] != 1 or 'logmel' in str(function_path_without_suffix):
                place_holder = np.zeros((func.shape[1], s_num))
            else:
                place_holder = np.zeros((func.shape[2], s_num))
            # Below is a temporary fix to make the code run faster. Clean-up is definitely needed later on
            if nn_neuron in ('avr', 'ave', 'mean', 'all'):
                for j in range(place_holder.shape[0]):
                    if 'decoder' in func_name and 'encoder_attn.k' not in func_name and 'encoder_attn.v' not in func_name:
                        if not mfa:
                            time_stamps_seconds = np.load(f'{function_path_without_suffix}_timestamp.npy')
                            time_stamps_samples = (time_stamps_seconds * 1000).astype(int)
                            time_stamps_samples = np.append(time_stamps_samples, 402_000)
                            for i in range(len(time_stamps_samples) - 1):
                                start_idx = time_stamps_samples[i]
                                end_idx = time_stamps_samples[i + 1]
                                if start_idx < s_num:
                                    if 'conv' in func_name:
                                        place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, j, i])
                                    elif func.shape[0] != 1:
                                        place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[i, j])
                                    else:
                                        place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, i, j])
                        else:
                            if os.path.exists(f'{function_path_without_suffix}_timestamp.npy'):
                                time_stamps_seconds = np.load(f'{function_path_without_suffix}_timestamp.npy')
                                time_stamps_samples = (time_stamps_seconds * 1000).astype(int)
                                time_stamps_samples = np.append(time_stamps_samples, 402_000)
                            whisper_text = [i.lower() for i in load_txt(f'{function_path_without_suffix}_whisper_transcription.txt') if i != '<|startoftranscript|>']
                            mfa_text = load_txt(f'{function_path_without_suffix}_mfa_text.txt')
                            mfa_time = np.array(load_txt(f'{function_path_without_suffix}_mfa_stime.txt')).astype(float)
                            mfa_time_samples = (mfa_time * 1000).astype(int)
                            mfa_time_samples = np.append(mfa_time_samples, 402_000)
                            special_tokens = ['<|en|>', '<|transcribe|>', '<|notimestamps|>', '<|endoftext|>']
                            k = 0       # k is the index for whisper space, and i is the index for mfa space
                            for i in range(len(mfa_time_samples) - 1):
                                start_idx = mfa_time_samples[i]
                                end_idx = mfa_time_samples[i + 1]
                                if start_idx < s_num:
                                    while whisper_text[k] in special_tokens:
                                        k += 1
                                    if mfa_text[i] != whisper_text[k]:
                                        if mfa_text[i] == '<sp>':
                                            # if whisper_text[k] == '.' or whisper_text[k] == ',':
                                            place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                                            # print('<sp> in mfa encountered')
                                        else:
                                            search_txt = whisper_text[k]
                                            id_tracker = [k]
                                            if os.path.exists(f'{function_path_without_suffix}_timestamp.npy'):
                                                weight_tracker = [time_stamps_samples[k+1]-time_stamps_samples[k]]
                                            # combine word pieces in whisper text, and also combine potential '.' and ',' to the following word in whisper
                                            while len(search_txt) < len(mfa_text[i]) + 1 and mfa_text[i] not in search_txt:
                                                k += 1
                                                if whisper_text[k] not in special_tokens:
                                                    search_txt += whisper_text[k]
                                                    id_tracker.append(k)
                                                    if os.path.exists(f'{function_path_without_suffix}_timestamp.npy'):
                                                        weight_tracker.append(time_stamps_samples[k+1]-time_stamps_samples[k])
                                            if not os.path.exists(f'{function_path_without_suffix}_timestamp.npy'):
                                                place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) , np.average([func[0, k, j] for k in id_tracker]))
                                            elif sum(weight_tracker) == 0:
                                                place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) , np.average([func[0, k, j] for k in id_tracker]))
                                            else:
                                                place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) , np.average([func[0, k, j] for k in id_tracker], weights=weight_tracker))
                                            k += 1
                                            # print(f'mapping is from the mfa token {[mfa_text[i]]} to the whisper token {[whisper_text[k] for k in id_tracker]}')
                                    else:
                                        place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                                        k += 1
                                        # print('match')
                            # import ipdb;ipdb.set_trace()
                            assert k == len(whisper_text) - 1, 'end of whisper text not reached'                            

                    else:    
                        if 'conv' in func_name or 'logmel' in str(function_path_without_suffix):
                            place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, func.shape[2]), func[0, j, :])
                        elif func.shape[0] != 1:
                            place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, func.shape[0]), func[:, j])
                        else:
                            place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, func.shape[1]), func[0, :, j])
            else:
                j = nn_neuron
                if 'decoder' in func_name and 'encoder_attn.k' not in func_name and 'encoder_attn.v' not in func_name:
                    if not mfa:
                        time_stamps_seconds = np.load(f'{function_path_without_suffix}_timestamp.npy')
                        time_stamps_samples = (time_stamps_seconds * 1000).astype(int)
                        time_stamps_samples = np.append(time_stamps_samples, 402_000)
                        for i in range(len(time_stamps_samples) - 1):
                            start_idx = time_stamps_samples[i]
                            end_idx = time_stamps_samples[i + 1]
                            if start_idx < s_num:
                                if 'conv' in func_name:
                                    place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, j, i])
                                elif func.shape[0] != 1:
                                    place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[i, j])
                                else:
                                    place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, i, j])
                    else:
                        if os.path.exists(f'{function_path_without_suffix}_timestamp.npy'):
                            time_stamps_seconds = np.load(f'{function_path_without_suffix}_timestamp.npy')
                            time_stamps_samples = (time_stamps_seconds * 1000).astype(int)
                            time_stamps_samples = np.append(time_stamps_samples, 402_000)
                        whisper_text = [i.lower() for i in load_txt(f'{function_path_without_suffix}_whisper_transcription.txt') if i != '<|startoftranscript|>']
                        mfa_text = load_txt(f'{function_path_without_suffix}_mfa_text.txt')
                        mfa_time = np.array(load_txt(f'{function_path_without_suffix}_mfa_stime.txt')).astype(float)
                        mfa_time_samples = (mfa_time * 1000).astype(int)
                        mfa_time_samples = np.append(mfa_time_samples, 402_000)
                        special_tokens = ['<|en|>', '<|transcribe|>', '<|notimestamps|>', '<|endoftext|>']
                        k = 0       # k is the index for whisper space, and i is the index for mfa space
                        for i in range(len(mfa_time_samples) - 1):
                            start_idx = mfa_time_samples[i]
                            end_idx = mfa_time_samples[i + 1]
                            if start_idx < s_num:
                                while whisper_text[k] in special_tokens:
                                    k += 1
                                if mfa_text[i] != whisper_text[k]:
                                    if mfa_text[i] == '<sp>':
                                        # if whisper_text[k] == '.' or whisper_text[k] == ',':
                                        place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                                        # print('<sp> in mfa encountered')
                                    else:
                                        search_txt = whisper_text[k]
                                        id_tracker = [k]
                                        if os.path.exists(f'{function_path_without_suffix}_timestamp.npy'):
                                            weight_tracker = [time_stamps_samples[k+1]-time_stamps_samples[k]]
                                        # combine word pieces in whisper text, and also combine potential '.' and ',' to the following word in whisper
                                        while len(search_txt) < len(mfa_text[i]) + 1 and mfa_text[i] not in search_txt:
                                            k += 1
                                            if whisper_text[k] not in special_tokens:
                                                search_txt += whisper_text[k]
                                                id_tracker.append(k)
                                                if os.path.exists(f'{function_path_without_suffix}_timestamp.npy'):
                                                    weight_tracker.append(time_stamps_samples[k+1]-time_stamps_samples[k])
                                        if not os.path.exists(f'{function_path_without_suffix}_timestamp.npy'):
                                            place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) , np.average([func[0, k, j] for k in id_tracker]))
                                        elif sum(weight_tracker) == 0:
                                            place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) , np.average([func[0, k, j] for k in id_tracker]))
                                        else:
                                            place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) , np.average([func[0, k, j] for k in id_tracker], weights=weight_tracker))
                                        k += 1
                                        # print(f'mapping is from the mfa token {[mfa_text[i]]} to the whisper token {[whisper_text[k] for k in id_tracker]}')
                                else:
                                    place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                                    k += 1
                                    # print('match')
                        # import ipdb;ipdb.set_trace()
                        # assert k == len(whisper_text) - 1, 'end of whisper text not reached'                            

                else:    
                    if 'conv' in func_name or 'logmel' in str(function_path_without_suffix):
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

            # import ipdb;ipdb.set_trace()

        else:
            func_dict = np.load(function_path_without_suffix.with_suffix(".npz"))
            func = func_dict[func_name]
            if nn_neuron in ('avr', 'ave', 'mean', 'all'):
                func = np.mean(func[:, :400_000], axis=0) #func[nn_neuron]
            else:
                func = func[nn_neuron, :400_000]
            func_name += f'_{str(nn_neuron)}'

    else:
        if not function_path_without_suffix.with_suffix(".npz").exists():
            if function_path_without_suffix.with_suffix(".mat").exists():
                convert_stimulisig_on_disk_mat_to_npz(function_path_without_suffix)
            else:
                raise FileNotFoundError(f"{function_path_without_suffix}: neither .mat nor .npz found.")

        assert function_path_without_suffix.with_suffix(".npz").exists()

        func_dict = np.load(str(function_path_without_suffix.with_suffix(".npz")))
        func = np.array(func_dict[func_name])

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


def convert_stimulisig_on_disk_mat_to_npz(function_path_without_suffix):
    mat = loadmat(str(function_path_without_suffix.with_suffix(".mat")))['stimulisig']
    print('Saving .mat as .npz ...')
    func_dict = {}
    for key in mat.dtype.names:
        if key == 'name':
            continue
        func_dict[key] = np.array(mat[key][0, 0], dtype=np.float16)
        func_dict[key].reshape((1, -1))  # Unwrap if it's a split matlab stimulisig
    np.savez(str(function_path_without_suffix.with_suffix(".npz")), **func_dict)

def load_function_pre(function_path_without_suffix: PathType, func_name: str):
    function_path_without_suffix = Path(function_path_without_suffix)
    func: NDArray
    func_dict = np.load(function_path_without_suffix.with_suffix(".npz"))
    func = func_dict[func_name]
    return func

def load_txt(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        return [x.strip() for x in lines]    
