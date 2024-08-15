from logging import getLogger
from pathlib import Path
import os
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from kymata.entities.functions import Function
from kymata.io.file import PathType

import torch

_logger = getLogger(__file__)


def load_function(function_path_without_suffix: PathType, func_name: str, replace_nans: Optional[bool] = None,
                  n_derivatives: int = 0, n_hamming: int = 0, bruce_neurons: tuple = (5, 10), nn_neuron: str = 'ave', mfa: bool = False) -> Function:
    function_path_without_suffix = Path(function_path_without_suffix)
    func: NDArray
    if 'neurogram' in func_name:
        _logger.info('USING BRUCE MODEL')
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

    elif 'asr_models' in str(function_path_without_suffix):
        if 'whisper' in str(function_path_without_suffix):
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
                            asr_text = [i.lower() for i in load_txt(f'{function_path_without_suffix}_whisper_transcription.txt') if i != '<|startoftranscript|>']
                            # mfa_text = load_txt(f'{function_path_without_suffix}_mfa_text.txt')
                            # mfa_time = np.array(load_txt(f'{function_path_without_suffix}_mfa_stime.txt')).astype(float)
                            mfa_text = load_txt(Path(function_path_without_suffix.parent, 'teacher_mfa_text.txt'))
                            mfa_time = np.array(load_txt(Path(function_path_without_suffix.parent, 'teacher_mfa_stime.txt'))).astype(float)
                            mfa_time_samples = (mfa_time * 1000).astype(int)
                            mfa_time_samples = np.append(mfa_time_samples, 402_000)
                            special_tokens = ['<|en|>', '<|transcribe|>', '<|notimestamps|>', '<|endoftext|>']
                            k = 0       # k is the index for whisper space, and i is the index for mfa space
                            for i in range(len(mfa_time_samples) - 1):
                                start_idx = mfa_time_samples[i]
                                end_idx = mfa_time_samples[i + 1]
                                if start_idx < s_num:
                                    while asr_text[k] in special_tokens:
                                        k += 1
                                    if mfa_text[i] != asr_text[k]:
                                        if mfa_text[i] == '<sp>':
                                            # if asr_text[k] == '.' or asr_text[k] == ',':
                                            place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                                            # print('<sp> in mfa encountered')
                                        else:
                                            search_txt = asr_text[k]
                                            id_tracker = [k]
                                            if os.path.exists(f'{function_path_without_suffix}_timestamp.npy'):
                                                weight_tracker = [time_stamps_samples[k+1]-time_stamps_samples[k]]
                                            # combine word pieces in asr text, and also combine potential '.' and ',' to the following word in whisper
                                            while len(search_txt) < len(mfa_text[i]) + 1 and mfa_text[i] not in search_txt:
                                                k += 1
                                                if asr_text[k] not in special_tokens:
                                                    search_txt += asr_text[k]
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
                                            # print(f'mapping is from the mfa token {[mfa_text[i]]} to the whisper token {[asr_text[k] for k in id_tracker]}')
                                    else:
                                        place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                                        k += 1
                                        # print('match')
                            # import ipdb;ipdb.set_trace()
                            assert k == len(asr_text) - 1, 'end of asr text not reached'                            

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
                        asr_text = [i.lower() for i in load_txt(f'{function_path_without_suffix}_whisper_transcription.txt') if i != '<|startoftranscript|>']
                        mfa_text = load_txt(Path(function_path_without_suffix.parent, 'teacher_mfa_text.txt'))
                        mfa_time = np.array(load_txt(Path(function_path_without_suffix.parent, 'teacher_mfa_stime.txt'))).astype(float)
                        mfa_time_samples = (mfa_time * 1000).astype(int)
                        mfa_time_samples = np.append(mfa_time_samples, 402_000)
                        special_tokens = ['<|en|>', '<|transcribe|>', '<|notimestamps|>', '<|endoftext|>']
                        k = 0       # k is the index for whisper space, and i is the index for mfa space
                        for i in range(len(mfa_time_samples) - 1):
                            start_idx = mfa_time_samples[i]
                            end_idx = mfa_time_samples[i + 1]
                            if start_idx < s_num:
                                while asr_text[k] in special_tokens:
                                    k += 1
                                if mfa_text[i] != asr_text[k]:
                                    if mfa_text[i] == '<sp>':
                                        # if asr_text[k] == '.' or asr_text[k] == ',':
                                        place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                                        # print('<sp> in mfa encountered')
                                    else:
                                        search_txt = asr_text[k]
                                        id_tracker = [k]
                                        if os.path.exists(f'{function_path_without_suffix}_timestamp.npy'):
                                            weight_tracker = [time_stamps_samples[k+1]-time_stamps_samples[k]]
                                        # combine word pieces in asr text, and also combine potential '.' and ',' to the following word in whisper
                                        while len(search_txt) < len(mfa_text[i]) + 1 and mfa_text[i] not in search_txt:
                                            k += 1
                                            if asr_text[k] not in special_tokens:
                                                search_txt += asr_text[k]
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
                                        # print(f'mapping is from the mfa token {[mfa_text[i]]} to the whisper token {[asr_text[k] for k in id_tracker]}')
                                else:
                                    place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                                    k += 1
                                    # print('match')
                        # import ipdb;ipdb.set_trace()
                        # assert k == len(asr_text) - 1, 'end of asr text not reached'                            

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
        elif 'salmonn' in str(function_path_without_suffix):
            for s in range(14):
                if s == 0:
                    func = torch.load(Path(function_path_without_suffix, f'segment_{s}_{func_name}.pt'), map_location=torch.device('cpu')).detach().numpy()
                else:
                    func = np.concatenate((func, torch.load(Path(function_path_without_suffix, f'segment_{s}_{func_name}.pt'), map_location=torch.device('cpu')).detach().numpy()), axis = 1)
            T_max = 401
            s_num = T_max * 1000
            place_holder = np.zeros((func.shape[2], s_num))

            asr_text = []
            for s in range(14):
                # Read the content of the file
                with open(Path(function_path_without_suffix, f'segment_{s}.txt'), 'r') as file:
                    content = file.read()
                # The content is a string representation of a list, so we need to evaluate it
                word_pieces = eval(content)
                # Remove the '▁' from each word piece
                asr_text += [word.replace('▁', '').lower() for word in word_pieces if word != '</s>']

            mfa_text = load_txt(Path(function_path_without_suffix.parent, 'teacher_mfa_text.txt'))
            mfa_time = np.array(load_txt(Path(function_path_without_suffix.parent, 'teacher_mfa_stime.txt'))).astype(float)
            mfa_time_samples = (mfa_time * 1000).astype(int)
            mfa_time_samples = np.append(mfa_time_samples, 402_000)
            if nn_neuron in ('avr', 'ave', 'mean', 'all'):
                for j in range(place_holder.shape[0]):
                    k = 0       # k is the index for salmonn space, and i is the index for mfa space
                    for i in range(len(mfa_time_samples) - 1):
                        start_idx = mfa_time_samples[i]
                        end_idx = mfa_time_samples[i + 1]
                        if start_idx < s_num:
                            if mfa_text[i] != asr_text[k]:
                                if mfa_text[i] == '<sp>':
                                    # if asr_text[k] == '.' or asr_text[k] == ',':
                                    place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                                    # print('<sp> in mfa encountered')
                                else:
                                    search_txt = asr_text[k]
                                    id_tracker = [k]
                                    # combine word pieces in asr text, and also combine potential '.' and ',' to the following word in salmonn
                                    while len(search_txt) < len(mfa_text[i]) + 1 and mfa_text[i] not in search_txt:
                                        k += 1
                                        search_txt += asr_text[k]
                                        id_tracker.append(k)
                                    place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) , np.average([func[0, k, j] for k in id_tracker]))
                                    k += 1
                                    # print(f'mapping is from the mfa token {[mfa_text[i]]} to the salmonn token {[asr_text[k] for k in id_tracker]}')
                            else:
                                place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                                k += 1
                                # print('match')
                    assert k == len(asr_text) - 1, 'end of asr text not reached'       
                    import ipdb;ipdb.set_trace()                     
            else:
                j = nn_neuron
                k = 0       # k is the index for salmonn space, and i is the index for mfa space
                for i in range(len(mfa_time_samples) - 1):
                    start_idx = mfa_time_samples[i]
                    end_idx = mfa_time_samples[i + 1]
                    if start_idx < s_num:
                        if mfa_text[i] != asr_text[k]:
                            if mfa_text[i] == '<sp>':
                                # if asr_text[k] == '.' or asr_text[k] == ',':
                                place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                                # print('<sp> in mfa encountered')
                            else:
                                search_txt = asr_text[k]
                                id_tracker = [k]
                                # combine word pieces in asr text, and also combine potential '.' and ',' to the following word in salmonn
                                while len(search_txt) < len(mfa_text[i]) + 1 and mfa_text[i] not in search_txt:
                                    k += 1
                                    search_txt += asr_text[k]
                                    id_tracker.append(k)
                                place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) , np.average([func[0, k, j] for k in id_tracker]))
                                k += 1
                                # print(f'mapping is from the mfa token {[mfa_text[i]]} to the salmonn token {[asr_text[k] for k in id_tracker]}')
                        else:
                            place_holder[j, start_idx:end_idx] = np.full((min(end_idx, s_num) - start_idx, ) ,func[0, k, j])
                            k += 1
                            # print('match')
                # import ipdb;ipdb.set_trace()
                # assert k == len(asr_text) - 1, 'end of asr text not reached'                        

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
        if not function_path_without_suffix.with_suffix(".npz").exists():
            if function_path_without_suffix.with_suffix(".mat").exists():
                convert_stimulisig_on_disk_mat_to_npz(function_path_without_suffix)
            else:
                raise FileNotFoundError(f"{function_path_without_suffix}: neither .mat nor .npz found.")

        assert function_path_without_suffix.with_suffix(".npz").exists()

        func_dict = np.load(str(function_path_without_suffix.with_suffix(".npz")))
        func = np.array(func_dict[func_name], dtype=float)

    n_nan = np.count_nonzero(np.isnan(func))
    if n_nan > 0:
        nan_message = f"Function contained {n_nan:,} NaNs, ({100*n_nan/np.prod(func.shape):.2f}%)"
        if replace_nans is None:
            raise ValueError(f"{nan_message}. "
                             "Consider replacing small numbers of unavoidable NaNs with zeros or the function mean "
                             "value.")
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
        func = np.convolve(func, [-1, 1], 'same')  # derivative

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
    mat = loadmat(str(function_path_without_suffix.with_suffix(".mat")))['stimulisig']
    _logger.info('Saving .mat as .npz ...')
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
