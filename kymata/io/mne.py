from mne import read_evokeds, minimum_norm, set_eeg_reference
import numpy as np
from numpy.typing import NDArray
from os.path import isfile


def load_single_emeg(emeg_path, need_names=False, inverse_operator=None, snr=4):
    emeg_path_npy = f"{emeg_path}.npy"
    emeg_path_fif = f"{emeg_path}.fif"
    if isfile(emeg_path_npy) and (not need_names) and (inverse_operator is None):
        ch_names: list[str] = []  # TODO: we'll need these
        emeg: NDArray = np.load(emeg_path_npy)
    else:
        evoked = read_evokeds(emeg_path_fif, verbose=False)  # should be len 1 list
        if inverse_operator is not None:
            lh_emeg, rh_emeg, ch_names = inverse_operate(evoked[0], inverse_operator, snr)
            # TODO: I think ch_names here is the wrong thing

            # TODO: currently this goes OOM (node-h04 atleast):
            #       looks like this will be faster when split up anyway
            #       note, don't run the inv_op twice for rh and lh!
            # TODO: move inverse operator to run after EMEG channel combination
            emeg = np.concatenate((lh_emeg, rh_emeg), axis=0)
            del lh_emeg, rh_emeg
        else:
            emeg = evoked[0].get_data()  # numpy array shape (sensor_num, N) = (370, 403_001)
            ch_names = evoked[0].ch_names
            emeg /= np.max(emeg, axis=1, keepdims=True)
            if not isfile(emeg_path_npy):
                np.save(emeg_path_npy, np.array(emeg, dtype=np.float16))
        del evoked
    return emeg, ch_names


def inverse_operate(evoked, inverse_operator, snr=4):
    lambda2 = 1.0 / snr ** 2
    inverse_operator = minimum_norm.read_inverse_operator(inverse_operator, verbose=False)
    set_eeg_reference(evoked, projection=True, verbose=False)
    stc = minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, 'MNE', pick_ori='normal', verbose=False)
    print("Inverse operator applied")
    return stc.lh_data, stc.rh_data, stc.lh_vertno


def load_emeg_pack(emeg_paths, need_names=False, ave_mode=None, inverse_operator=None, p_tshift=None, snr=4):  # TODO: FIX PRE-AVE-NORMALISATION
    if p_tshift is None:
        p_tshift = [0]*len(emeg_paths)
    emeg, emeg_names = load_single_emeg(emeg_paths[0], need_names, inverse_operator, snr)
    emeg = emeg[:,p_tshift[0]:402001 + p_tshift[0]]
    emeg = np.expand_dims(emeg, 1)
    if ave_mode == 'add':
        for i in range(1, len(emeg_paths)):
            t_shift = p_tshift[i]
            new_emeg = load_single_emeg(emeg_paths[i], need_names, inverse_operator, snr)[0][:,t_shift:402001 + t_shift]
            emeg = np.concatenate((emeg, np.expand_dims(new_emeg, 1)), axis=1)
    elif ave_mode == 'ave':
        for i in range(1, len(emeg_paths)):
            t_shift = p_tshift[i]
            emeg += np.expand_dims(load_single_emeg(emeg_paths[i], need_names, inverse_operator, snr)[0][:,t_shift:402001 + t_shift], 1)
    elif len(emeg_paths) > 1:
        raise NotImplementedError(f'ave_mode "{ave_mode}" not known')
    return emeg, emeg_names

