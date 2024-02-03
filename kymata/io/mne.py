from os.path import isfile
from pathlib import Path
from typing import Optional

import numpy as np
import mne


def load_single_emeg(emeg_path: Path, need_names=False, inverse_operator=None, snr=4, morph_path: Optional[Path] = None):
    """
    When using the inverse operator, returns left and right hemispheres concatenated
    """
    emeg_path_npy = emeg_path.with_suffix(".npy")
    emeg_path_fif = emeg_path.with_suffix(".fif")
    if isfile(emeg_path_npy) and (not need_names) and (inverse_operator is None) and (morph_path is None):
        ch_names: list[str] = []  # TODO: we'll need these
        emeg = np.load(emeg_path_npy)
    else:
        evoked = mne.read_evokeds(emeg_path_fif, verbose=False)  # should be len 1 list
        if inverse_operator is not None:
            morph_map = mne.read_source_morph(morph_path) if morph_path is not None else None
            lh_emeg, rh_emeg, ch_names = inverse_operate(evoked[0], inverse_operator, snr, morph_map=morph_map)

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


def inverse_operate(evoked, inverse_operator, snr=4, morph_map = None):
    lambda2 = 1.0 / snr ** 2
    inverse_operator = mne.minimum_norm.read_inverse_operator(inverse_operator, verbose=False)
    mne.set_eeg_reference(evoked, projection=True, verbose=False)
    stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, 'MNE', pick_ori='normal', verbose=False)
    print("Inverse operator applied")

    if morph_map is not None:
        stc = morph_map.apply(stc)

    return stc.lh_data, stc.rh_data, stc.vertices


def load_emeg_pack(emeg_filenames, emeg_dir, morph_dir, need_names=False, ave_mode=None, inverse_operator_dir=None, inverse_operator_suffix=None, p_tshift=None, snr=4,
                   use_morph: bool = False):
    # TODO: FIX PRE-AVE-NORMALISATION
    emeg_paths = [
        Path(emeg_dir, emeg_fn)
        for emeg_fn in emeg_filenames
    ]
    morph_paths = [
        Path(morph_dir, f"{_strip_ave(emeg_fn)}_fsaverage_morph.h5") if use_morph else None
        for emeg_fn in emeg_filenames
    ]
    inverse_operator_paths = [
        Path(inverse_operator_dir, f"{_strip_ave(emeg_fn)}{inverse_operator_suffix}")
        for emeg_fn in emeg_filenames
    ]
    if p_tshift is None:
        p_tshift = [0]*len(emeg_paths)
    emeg, emeg_names = load_single_emeg(emeg_paths[0], need_names, inverse_operator_paths[0], snr, morph_paths[0])
    emeg=emeg[:, p_tshift[0]:402001 + p_tshift[0]]
    emeg = np.expand_dims(emeg, 1)
    if ave_mode == 'add':
        for i in range(1, len(emeg_paths)):
            t_shift = p_tshift[i]
            new_emeg = load_single_emeg(emeg_paths[i], need_names, inverse_operator_paths[i], snr,
                                        morph_paths[i])[0][:, t_shift:402001 + t_shift]
            emeg = np.concatenate((emeg, np.expand_dims(new_emeg, 1)), axis=1)
    elif ave_mode == 'ave':
        for i in range(1, len(emeg_paths)):
            t_shift = p_tshift[i]
            emeg += np.expand_dims(load_single_emeg(emeg_paths[i], need_names, inverse_operator_paths[i], snr,
                                                    morph_paths[i])[0][:, t_shift:402001 + t_shift], 1)
    elif len(emeg_paths) > 1:
        raise NotImplementedError(f'ave_mode "{ave_mode}" not known')
    return emeg, emeg_names


def _strip_ave(name: str) -> str:
    if name.endswith("-ave"):
        return name[:-4]
    else:
        return name

