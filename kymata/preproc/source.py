from os.path import isfile
from pathlib import Path
from typing import Optional
from logging import getLogger
from warnings import warn

import numpy as np
import mne

from kymata.io.file import PathType

_logger = getLogger(__name__)


def load_single_emeg(emeg_path: Path, need_names=False, inverse_operator=None, snr=4, morph_path: Optional[Path] = None):
    """
    When using the inverse operator, returns left and right hemispheres concatenated
    """
    emeg_path_npy = emeg_path.with_suffix(".npy")
    emeg_path_fif = emeg_path.with_suffix(".fif")
    morph_path_h5  = morph_path.with_suffix(".h5") if morph_path is not None else None
    morph_path_fif = morph_path.with_suffix(".fif") if morph_path is not None else None
    if isfile(emeg_path_npy) and (not need_names) and (inverse_operator is None) and (morph_path is None):
        ch_names: list[str] = []
        emeg = np.load(emeg_path_npy)
    else:
        _logger.info(f"Reading EMEG evokeds from {emeg_path_fif}")
        evoked = mne.read_evokeds(emeg_path_fif, verbose=False)  # should be len 1 list
        if inverse_operator is not None:
            if morph_path is not None:
                _logger.info(f"Reading source morph from {morph_path_h5}")
                try:
                    morph_map = mne.read_source_morph(morph_path_h5)
                except FileNotFoundError:
                    morph_map = mne.read_source_morph(morph_path_fif)
            else:
                morph_map = None
            lh_emeg, rh_emeg, ch_names = inverse_operate(evoked[0], inverse_operator, snr, morph_map=morph_map)
            # Stack into a single matrix, to be split after gridsearch
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


def inverse_operate(evoked, inverse_operator, snr=4, morph_map: Optional[mne.SourceMorph] = None):
    lambda2 = 1.0 / snr ** 2
    _logger.info(f"Reading inverse operator from {inverse_operator}")
    inverse_operator = mne.minimum_norm.read_inverse_operator(inverse_operator, verbose=False)
    mne.set_eeg_reference(evoked, projection=True, verbose=False)
    _logger.info("Applying inverse operator")
    stc: mne.VectorSourceEstimate = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, 'MNE', pick_ori='normal', verbose=False)
    print("Inverse operator applied")

    if morph_map is not None:
        stc = apply_morph_map(morph_map, stc)

    return stc.lh_data, stc.rh_data, stc.vertices


def apply_morph_map(morph_map: mne.SourceMorph, stc: mne.VectorSourceEstimate):
    _logger.info("Applying morph map")
    # morph_map.apply() is very slow in some cases, for unknown reasons.
    # So we instead use a copied, patched version of the same code to make it faster,
    # at the cost of somewhat increased memory usage.
    # stc = morph_map.apply(stc)
    stc = __morph_apply(morph_map, stc)
    return stc


def __morph_apply(morph: mne.SourceMorph, stc_from, output="stc", mri_resolution=False, mri_space=None, verbose=None):
    """A copy of mne.SourceMorph.apply, for optimisation."""
    import copy
    from mne.morph import _morphed_stc_as_volume
    # _validate_type(output, str, "output")
    # _validate_type(stc_from, _BaseSourceEstimate, "stc_from", "source estimate")
    # if isinstance(stc_from, _BaseSurfaceSourceEstimate):
    #     allowed_kinds = ("stc",)
    #     extra = "when stc is a surface source estimate"
    # else:
    #     allowed_kinds = ("stc", "nifti1", "nifti2")
    #     extra = ""
    # _check_option("output", output, allowed_kinds, extra)
    stc = copy.deepcopy(stc_from)

    mri_space = mri_resolution if mri_space is None else mri_space
    if stc.subject is None:
        stc.subject = morph.subject_from
    if morph.subject_from is None:
        morph.subject_from = stc.subject
    if stc.subject != morph.subject_from:
        raise ValueError(
            "stc_from.subject and "
            "morph.subject_from must match. (%s != %s)"
            % (stc.subject, morph.subject_from)
        )
    out = __mne_apply_morph_data(morph, stc)
    if output != "stc":  # convert to volume
        out = _morphed_stc_as_volume(
            morph,
            out,
            mri_resolution=mri_resolution,
            mri_space=mri_space,
            output=output,
        )
    return out


def __mne_apply_morph_data(morph, stc_from):
    """A copy of mne.morph._apply_morph_data, for optimisation."""
    from mne.morph import _BaseSurfaceSourceEstimate, _BaseVolSourceEstimate, _check_vertices_match, \
        _VOL_MAT_CHECK_RATIO
    if stc_from.subject is not None and stc_from.subject != morph.subject_from:
        raise ValueError(
            "stc.subject (%s) != morph.subject_from (%s)"
            % (stc_from.subject, morph.subject_from)
        )
    # _check_option("morph.kind", morph.kind, ("surface", "volume", "mixed"))
    # if morph.kind == "surface":
    #     _validate_type(
    #         stc_from,
    #         _BaseSurfaceSourceEstimate,
    #         "stc_from",
    #         "volume source estimate when using a surface morph",
    #     )
    # elif morph.kind == "volume":
    #     _validate_type(
    #         stc_from,
    #         _BaseVolSourceEstimate,
    #         "stc_from",
    #         "surface source estimate when using a volume morph",
    #     )
    # else:
    #     assert morph.kind == "mixed"  # can handle any
    #     _validate_type(
    #         stc_from,
    #         _BaseSourceEstimate,
    #         "stc_from",
    #         "source estimate when using a mixed source morph",
    #     )

    # figure out what to actually morph
    do_vol = not isinstance(stc_from, _BaseSurfaceSourceEstimate)
    do_surf = not isinstance(stc_from, _BaseVolSourceEstimate)

    vol_src_offset = 2 if do_surf else 0
    from_surf_stop = sum(len(v) for v in stc_from.vertices[:vol_src_offset])
    to_surf_stop = sum(len(v) for v in morph.vertices_to[:vol_src_offset])
    from_vol_stop = stc_from.data.shape[0]
    vertices_to = morph.vertices_to
    if morph.kind == "mixed":
        vertices_to = vertices_to[0 if do_surf else 2: None if do_vol else 2]
    to_vol_stop = sum(len(v) for v in vertices_to)

    mesg = "Ori × Time" if stc_from.data.ndim == 3 else "Time"
    data_from = np.reshape(stc_from.data, (stc_from.data.shape[0], -1))
    n_times = data_from.shape[1]  # oris treated as times
    data = np.empty((to_vol_stop, n_times), stc_from.data.dtype)
    to_used = np.zeros(data.shape[0], bool)
    from_used = np.zeros(data_from.shape[0], bool)
    if do_vol:
        stc_from_vertices = stc_from.vertices[vol_src_offset:]
        vertices_from = morph._vol_vertices_from
        for ii, (v1, v2) in enumerate(zip(vertices_from, stc_from_vertices)):
            _check_vertices_match(v1, v2, "volume[%d]" % (ii,))
        from_sl = slice(from_surf_stop, from_vol_stop)
        assert not from_used[from_sl].any()
        from_used[from_sl] = True
        to_sl = slice(to_surf_stop, to_vol_stop)
        assert not to_used[to_sl].any()
        to_used[to_sl] = True
        # Loop over time points to save memory
        if morph.vol_morph_mat is None and n_times >= _VOL_MAT_CHECK_RATIO * (
                to_vol_stop - to_surf_stop
        ):
            warn(
                "Computing a sparse volume morph matrix will save time over "
                "directly morphing, calling morph.compute_vol_morph_mat(). "
                "Consider (re-)saving your instance to disk to avoid "
                "subsequent recomputation."
            )
            morph.compute_vol_morph_mat()
        if morph.vol_morph_mat is None:
            _logger.debug("Using individual volume morph")
            data[to_sl, :] = morph._morph_vols(data_from[from_sl], mesg)
        else:
            _logger.debug("Using sparse volume morph matrix")
            data[to_sl, :] = morph.vol_morph_mat @ data_from[from_sl]
    if do_surf:
        for hemi, v1, v2 in zip(
                ("left", "right"), morph.src_data["vertices_from"], stc_from.vertices[:2]
        ):
            _check_vertices_match(v1, v2, "%s hemisphere" % (hemi,))
        from_sl = slice(0, from_surf_stop)
        assert not from_used[from_sl].any()
        from_used[from_sl] = True
        to_sl = slice(0, to_surf_stop)
        assert not to_used[to_sl].any()
        to_used[to_sl] = True
        #
        #
        # Our modification
        # data[to_sl] = morph.morph_mat * data_from[from_sl]
        data[to_sl] = morph.morph_mat.todense() @ data_from[from_sl]
        #
        #
        #
    assert to_used.all()
    assert from_used.all()
    data.shape = (data.shape[0],) + stc_from.data.shape[1:]
    klass = stc_from.__class__
    stc_to = klass(data, vertices_to, stc_from.tmin, stc_from.tstep, morph.subject_to)
    return stc_to


def load_emeg_pack(emeg_filenames,
                   emeg_dir: PathType,
                   inverse_operator_dir: Optional[PathType],
                   morph_dir: Optional[PathType] = None,
                   need_names=False,
                   ave_mode=None,
                   inverse_operator_suffix=None,
                   p_tshift=None,
                   snr=4):
    emeg_paths = [
        Path(emeg_dir, emeg_fn)
        for emeg_fn in emeg_filenames
    ]
    n_reps = len(emeg_paths)
    morph_paths = [
        Path(morph_dir, f"{_strip_ave(emeg_fn)}_fsaverage_morph") if morph_dir is not None else None
        for emeg_fn in emeg_filenames
    ]
    if inverse_operator_dir is not None:
        inverse_operator_paths = [
            Path(inverse_operator_dir, f"{_strip_ave(emeg_fn)}{inverse_operator_suffix}")
            for emeg_fn in emeg_filenames
        ]
    else:
        inverse_operator_paths = [None] * len(emeg_filenames)
    if p_tshift is None:
        p_tshift = [0]*len(emeg_paths)
    emeg, emeg_names = load_single_emeg(emeg_paths[0], need_names, inverse_operator_paths[0], snr, morph_paths[0])
    emeg=emeg[:, p_tshift[0]:402001 + p_tshift[0]]
    emeg = np.expand_dims(emeg, 1)
    if ave_mode == 'concatenate':
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
        n_reps = 1 # n_reps is now 1 as all averaged
    elif len(emeg_paths) > 1:
        raise NotImplementedError(f'ave_mode "{ave_mode}" not known')
    return emeg, emeg_names, n_reps


def _strip_ave(name: str) -> str:
    if name.endswith("-ave"):
        return name[:-4]
    else:
        return name

