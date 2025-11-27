from os.path import isfile
from pathlib import Path
from typing import Optional
from logging import getLogger
from warnings import warn
from scipy.signal import resample

import numpy as np
from numpy.typing import NDArray
import mne

from kymata.io.file import PathType
from kymata.preproc.premorph import (
    pick_channels_inverse_operator,
    premorph_inverse_operator,
)

_logger = getLogger(__name__)


def load_single_emeg(
    emeg_path: Path,
    need_names=False,
    inverse_operator_path: Optional[Path] = None,
    snr=4,
    morph_path: Optional[Path] = None,
    old_morph=False,
    premorphed_inverse_operator_path: Optional[Path] = None,
    ch_names_path: Optional[Path] = None,
) -> tuple[NDArray, list[str]]:
    """
    When using the inverse operator, returns left and right hemispheres concatenated.

    old_morph: forces loading of mne morph map
    """
    emeg_path_npy = emeg_path.with_suffix(".npy")
    emeg_path_fif = emeg_path.with_suffix(".fif")

    if 'ecog' in emeg_path.name.lower() or 'ieeg' in emeg_path.name.lower():

        ch_names_path = Path(emeg_path.parent, "ch_names_kmeans300.npy")
        channel_names: list[str] = np.load(ch_names_path, allow_pickle=True)
        emeg = np.load(emeg_path_npy)
        emeg = resample(emeg, 360000, axis=1)

        return emeg, channel_names

    else:

        if inverse_operator_path is None:
            ch_names_path = Path(emeg_path.parent, "ch_names.npy")

        if (
            isfile(emeg_path_npy)
            and (not need_names)
            and (inverse_operator_path is None)
            and (morph_path is None)
        ):
            # Load npy-format sensor data
            channel_names: list[str] = np.load(ch_names_path)
            emeg = np.load(emeg_path_npy)

            return emeg, channel_names

        _logger.info(f"Reading EMEG evokeds from {emeg_path_fif}")
        evoked = mne.read_evokeds(emeg_path_fif, verbose=False)
        assert len(evoked) == 1
        evoked = evoked[0]

        if inverse_operator_path is None:
            # Want sensor data
            emeg = evoked.get_data()  # numpy array shape (sensor_num, N) = (370, 403_001)
            return emeg, evoked.ch_names

        if old_morph:
            # Load and apply fif-format morph data
            _logger.info(f"Reading source morph from {morph_path}")
            morph_map = (
                mne.read_source_morph(morph_path)
                if morph_path is not None
                else None
            )

            lh_emeg, rh_emeg, morph_hexel_names = inverse_operate(evoked, inverse_operator_path, snr, morph_map=morph_map)
            # Stack into a single matrix, to be split after gridsearch
            emeg = np.concatenate((lh_emeg, rh_emeg), axis=0)

            return emeg, morph_hexel_names

        if premorphed_inverse_operator_path is not None:
            common_channels_path = Path(
                premorphed_inverse_operator_path.parent,
                Path(premorphed_inverse_operator_path.name).stem
                + "_list_of_common_channels",
            )
            if not Path(premorphed_inverse_operator_path).exists():
                # Compute premorphed operator path
                inverse_operator, premorphed_inverse_operator, morph_hexel_names = (
                    premorph_inverse_operator(
                        morph_path,
                        evoked,
                        inverse_operator_path,
                        snr**-2,
                        "MNE",
                        pick_ori="normal",
                    )
                )

                # Common channels to restrict to
                common_channels = pick_channels_inverse_operator(evoked.ch_names, inverse_operator)

                np.save(premorphed_inverse_operator_path, premorphed_inverse_operator)
                np.save(ch_names_path, morph_hexel_names)
                np.save(common_channels_path, common_channels)

            else:
                # Load precomputed
                premorphed_inverse_operator = np.load(premorphed_inverse_operator_path)
                morph_hexel_names = np.load(ch_names_path, allow_pickle=True)
                common_channels = np.load(common_channels_path.with_suffix(".npy"))

            emeg = np.matmul(
                premorphed_inverse_operator,
                # Restrict to common channels
                evoked.data[common_channels],
            )

            del evoked, premorphed_inverse_operator

        else:
            raise ValueError("Please supply premorphed_inverse_operator_path or old_morph.")

        return emeg, morph_hexel_names


def inverse_operate(evoked, inverse_operator, snr=4, morph_map: Optional[mne.SourceMorph] = None):
    lambda2 = 1.0 / snr**2
    _logger.info(f"Reading inverse operator from {inverse_operator}")
    inverse_operator = mne.minimum_norm.read_inverse_operator(inverse_operator, verbose=False)
    _logger.info("Applying inverse operator")
    stc: mne.VectorSourceEstimate = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, "MNE",
                                                                   pick_ori="normal", verbose=False)
    _logger.info("Inverse operator applied")

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


def __morph_apply(
    morph: mne.SourceMorph,
    stc_from,
    output="stc",
    mri_resolution=False,
    mri_space=None,
    verbose=None,
):
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
    from mne.morph import (
        _BaseSurfaceSourceEstimate,
        _BaseVolSourceEstimate,
        _check_vertices_match,
        _VOL_MAT_CHECK_RATIO,
    )

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
        vertices_to = vertices_to[0 if do_surf else 2 : None if do_vol else 2]
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
        if morph.vol_morph_mat is None and n_times >= _VOL_MAT_CHECK_RATIO * (to_vol_stop - to_surf_stop):
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
        for hemi, v1, v2 in zip(("left", "right"),
                                morph.src_data["vertices_from"],
                                stc_from.vertices[:2]):
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


def load_emeg_pack(
    emeg_filenames,
    emeg_dir: PathType,
    inverse_operator_dir: Optional[PathType],
    ch_names_path: Path,
    morph_dir: Optional[PathType] = None,
    need_names=False,
    ave_mode=None,
    inverse_operator_suffix=None,
    snr=4,
    old_morph=False,
    invsol_npy_dir=None,
):
    """

    Args:
        emeg_filenames:
            Either: a list of reps for a single participant,
            Or: a list of the rep-average (-ave) for all participants
        emeg_dir:
        inverse_operator_dir:
        ch_names_path:
        morph_dir:
        need_names:
        ave_mode: "ave" or "concatenate"
            "ave" = average over all repetitions
            "concatenate" = treat all repetitions as if it's a single continuous stimulus
        inverse_operator_suffix:
        snr:
        old_morph:
        invsol_npy_dir:

    Returns:
        emeg: NDArray
        emeg_names: list[str]
        n_reps: int
            The number of repetitions present in the emeg array
    """

    emeg_paths = [Path(emeg_dir, emeg_fn) for emeg_fn in emeg_filenames]
    n_reps = len(emeg_paths)
    morph_paths = [
        Path(morph_dir, f"{_strip_ave(emeg_fn)}_fsaverage_morph.h5")
        if morph_dir is not None
        else None
        for emeg_fn in emeg_filenames
    ]
    invsol_paths = [
        Path(
            invsol_npy_dir,
            f"{_strip_ave(emeg_fn)}{_strip_file_ext(inverse_operator_suffix)}_{_strip_ave(emeg_fn)}_fsaverage_morph.npy",
        )
        for emeg_fn in emeg_filenames
    ]
    if inverse_operator_dir is not None:
        inverse_operator_paths = [
            Path(inverse_operator_dir, f"{_strip_ave(emeg_fn)}{inverse_operator_suffix}")
            for emeg_fn in emeg_filenames
        ]
    else:
        inverse_operator_paths = [None] * len(emeg_filenames)

    # Load first one
    try:
        emeg, emeg_names = load_single_emeg(
            emeg_paths[0],
            need_names,
            inverse_operator_paths[0],
            snr,
            morph_paths[0],
            old_morph=old_morph,
            premorphed_inverse_operator_path=invsol_paths[0],
            ch_names_path=ch_names_path,
        )
    except Exception as ex:
        _logger.error(f"Error loading EMEG data from {str(emeg_paths[0])}")
        _logger.error(f"\tinverse operator {str(inverse_operator_paths[0])} or {str(invsol_paths[0])}")
        _logger.error(f"\tmorph {str(morph_paths[0])}")
        raise ex
    emeg = np.expand_dims(emeg, 1)

    # Load remaining ones in using the appropriate ave_mode
    if ave_mode == "concatenate":
        # Concatenating all reps (or participant_averages - although this would be a non-standard use) into a single long stimulus
        for i in range(1, len(emeg_paths)):
            new_emeg, _ch_names = load_single_emeg(
                emeg_paths[i],
                need_names,
                inverse_operator_paths[i],
                snr,
                morph_paths[i],
                old_morph=old_morph,
                premorphed_inverse_operator_path=invsol_paths[i],
                ch_names_path=ch_names_path,
            )
            emeg = np.concatenate((emeg, np.expand_dims(new_emeg, 1)), axis=1)

    elif ave_mode == "ave":
        # Averaging together all participants

        for i in range(1, len(emeg_paths)):
            new_emeg, _ch_names = load_single_emeg(
                emeg_paths[i],
                need_names,
                inverse_operator_paths[i],
                snr,
                morph_paths[i],
                old_morph=old_morph,
                premorphed_inverse_operator_path=invsol_paths[i],
                ch_names_path=ch_names_path,
            )
            emeg += np.expand_dims(new_emeg, 1)

        n_reps = 1  # n_reps is now 1 as all averaged

    elif len(emeg_paths) > 1:
        raise NotImplementedError(f'ave_mode "{ave_mode}" not known')

    return emeg.astype(np.float32), emeg_names, n_reps


def _strip_ave(name: str) -> str:
    if name.endswith("-ave"):
        return name[:-4]
    else:
        return name


def _strip_file_ext(name: str) -> str:
    """Returns a string, minus the final `.` and anything following it."""
    parts = name.split(".")
    return ".".join(parts[:-1])
