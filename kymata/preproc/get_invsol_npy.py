"""
NOTE: MOST OF THIS CODE IS LIFTED FROM THE MNE GITHUB REPOSITORY AT: https://github.com/mne-tools/mne-python/blob/main/mne/morph.py
"""



import mne
from mne.minimum_norm.inverse import INVERSE_METHODS, combine_xyz, prepare_inverse_operator
import time
import sys
import numpy as np

import os

from mne._fiff.constants import FIFF
from mne._fiff.pick import pick_info
from mne._fiff.proj import (
    _electrode_types,
    _needs_eeg_average_ref_proj,
)
from mne.evoked import Evoked
from mne.source_estimate import _get_src_type, _make_stc
from mne.source_space._source_space import (
    _get_src_nn,
    _get_vertno,
    label_src_vertno_sel,
)
from mne.surface import _normal_orth
from mne.utils import (
    _check_compensation_grade,
    _check_option,
    _check_src_normal,
    _validate_type,
    logger,
)




def _pick_channels_inverse_operator(ch_names, inv):
    """Return data channel indices to be used knowing an inverse operator.

    Unlike ``pick_channels``, this respects the order of ch_names.
    """
    sel = list()
    for name in inv["noise_cov"].ch_names:
        try:
            sel.append(ch_names.index(name))
        except ValueError:
            raise ValueError(
                "The inverse operator was computed with "
                f"channel {name} which is not present in "
                "the data. You should compute a new inverse "
                "operator restricted to the good data "
                "channels."
            )
    return sel



def _check_ch_names(inv, info):
    """Check that channels in inverse operator are measurements."""
    inv_ch_names = inv["eigen_fields"]["col_names"]

    if inv["noise_cov"].ch_names != inv_ch_names:
        raise ValueError(
            "Channels in inverse operator eigen fields do not "
            "match noise covariance channels."
        )
    data_ch_names = info["ch_names"]

    missing_ch_names = sorted(set(inv_ch_names) - set(data_ch_names))
    n_missing = len(missing_ch_names)
    if n_missing > 0:
        raise ValueError(
            "%d channels in inverse operator " % n_missing
            + f"are not present in the data ({missing_ch_names})"
        )
    _check_compensation_grade(inv["info"], info, "inverse")


def _check_or_prepare(inv, nave, lambda2, method, method_params, prepared, copy=True):
    """Check if inverse was prepared, or prepare it."""
    if not prepared:
        inv = prepare_inverse_operator(
            inv, nave, lambda2, method, method_params, copy=copy
        )
    elif "colorer" not in inv:
        raise ValueError(
            "inverse operator has not been prepared, but got "
            "argument prepared=True. Either pass prepared=False "
            "or use prepare_inverse_operator."
        )
    return inv


def _assemble_kernel(inv, label, method, pick_ori, use_cps=True, verbose=None):
    """Assemble the kernel.

    Simple matrix multiplication followed by combination of the current
    components. This does all the data transformations to compute the weights
    for the eigenleads.

    Parameters
    ----------
    inv : instance of InverseOperator
        The inverse operator to use. This object contains the matrices that
        will be multiplied to assemble the kernel.
    label : Label | None
        Restricts the source estimates to a given label. If None,
        source estimates will be computed for the entire source space.
    method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Use minimum norm, dSPM, sLORETA, or eLORETA.
    pick_ori : None | "normal" | "vector"
        Which orientation to pick (only matters in the case of 'normal').
    %(use_cps_restricted)s

    Returns
    -------
    K : array, shape (n_vertices, n_channels) | (3 * n_vertices, n_channels)
        The kernel matrix. Multiply this with the data to obtain the source
        estimate.
    noise_norm : array, shape (n_vertices, n_samples) | (3 * n_vertices, n_samples)
        Normalization to apply to the source estimate in order to obtain dSPM
        or sLORETA solutions.
    vertices : list of length 2
        Vertex numbers for lh and rh hemispheres that correspond to the
        vertices in the source estimate. When the label parameter has been
        set, these correspond to the vertices in the label. Otherwise, all
        vertex numbers are returned.
    source_nn : array, shape (3 * n_vertices, 3)
        The direction in cartesian coordicates of the direction of the source
        dipoles.
    """  # noqa: E501
    eigen_leads = inv["eigen_leads"]["data"]
    source_cov = inv["source_cov"]["data"]
    if method in ("dSPM", "sLORETA"):
        noise_norm = inv["noisenorm"][:, np.newaxis]
    else:
        noise_norm = None

    src = inv["src"]
    vertno = _get_vertno(src)
    source_nn = inv["source_nn"]

    if label is not None:
        vertno, src_sel = label_src_vertno_sel(label, src)

        if method not in ["MNE", "eLORETA"]:
            noise_norm = noise_norm[src_sel]

        if inv["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        eigen_leads = eigen_leads[src_sel]
        source_cov = source_cov[src_sel]
        source_nn = source_nn[src_sel]

    # vector or normal, might need to rotate
    if (
        pick_ori == "normal"
        and all(s["type"] == "surf" for s in src)
        and np.allclose(
            inv["source_nn"].reshape(inv["nsource"], 3, 3), np.eye(3), atol=1e-6
        )
    ):
        offset = 0
        eigen_leads = np.reshape(eigen_leads, (-1, 3, eigen_leads.shape[1])).copy()
        source_nn = np.reshape(source_nn, (-1, 3, 3)).copy()
        for s, v in zip(src, vertno):
            sl = slice(offset, offset + len(v))
            source_nn[sl] = _normal_orth(_get_src_nn(s, use_cps, v))
            eigen_leads[sl] = np.matmul(source_nn[sl], eigen_leads[sl])
            # No need to rotate source_cov because it should be uniform
            # (loose=1., and depth weighting is uniform across columns)
            offset = sl.stop
        eigen_leads.shape = (-1, eigen_leads.shape[2])
        source_nn.shape = (-1, 3)

    if pick_ori == "normal":
        if not inv["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
            raise ValueError(
                "Picking normal orientation can only be done "
                "with a free orientation inverse operator."
            )

        is_loose = 0 < inv["orient_prior"]["data"][0] <= 1
        if not is_loose:
            raise ValueError(
                "Picking normal orientation can only be done "
                "when working with loose orientations."
            )

    trans = np.dot(inv["eigen_fields"]["data"], np.dot(inv["whitener"], inv["proj"]))
    trans *= inv["reginv"][:, None]

    #
    #   Transformation into current distributions by weighting the eigenleads
    #   with the weights computed above
    #
    K = np.dot(eigen_leads, trans)
    if inv["eigen_leads_weighted"]:
        #
        #     R^0.5 has been already factored in
        #
        logger.info("    Eigenleads already weighted ... ")
    else:
        #
        #     R^0.5 has to be factored in
        #
        logger.info("    Eigenleads need to be weighted ...")
        K *= np.sqrt(source_cov)[:, np.newaxis]

    if pick_ori == "normal":
        K = K[2::3]

    return K, noise_norm, vertno, source_nn


def _check_ori(pick_ori, source_ori, src):
    """Check pick_ori."""
    _check_option("pick_ori", pick_ori, [None, "normal", "vector"])
    _check_src_normal(pick_ori, src)


def _check_reference(inst, ch_names=None):
    """Check for EEG ref."""
    info = inst.info
    if ch_names is not None:
        picks = [
            ci for ci, ch_name in enumerate(info["ch_names"]) if ch_name in ch_names
        ]
        info = pick_info(info, sel=picks)
    if _needs_eeg_average_ref_proj(info):
        raise ValueError(
            "EEG average reference (using a projector) is mandatory for "
            "modeling, use the method set_eeg_reference(projection=True)"
        )
    if _electrode_types(info) and info.get("custom_ref_applied", False):
        raise ValueError("Custom EEG reference is not allowed for inverse modeling.")


def _subject_from_inverse(inverse_operator):
    """Get subject id from inverse operator."""
    return inverse_operator["src"]._subject





def apply_inverse(
    evoked,
    inverse_operator,
    lambda2=1.0 / 9.0,
    method="dSPM",
    pick_ori=None,
    prepared=False,
    label=None,
    method_params=None,
    return_residual=False,
    use_cps=True,
    verbose=None,
):
    """Apply inverse operator to evoked data.

    Parameters
    ----------
    evoked : Evoked object
        Evoked data.
    inverse_operator : instance of InverseOperator
        Inverse operator.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Use minimum norm :footcite:`HamalainenIlmoniemi1994`,
        dSPM (default) :footcite:`DaleEtAl2000`,
        sLORETA :footcite:`Pascual-Marqui2002`, or
        eLORETA :footcite:`Pascual-Marqui2011`.
    %(pick_ori)s
    prepared : bool
        If True, do not call :func:`prepare_inverse_operator`.
    label : Label | None
        Restricts the source estimates to a given label. If None,
        source estimates will be computed for the entire source space.
    method_params : dict | None
        Additional options for eLORETA. See Notes for details.

        .. versionadded:: 0.16
    return_residual : bool
        If True (default False), return the residual evoked data.
        Cannot be used with ``method=='eLORETA'``.

        .. versionadded:: 0.17
    %(use_cps_restricted)s

        .. versionadded:: 0.20
    %(verbose)s

    Returns
    -------
    stc : SourceEstimate | VectorSourceEstimate | VolSourceEstimate
        The source estimates.
    residual : instance of Evoked
        The residual evoked data, only returned if return_residual is True.

    See Also
    --------
    apply_inverse_raw : Apply inverse operator to raw object.
    apply_inverse_epochs : Apply inverse operator to epochs object.
    apply_inverse_tfr_epochs : Apply inverse operator to epochs tfr object.
    apply_inverse_cov : Apply inverse operator to covariance object.

    Notes
    -----
    Currently only the ``method='eLORETA'`` has additional options.
    It performs an iterative fit with a convergence criterion, so you can
    pass a ``method_params`` :class:`dict` with string keys mapping to values
    for:

        'eps' : float
            The convergence epsilon (default 1e-6).
        'max_iter' : int
            The maximum number of iterations (default 20).
            If less regularization is applied, more iterations may be
            necessary.
        'force_equal' : bool
            Force all eLORETA weights for each direction for a given
            location equal. The default is None, which means ``True`` for
            loose-orientation inverses and ``False`` for free- and
            fixed-orientation inverses. See below.

    The eLORETA paper :footcite:`Pascual-Marqui2011` defines how to compute
    inverses for fixed- and
    free-orientation inverses. In the free orientation case, the X/Y/Z
    orientation triplet for each location is effectively multiplied by a
    3x3 weight matrix. This is the behavior obtained with
    ``force_equal=False`` parameter.

    However, other noise normalization methods (dSPM, sLORETA) multiply all
    orientations for a given location by a single value.
    Using ``force_equal=True`` mimics this behavior by modifying the iterative
    algorithm to choose uniform weights (equivalent to a 3x3 diagonal matrix
    with equal entries).

    It is necessary to use ``force_equal=True``
    with loose orientation inverses (e.g., ``loose=0.2``), otherwise the
    solution resembles a free-orientation inverse (``loose=1.0``).
    It is thus recommended to use ``force_equal=True`` for loose orientation
    and ``force_equal=False`` for free orientation inverses. This is the
    behavior used when the parameter ``force_equal=None`` (default behavior).

    References
    ----------
    .. footbibliography::
    """
    out = _apply_inverse(
        evoked,
        inverse_operator,
        lambda2,
        method,
        pick_ori,
        prepared,
        label,
        method_params,
        return_residual,
        use_cps,
    )
    # logger.info("[done]")
    return out


def _log_exp_var(data, est, prefix="    "):
    res = data - est
    var_exp = 1 - ((res * res.conj()).sum().real / (data * data.conj()).sum().real)
    var_exp *= 100
    logger.info(f"{prefix}Explained {var_exp:5.1f}% variance")
    return var_exp


def _apply_inverse(
    evoked,
    inverse_operator,
    lambda2,
    method,
    pick_ori,
    prepared,
    label,
    method_params,
    return_residual,
    use_cps,
):
    _validate_type(evoked, Evoked, "evoked")
    _check_reference(evoked, inverse_operator["info"]["ch_names"])
    _check_option("method", method, INVERSE_METHODS)
    _check_ori(pick_ori, inverse_operator["source_ori"], inverse_operator["src"])
    #
    #   Set up the inverse according to the parameters
    #
    nave = evoked.nave

    T0 = time.time()

    _check_ch_names(inverse_operator, evoked.info)

    inv = _check_or_prepare(
        inverse_operator, nave, lambda2, method, method_params, prepared, copy="non-src"
    )
    del inverse_operator

    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(evoked.ch_names, inv)
    #logger.info(f'Applying inverse operator to "{evoked.comment}"...')
    #logger.info("    Picked %d channels from the data" % len(sel))
    #logger.info("    Computing inverse...")
    K, noise_norm, vertno, source_nn = _assemble_kernel(
        inv, label, method, pick_ori, use_cps=use_cps
    )


    T1 = time.time()
    print(f'make invsol: {T1 - T0}')

    print(K.shape)
    print(evoked.data[sel].shape)

    # sol = K @ evoked.data[sel]  # apply imaging kernel
    # sol = np.dot(K, evoked.data[sel])  # apply imaging kernel
    sol = np.matmul(K, evoked.data[sel])  # apply imaging kernel

    # print(sol)

    # sol0 = sol

    T0 = time.time()
    print(f'app. invsol: {T0 - T1}')

    # logger.info("    Computing residual...")
    # x̂(t) = G ĵ(t) = C ** 1/2 U Π w(t)
    # where the diagonal matrix Π has elements πk = λk γk
    Pi = inv["sing"] * inv["reginv"]
    data_w = np.dot(inv["whitener"], np.dot(inv["proj"], evoked.data[sel]))  # C ** -0.5
    w_t = np.dot(inv["eigen_fields"]["data"], data_w)  # U.T @ data
    data_est = np.dot(
        inv["colorer"],  # C ** 0.5
        np.dot(inv["eigen_fields"]["data"].T, Pi[:, np.newaxis] * w_t),  # U
    )
    data_est_w = np.dot(inv["whitener"], np.dot(inv["proj"], data_est))
    _log_exp_var(data_w, data_est_w)
    if return_residual:
        residual = evoked.copy()
        residual.data[sel] -= data_est
    is_free_ori = inv["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI and pick_ori != "normal"

    if is_free_ori and pick_ori != "vector":
        # logger.info("    Combining the current components...")
        sol = combine_xyz(sol)

    if noise_norm is not None:
        logger.info(f"    {method}...")
        if is_free_ori and pick_ori == "vector":
            noise_norm = noise_norm.repeat(3, axis=0)
        sol *= noise_norm

    tstep = 1.0 / evoked.info["sfreq"]
    tmin = float(evoked.times[0])
    subject = _subject_from_inverse(inv)
    src_type = _get_src_type(inv["src"], vertno)

    # print(f'MSE: {np.sum((sol - sol0)**2)}')

    stc = _make_stc(
        sol,
        vertno,
        tmin=tmin,
        tstep=tstep,
        subject=subject,
        vector=(pick_ori == "vector"),
        source_nn=source_nn,
        src_type=src_type,
    )

    return (stc, residual) if return_residual else stc






def get_invsol_npy(
    morph_path,
    evoked,
    inverse_operator_path,
    lambda2=1.0 / 9.0,
    method='dSPM',
    pick_ori=None,
    prepared=False,
    label=None,
    method_params=None,
    return_residual=False,
    use_cps=True,
):
    inverse_operator = mne.minimum_norm.read_inverse_operator(inverse_operator_path, verbose=False)
    _validate_type(evoked, Evoked, "evoked")
    _check_reference(evoked, inverse_operator["info"]["ch_names"])
    _check_option("method", method, INVERSE_METHODS)
    _check_ori(pick_ori, inverse_operator["source_ori"], inverse_operator["src"])
    #
    #   Set up the inverse according to the parameters
    #
    nave = evoked.nave


    # lambda2 = 1.0 / snr ** 2

    _check_ch_names(inverse_operator, evoked.info)

    inv = _check_or_prepare(
        inverse_operator, nave, lambda2, method, method_params, prepared, copy="non-src"
    )
    del inverse_operator

    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(evoked.ch_names, inv)
    logger.info(f'Applying inverse operator to "{evoked.comment}"...')
    logger.info("    Picked %d channels from the data" % len(sel))
    logger.info("    Computing inverse...")
    K, noise_norm, vertno, source_nn = _assemble_kernel(
        inv, label, method, pick_ori, use_cps=use_cps
    )

    morph = mne.read_source_morph(morph_path)
    morph_csr = morph.morph_mat

    npy_invsol = morph_csr.dot(K)

    new_save_path = os.path.dirname(os.path.dirname(morph_path))
    new_save_path = os.path.join(new_save_path, 'npy_invsol')

    os.makedirs(new_save_path, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(inverse_operator_path))[0] + '_' + os.path.basename(morph_path)
    base_name = os.path.splitext(base_name)[0] + '.npy'
    save_path = os.path.join(new_save_path, base_name)
    ch_name_path = os.path.join(new_save_path, 'ch_names.npy')

    # Combine the new file name with the new directory

    np.save(save_path, npy_invsol)

    np.save(ch_name_path, morph.vertices_to)

    # print(np.load(ch_name_path, allow_pickle=True))

    eee = evoked.data[sel]

    # T0 = time.time()
    # inv = np.matmul(K, eee)
    # inv = morph_csr.dot(inv)
    # T1 = time.time(); print(T1 - T0)

    inv = np.matmul(npy_invsol, eee)
    # T0 = time.time(); print(T0 - T1)

    #print(f'My version of inv: {inv.shape}')
    #print(inv)

    return inv, morph.vertices_to 









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

    T0 = time.time()

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

    T1 = time.time()
    print(f'make invsol: {T1 - T0}')
    sys.stdout.flush()
    
    out = __mne_apply_morph_data(morph, stc)

    T0 = time.time()
    print(f'make invsol: {T0 - T1}')
    sys.stdout.flush()


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
    from mne.morph import _BaseSurfaceSourceEstimate, _BaseVolSourceEstimate, _check_vertices_match
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

    print(do_vol, do_surf)
    sys.stdout.flush()  # False, True 

    vol_src_offset = 2 if do_surf else 0
    from_surf_stop = sum(len(v) for v in stc_from.vertices[:vol_src_offset])
    to_surf_stop = sum(len(v) for v in morph.vertices_to[:vol_src_offset])
    # from_vol_stop = stc_from.data.shape[0]
    vertices_to = morph.vertices_to
    if morph.kind == "mixed":
        vertices_to = vertices_to[0 if do_surf else 2: None if do_vol else 2]
    to_vol_stop = sum(len(v) for v in vertices_to)

    data_from = np.reshape(stc_from.data, (stc_from.data.shape[0], -1))
    n_times = data_from.shape[1]  # oris treated as times
    data = np.empty((to_vol_stop, n_times), stc_from.data.dtype)
    to_used = np.zeros(data.shape[0], bool)
    from_used = np.zeros(data_from.shape[0], bool)

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

        # print(morph.morph_mat)
        print(type(morph.morph_mat))
        print(morph.morph_mat.todense().shape, 'morph map shape')
        print(data_from[from_sl].shape, 'data_from shape')

        sys.stdout.flush()

        # data[to_sl] = morph.morph_mat.todense() @ data_from[from_sl]
        
        data[to_sl] = morph.morph_mat.dot(data_from[from_sl])

        # result = dense_array.dot(sparse_matrix)


    assert to_used.all()
    assert from_used.all()
    data.shape = (data.shape[0],) + stc_from.data.shape[1:]
    klass = stc_from.__class__
    stc_to = klass(data, vertices_to, stc_from.tmin, stc_from.tstep, morph.subject_to)

    print('morph_output: ')
    print(data)
    return stc_to









if __name__ == '__main__':

    inverse_operator_path = "/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/interim_preprocessing_files/4_hexel_current_reconstruction/inverse-operators/participant_02_ico5-3L-loose02-cps-nodepth-fusion-inv.fif"
    emeg_path = "/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/interim_preprocessing_files/3_trialwise_sensorspace/evoked_data/participant_02-ave.fif"
    morph_path = "/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/interim_preprocessing_files/4_hexel_current_reconstruction/morph_maps/participant_02_fsaverage_morph.h5"
    snr = 3

    print('\n')

    mm = mne.read_source_morph(morph_path)
    print(mm.morph_mat.toarray().shape)
    print(mm.vertices_to)
    print(mm.vertices_to[0].shape)

    print('\n')

    t0 = time.time()
    evoked = mne.read_evokeds(emeg_path, verbose=False)[0]
    t1 = time.time()
    print(f'read evoked: {t1 - t0:.3f}s')

    lambda2 = 1.0 / snr ** 2
    inverse_operator = mne.minimum_norm.read_inverse_operator(inverse_operator_path, verbose=False)
    t0 = time.time()
    print(f'read invsol: {t0 - t1:.3f}s')

    mne.set_eeg_reference(evoked, projection=True, verbose=False)
    t1 = time.time()
    print(f'set eeg ref: {t1 - t0:.3f}s')

    #stc = apply_inverse(evoked, inverse_operator, lambda2, 'MNE', pick_ori='normal', verbose=False)
    # t0 = time.time(); print(f'app. invsol: {t0 - t1:.3f}s')

    # morph_map = mne.read_source_morph(morph_path)
    get_invsol_npy(morph_path, evoked, inverse_operator_path, lambda2, 'MNE', pick_ori='normal')
    t0 = time.time()
    print(f'app. invsol: {t0 - t1:.3f}s')



    """tc = __morph_apply(morph_map, stc)
    t1 = time.time(); print(f'app. morph: {t1 - t0:.3f}s')

    print(tc.lh_data.shape, tc.rh_data.shape, tc.vertices)
    print(tc.lh_data)
    print(tc.rh_data)
    print(tc.data)"""

    

