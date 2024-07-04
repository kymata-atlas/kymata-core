import mne
from mne.evoked import Evoked
from mne.minimum_norm.inverse import (
    INVERSE_METHODS, _check_ch_names, _check_or_prepare, _assemble_kernel, _check_ori,
    _check_reference
)
from mne.utils import (
    _check_option,
    _validate_type,
)


def pick_channels_inverse_operator(ch_names, inv):
    """Exposes `mne.minimum_norm.inverse._pick_channels_inverse_operator(ch_names, inv)`."""
    from mne.minimum_norm.inverse import _pick_channels_inverse_operator
    return _pick_channels_inverse_operator(ch_names, inv)


def premorph_inverse_operator(
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
    mne.set_eeg_reference(evoked, projection=True, verbose=False)
    inverse_operator = mne.minimum_norm.read_inverse_operator(inverse_operator_path, verbose=False)
    _validate_type(evoked, Evoked, "evoked")
    _check_reference(evoked, inverse_operator["info"]["ch_names"])
    _check_option("method", method, INVERSE_METHODS)
    _check_ori(pick_ori, inverse_operator["source_ori"], inverse_operator["src"])
    #
    #   Set up the inverse according to the parameters
    #
    nave = evoked.nave

    _check_ch_names(inverse_operator, evoked.info)

    inv = _check_or_prepare(
        inverse_operator, nave, lambda2, method, method_params, prepared, copy="non-src"
    )

    K, noise_norm, vertno, source_nn = _assemble_kernel(
        inv, label, method, pick_ori, use_cps=use_cps
    )

    morph = mne.read_source_morph(morph_path)
    morph_csr = morph.morph_mat

    premorphed_inverse_operator = morph_csr.dot(K)

    return inv, premorphed_inverse_operator, morph.vertices_to
