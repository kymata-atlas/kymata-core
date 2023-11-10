from __future__ import annotations

from pathlib import Path
from typing import Optional

from numpy import nan_to_num, minimum
from scipy.io import loadmat as loadmat_pre_73
from mat73 import loadmat as loadmat_post_73

from kymata.entities.expression import ExpressionSet
from kymata.entities.iterables import all_equal


def load_mat(path):
    """Loads all variables in a matlab file, regardless of version."""
    try:
        with path.open("rb") as f:
            mat = loadmat_pre_73(f, appendmat=False)
    except NotImplementedError:
        mat = loadmat_post_73(path)
    return mat


def load_mat_variable(path, variable_name: str):
    """Loads a specified named variable from a matlab file"""
    return load_mat(path)[variable_name]


def load_matab_expression_files(
        function_name: str,
        lh_file: Path | str,
        rh_file: Path | str,
        flipped_lh_file: Optional[Path | str] = None,
        flipped_rh_file: Optional[Path | str] = None,
) -> ExpressionSet:
    """Load from a set of MATLAB files."""

    if flipped_lh_file is None or flipped_rh_file is None:
        assert flipped_lh_file is None and flipped_rh_file is None, "Please supply 2 or 4 files."
        return _load_matab_expression_files_combined_flipped(
            function_name=function_name, lh_file=lh_file, rh_file=rh_file
        )
    else:
        return _load_matab_expression_files_separate_flipped(
            function_name=function_name, lh_file=lh_file, rh_file=rh_file,
            flipped_lh_file=flipped_lh_file, flipped_rh_file=flipped_rh_file
        )


def _load_matlab_validate(all_mats: tuple[dict, ...]) -> None:
    """
    Returns silently unless there are validation errors
    """
    assert len(all_mats) in {2, 4}

    # All the same function
    assert all_equal([_base_function_name(mat["functionname"]) for mat in all_mats])
    # Timing information is the same
    assert all_equal([mat["latency_step"]               for mat in all_mats])
    assert all_equal([len(mat["latencies"])             for mat in all_mats])
    assert all_equal([mat["nTimePoints"]                for mat in all_mats])
    assert all_equal([mat["outputSTC"]["tmin"]          for mat in all_mats])
    assert all_equal([mat["outputSTC"]["tstep"]         for mat in all_mats])
    assert all_equal([mat["outputSTC"]["data"].shape[0] for mat in all_mats])

    assert all_mats[0]["outputSTC"]["data"].shape[0] == all_mats[0]["nTimePoints"]
    # Spatial information is the same
    assert all_equal([mat["nVertices"]                       for mat in all_mats])
    assert all_equal([len(mat["outputSTC"]["vertices"])      for mat in all_mats])
    assert all_equal([mat["outputSTC"]["data"].shape[1]      for mat in all_mats])
    assert all_mats[0]["outputSTC"]["data"].shape[1] == all_mats[0]["nVertices"]


def _load_matlab_downsample_ratio(lh_mat: dict) -> int:
    # If the data has been downsampled
    if lh_mat["outputSTC"]["tstep"] != lh_mat["latency_step"]/1000:
        downsample_ratio = (lh_mat["latency_step"] / 1000) / lh_mat["outputSTC"]["tstep"]
        assert downsample_ratio == int(downsample_ratio)
        downsample_ratio = int(downsample_ratio)
    else:
        downsample_ratio = 1
    return downsample_ratio


def _prep_matlab_data(data, n_latencies, downsample_ratio):
    return (
        # Some hexels are all nans because they're on the medial wall
        # or otherwise intentionally excluded from the analysis;
        # we replace those with p=1.0 to ignore
        # TODO: could also delete
        nan_to_num(
            nan=1.0,
            x=_downsample_data(data, downsample_ratio)
            # Trim excess
            [:n_latencies, :],
        )
    )


def _load_matab_expression_files_separate_flipped(
        function_name: str,
        lh_file: Path | str, flipped_lh_file: Path | str,
        rh_file: Path | str, flipped_rh_file: Path | str) -> ExpressionSet:
    """
    For loading Matlab files where the flipped and non-flipped versions are separate
    (expects 4 files).
    """
    for p in [lh_file, rh_file, flipped_lh_file, flipped_rh_file]:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    lh_mat = load_mat(Path(lh_file))
    rh_mat = load_mat(Path(rh_file))
    flipped_lh_mat = load_mat(Path(flipped_lh_file))
    flipped_rh_mat = load_mat(Path(flipped_rh_file))

    # Check 4 files are compatible
    all_mats = (lh_mat, rh_mat, flipped_lh_mat, flipped_rh_mat)

    # Chirality is correct
    assert lh_mat["leftright"] == flipped_lh_mat["leftright"] == "lh"
    assert rh_mat["leftright"] == flipped_rh_mat["leftright"] == "rh"

    _load_matlab_validate(all_mats)

    downsample_ratio = _load_matlab_downsample_ratio(lh_mat)

    # Combine flipped and non-flipped
    pmatrix_lh = minimum(
        _prep_matlab_data(lh_mat["outputSTC"]["data"], n_latencies=len(lh_mat["latencies"]), downsample_ratio=downsample_ratio),
        _prep_matlab_data(flipped_lh_mat["outputSTC"]["data"], n_latencies=len(lh_mat["latencies"]), downsample_ratio=downsample_ratio)
    )
    pmatrix_rh = minimum(
        _prep_matlab_data(rh_mat["outputSTC"]["data"], n_latencies=len(lh_mat["latencies"]), downsample_ratio=downsample_ratio),
        _prep_matlab_data(flipped_rh_mat["outputSTC"]["data"], n_latencies=len(lh_mat["latencies"]), downsample_ratio=downsample_ratio)
    )

    return ExpressionSet(
        functions=function_name,
        hexels=lh_mat["outputSTC"]["vertices"],
        latencies=lh_mat["latencies"] / 1000,
        data_lh=pmatrix_lh.T, data_rh=pmatrix_rh.T,
    )


def _load_matab_expression_files_combined_flipped(
        function_name: str,
        lh_file: Path | str, rh_file: Path | str) -> ExpressionSet:
    """
    For loading Matlab files where the flipped and non-flipped versions are already combined
    (expects 2 files).
    """

    for p in [lh_file, rh_file]:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    lh_mat = load_mat(Path(lh_file))
    rh_mat = load_mat(Path(rh_file))

    # Check 2 files are compatible
    all_mats = (lh_mat, rh_mat)

    # Chirality is correct
    assert lh_mat["leftright"] == "lh"
    assert rh_mat["leftright"] == "rh"

    _load_matlab_validate(all_mats)

    downsample_ratio = _load_matlab_downsample_ratio(lh_mat)

    pmatrix_lh = _prep_matlab_data(lh_mat["outputSTC"]["data"], n_latencies=len(lh_mat["latencies"]), downsample_ratio=downsample_ratio)
    pmatrix_rh = _prep_matlab_data(rh_mat["outputSTC"]["data"], n_latencies=len(lh_mat["latencies"]), downsample_ratio=downsample_ratio)

    return ExpressionSet(
        functions=function_name,
        hexels=lh_mat["outputSTC"]["vertices"],
        latencies=lh_mat["latencies"] / 1000,
        data_lh=pmatrix_lh.T, data_rh=pmatrix_rh.T,
    )


def _base_function_name(function_name: str) -> str:
    """
    Removes extraneous metadata from function names.
    """
    function_name = function_name.removesuffix("-flipped")
    return function_name


def _downsample_data(data, ratio):
    """
    Subsample a numpy array in the first dimension.
    """
    if ratio == 1:
        return data
    else:
        return data[::ratio, :]
