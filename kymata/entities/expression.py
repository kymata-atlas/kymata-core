"""
Classes and functions for storing expression information.
"""

from os import PathLike
from pathlib import Path
from typing import Sequence

from numpy import ndarray
from scipy.sparse import sparray, dok_array

from kymata.entities.iterables import all_equal
from kymata.io.matlab import load_mat

_data_matrix_arg = sparray | ndarray


#  TODO: this should be able to contain arbitrary quantities of functions
#  TODO: and have merge/split/slice functionality
class ExpressionSet:
    """
    Brain data associated with expression of a single function.
    Includes lh, rh, flipped, non-flipped.
    """
    def __init__(self,
                 function_name: str,
                 # Metadata
                 hexels: Sequence[int],
                 latencies: Sequence[float],  # Seconds
                 # Underlying data
                 lh_data: _data_matrix_arg,
                 rh_data: _data_matrix_arg,
                 lh_flipped_data: _data_matrix_arg,
                 rh_flipped_data: _data_matrix_arg):

        self.function_name: str = function_name

        self._data_lh_sparse: dok_array = dok_array(lh_data)
        self._data_rh_sparse: dok_array = dok_array(rh_data)
        self._data_flipped_lh_sparse: dok_array = dok_array(lh_flipped_data)
        self._data_flipped_rh_sparse: dok_array = dok_array(rh_flipped_data)
        assert all_equal(data.shape for data in (rh_data, lh_data, lh_flipped_data, rh_flipped_data))

        self.hexels: Sequence[int] = hexels
        self.latencies: Sequence[float] = latencies

        self.n_vertices: int
        self.n_timepoints: int
        self.n_timepoints, self.n_vertices = lh_data.shape
        assert self.n_timepoints == len(self.latencies) == lh_data.shape[0]
        assert self.n_vertices == len(self.hexels) == lh_data.shape[1]

    # Public data access separated from data storage

    @property
    def data_lh(self) -> sparray:
        return self._data_lh_sparse

    @property
    def data_rh(self) -> sparray:
        return self._data_rh_sparse

    @property
    def data_flipped_lh(self) -> sparray:
        return self._data_flipped_lh_sparse

    @property
    def data_flipped_rh(self) -> sparray:
        return self._data_flipped_rh_sparse


def minimise_pmatrix(pmatrix: _data_matrix_arg) -> dok_array:
    """
    Converts a data matrix containing p-values into a sparse matrix
    only storing the minimum value for each row.
    """
    # TODO
    raise NotImplementedError()


# TODO: don't in general need to store flipped, so this should be a specialised load_flipped function
def load_from_matab_expression_files(function_name: str,
                                     lh_file: PathLike, rh_file: PathLike,
                                     flipped_lh_file: PathLike, flipped_rh_file: PathLike) -> ExpressionSet:
    """Load from a set of MATLAB files."""
    lh_mat = load_mat(Path(lh_file))
    rh_mat = load_mat(Path(rh_file))
    flipped_lh_mat = load_mat(Path(flipped_lh_file))
    flipped_rh_mat = load_mat(Path(flipped_rh_file))

    # Check 4 files are compatible
    all_mats = (lh_mat, rh_mat, flipped_lh_mat, flipped_rh_mat)
    # All the same function
    assert all_equal(_base_function_name(mat["functionname"]) for mat in all_mats)
    # Chirality is correct
    assert lh_mat["leftright"] == flipped_lh_mat["leftright"] == "lh"
    assert rh_mat["leftright"] == flipped_rh_mat["leftright"] == "rh"
    # Timing information is the same
    assert all_equal(mat["latency_step"]               for mat in all_mats)
    assert all_equal(len(mat["latencies"])             for mat in all_mats)
    assert all_equal(mat["nTimePoints"]                for mat in all_mats)
    assert all_equal(mat["outputSTC"]["tmin"]          for mat in all_mats)
    assert all_equal(mat["outputSTC"]["tstep"]         for mat in all_mats)
    assert all_equal(mat["outputSTC"]["data"].shape[0] for mat in all_mats)
    assert lh_mat["outputSTC"]["data"].shape[0] == lh_mat["nTimePoints"]
    # Spatial information is the same
    assert all_equal(mat["nVertices"]                       for mat in all_mats)
    assert all_equal(len(mat["outputSTC"]["vertices"])      for mat in all_mats)
    assert all_equal(mat["outputSTC"]["data"].shape[1]      for mat in all_mats)
    assert lh_mat["outputSTC"]["data"].shape[1] == lh_mat["nVertices"]

    # If the data has been downsampled
    if lh_mat["outputSTC"]["tstep"] != lh_mat["latency_step"]/1000:
        downsample_ratio = (lh_mat["latency_step"] / 1000) / lh_mat["outputSTC"]["tstep"]
        assert downsample_ratio == int(downsample_ratio)
        downsample_ratio = int(downsample_ratio)
    else:
        downsample_ratio = 1

    def _prep_data(data):
        return (
            minimise_pmatrix(
                _downsample_data(data, downsample_ratio)
                [:len(lh_mat["latencies"]), :]
        ))

    return ExpressionSet(
        function_name=function_name,
        hexels=lh_mat["outputSTC"]["vertices"],
        latencies=lh_mat["latencies"] / 1000,
        lh_data=        _prep_data(lh_mat["outputSTC"]["data"]),
        rh_data=        _prep_data(lh_mat["outputSTC"]["data"]),
        lh_flipped_data=_prep_data(lh_mat["outputSTC"]["data"]),
        rh_flipped_data=_prep_data(lh_mat["outputSTC"]["data"]),
    )


def _base_function_name(function_name: str) -> str:
    """Removes extraneous metadata from function names."""
    function_name = function_name.removesuffix("-flipped")
    return function_name


def _downsample_data(data, ratio):
    if ratio == 1:
        return data
    else:
        return data[::ratio, :]


if __name__ == '__main__':
    sample_data_dir = Path(Path(__file__).parent.parent.parent, "data", "sample-data")
    expression_data = load_from_matab_expression_files(
        function_name="hornschunck_horizontalPosition",
        lh_file=Path(sample_data_dir, "hornschunck_horizontalPosition_lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
        rh_file=Path(sample_data_dir, "hornschunck_horizontalPosition_rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
        flipped_lh_file=Path(sample_data_dir, "hornschunck_horizontalPosition-flipped_lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
        flipped_rh_file=Path(sample_data_dir, "hornschunck_horizontalPosition-flipped_rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    )
