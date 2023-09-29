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


class Expression:
    """
    Brain data associated with expression of a single function.
    Includes lh, rh, flipped, non-flipped.
    """
    def __init__(self,
                 name: str,
                 # Metadata
                 tmin: float, tstep: float,
                 vertices: Sequence[int],
                 latencies: Sequence[float],
                 # Underlying data
                 lh_data: _data_matrix_arg,
                 rh_data: _data_matrix_arg,
                 lh_flipped_data: _data_matrix_arg,
                 rh_flipped_data: _data_matrix_arg):

        self.name: str = name

        self._data_lh_sparse: dok_array = dok_array(lh_data)
        self._data_rh_sparse: dok_array = dok_array(rh_data)
        self._data_flipped_lh_sparse: dok_array = dok_array(lh_flipped_data)
        self._data_flipped_rh_sparse: dok_array = dok_array(rh_flipped_data)
        assert all_equal(data.shape for data in (rh_data, lh_data, lh_flipped_data, rh_flipped_data))

        self.vertices: Sequence[int] = vertices
        self.latencies: Sequence[float] = latencies

        self.n_vertices: int
        self.n_timepoints: int
        self.n_timepoints, self.n_vertices = lh_data.shape
        assert self.n_timepoints == len(self.latencies) == lh_data.shape[0]
        assert self.n_vertices == len(self.vertices) == lh_data.shape[1]

        self.t_min: float = tmin
        self.t_step: float = tstep
        self.t_step: float = tmin + (self.n_timepoints - 1) * tstep  # TODO: Verify I didn't make an out-by-one
        assert self.latencies[0] == self.t_min
        if len(latencies) >= 2:
            assert latencies[0] + self.t_step == latencies[1]

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


def load_from_matab_expression_files(name: str,
                                     lh_file: PathLike, rh_file: PathLike,
                                     flipped_lh_file: PathLike, flipped_rh_file: PathLike) -> Expression:
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

    return Expression(
        name=name,
        vertices=lh_mat["outputSTC"]["vertices"],
        latencies=lh_mat["latencies"],
        tmin=lh_mat["outputSTC"]["tmin"], tstep=lh_mat["outputSTC"]["tstep"],
        lh_data=        minimise_pmatrix(lh_mat["outputSTC"]["data"]),
        rh_data=        minimise_pmatrix(lh_mat["outputSTC"]["data"]),
        lh_flipped_data=minimise_pmatrix(lh_mat["outputSTC"]["data"]),
        rh_flipped_data=minimise_pmatrix(lh_mat["outputSTC"]["data"]),
    )


def _base_function_name(function_name: str) -> str:
    """Removes extraneous metadata from function names."""
    function_name = function_name.removesuffix("-flipped")
    return function_name
