"""
Classes and functions for storing expression information.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Sequence, Union, get_args

from numpy import nan_to_num, minimum, int_, float_, str_, array
from numpy.typing import NDArray
import sparse
from xarray import DataArray, Dataset, concat
from pandas import DataFrame

from kymata.entities.iterables import all_equal
from kymata.entities.sparse import expand_dims, minimise_pmatrix, densify_dataset
from kymata.io.matlab import load_mat

Hexel = int  # Todo: change this and others to `type Hexel = int` on dropping support for python <3.12
Latency = float

_InputDataArray = Union[NDArray, sparse.SparseArray]  # Type alias for data which can be accepted  # TODO: replace with nicer | syntax when dropping supprot for python <3.12

# Data dimension labels
_HEXEL = "hexel"
_LATENCY = "latency"
_FUNCTION = "function"
# Hemisphere
_LEFT = "left"
_RIGHT = "right"

_HexelType = int_
_LatencyType = float_
_FunctionNameType = str_

class ExpressionSet:
    """
    Brain data associated with expression of a single function.
    Includes lh, rh, flipped, non-flipped.
    """

    def __init__(self,
                 functions: str | Sequence[str],
                 # Metadata
                 hexels: Sequence[Hexel],
                 latencies: Sequence[Latency],
                 # In general, we will combine flipped and non-flipped versions
                 data_lh: _InputDataArray | Sequence[_InputDataArray],
                 data_rh: _InputDataArray | Sequence[_InputDataArray]):
        # TODO: Docstring

        # Validate arguments
        if isinstance(functions, str):
            assert isinstance(data_lh, get_args(_InputDataArray)), "Argument length mismatch, please supply one function name and accomanying data, or equal-lenth sequences of the same."
            assert isinstance(data_rh, get_args(_InputDataArray)), "Argument length mismatch, please supply one function name and accomanying data, or equal-lenth sequences of the same."
            # Wrap into sequence
            functions = [functions]
            data_lh = [data_lh]
            data_rh = [data_rh]
        assert len(functions) == len(data_lh) == len(data_rh), "Argument length mismatch, please supply one function name and accomanying data, or equal-lenth sequences of the same."

        hexels = array(hexels, dtype=_HexelType)
        latencies = array(latencies, dtype=_LatencyType)
        functions = array(functions, dtype=_FunctionNameType)

        datasets = []
        for f, dl, dr in zip(functions, data_lh, data_rh):
            # Check validity of input data dimensions
            dl = self._init_prep_data(dl)
            dr = self._init_prep_data(dr)
            assert dl.shape == dr.shape
            assert len(hexels) == dl.shape[0], f"Hexels mismatch for {f}"
            assert len(latencies) == dl.shape[1], f"Latencies mismatch for {f}"
            datasets.append(
                Dataset({
                    _LEFT: DataArray(
                        data=dl,
                        dims=ExpressionSet._dims,
                    ),
                    _RIGHT: DataArray(
                        data=dr,
                        dims=ExpressionSet._dims,
                    )},
                    coords={_HEXEL: hexels, _LATENCY: latencies, _FUNCTION: [f]},
                )
            )
        self._data: Dataset = concat(datasets, dim=_FUNCTION)

        # TODO: verify coords and dims exist at dataset level

    @classmethod
    def _init_prep_data(cls, data: _InputDataArray) -> sparse.COO:
        """Prep data for ExpressionSet.__init__"""
        data = minimise_pmatrix(data, axis=1)
        data = expand_dims(data, 2)
        return data

    @property
    def functions(self) -> NDArray[_FunctionNameType]:
        """Function names."""
        return self._data.coords[_FUNCTION].values

    @property
    def hexels(self) -> NDArray[_HexelType]:
        """Hexels, canonical ID."""
        return self._data.coords[_HEXEL].values

    @property
    def latencies(self) -> NDArray[_LatencyType]:
        """Latencies, in seconds."""
        return self._data.coords[_LATENCY].values

    def __contains__(self, function_name: str):
        """
        Whether a named function is in the ExpressionSet.
        """
        return function_name in self._data.coords[_FUNCTION]

    def __getitem__(self, item):
        """
        Select data for specified function(s) only.
        Use a function name or list/array of function names
        """
        return ExpressionSet(
            functions=item,
            hexels=self.hexels,
            latencies=self.latencies,
            data_lh=self._data[_LEFT].sel({_FUNCTION: item}),
            data_rh=self._data[_RIGHT].sel({_FUNCTION: item})
        )

    def __copy__(self):
        return ExpressionSet(
            functions=self.functions.copy(),
            hexels=self.hexels.copy(), latencies=self.latencies.copy(),
            data_lh=self._data[_LEFT].values.copy(),
            data_rh=self._data[_RIGHT].valuies.copy()
        )

    def __add__(self, other: ExpressionSet) -> ExpressionSet:
        assert self.hexels == other.hexels, "Hexels mismatch"
        assert self.latencies == other.latencies, "Latencies mismatch"
        data = concat((self._data, other._data), dim=_FUNCTION)
        return ExpressionSet(
            functions=self.functions + other.functions,
            hexels=self.hexels, latencies=self.latencies,
            data_lh=data[_LEFT], data_rh=data[_RIGHT]
        )

    # TODO: plotting in here

    # Public data access separated from data storage

    def best_function(self) -> DataFrame:
        """"""
        # sparse.COO doesn't implement argmin, so we have to do it in two steps:
        # Get the best latency
        p_at_best_latency = self._data.min(dim=_LATENCY)
        # Then find the best function on the densified version (to get argmin support)
        best_function = densify_dataset(p_at_best_latency).idxmin(dim=_FUNCTION)
        return best_function.to_dataframe()

    # Canonical order of dimensions
    _dims = (_HEXEL, _LATENCY, _FUNCTION)


def load_matab_expression_files(function_name: str,
                                lh_file: PathLike, flipped_lh_file: PathLike,
                                rh_file: PathLike, flipped_rh_file: PathLike) -> ExpressionSet:
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

    def _prep_matlab_data(data):
        return (
            # Some hexels are all nans because they're on the medial wall
            # or otherwise intentionally excluded from the analysis;
            # we replace those with p=1.0 to ignore
            # TODO: could also delete
            nan_to_num(
                nan=1.0,
                x=_downsample_data(data, downsample_ratio)
                # Trim excess
                [:len(lh_mat["latencies"]), :],
            )
        )

    # Combine flipped and non-flipped
    # TODO: verify theoretically that this is ok
    pmatrix_lh = minimum(_prep_matlab_data(lh_mat["outputSTC"]["data"]), _prep_matlab_data(flipped_lh_mat["outputSTC"]["data"]))
    pmatrix_rh = minimum(_prep_matlab_data(rh_mat["outputSTC"]["data"]), _prep_matlab_data(flipped_rh_mat["outputSTC"]["data"]))

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


if __name__ == '__main__':
    sample_data_dir = Path(Path(__file__).parent.parent.parent, "data", "sample-data")
    expression_data = load_matab_expression_files(
        function_name="hornschunck_horizontalPosition",
        lh_file=Path(sample_data_dir, "hornschunck_horizontalPosition_lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
        rh_file=Path(sample_data_dir, "hornschunck_horizontalPosition_rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
        flipped_lh_file=Path(sample_data_dir, "hornschunck_horizontalPosition-flipped_lh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
        flipped_rh_file=Path(sample_data_dir, "hornschunck_horizontalPosition-flipped_rh_10242verts_-200-800ms_cuttoff1000_5perms_ttestpval.mat"),
    )
    print(expression_data.best_function())
    print("hornschunck_horizontalPosition" in expression_data)
