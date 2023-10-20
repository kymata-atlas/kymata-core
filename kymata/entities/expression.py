"""
Classes and functions for storing expression information.
"""

from __future__ import annotations

from io import TextIOWrapper
from os import PathLike
from pathlib import Path
from typing import Sequence, Union, get_args, Tuple
from warnings import warn
from zipfile import ZipFile

from numpy import nan_to_num, minimum, int_, float_, str_, array, array_equal, ndarray, frombuffer
from numpy.typing import NDArray
from sparse import SparseArray, COO
from xarray import DataArray, Dataset, concat
from pandas import DataFrame

from kymata.entities.iterables import all_equal
from kymata.entities.sparse_data import expand_dims, minimise_pmatrix, densify_dataset
from kymata.io.matlab import load_mat

Hexel = int  # Todo: change this and others to `type Hexel = int` on dropping support for python <3.12
Latency = float

_InputDataArray = Union[ndarray, SparseArray]  # Type alias for data which can be accepted  # TODO: replace with nicer | syntax when dropping supprot for python <3.12

# Data dimension labels
_HEXEL = "hexel"
_LATENCY = "latency"
_FUNCTION = "function"
# Hemisphere
_LEFT = "left"
_RIGHT = "right"

# Set consistent dtypes
_HexelType = int_
_LatencyType = float_
_FunctionNameType = str_


class ExpressionSet:
    """
    Brain data associated with expression of a single function.
    Includes lh, rh, flipped, non-flipped.
    """

    # Canonical order of dimensions
    _dims = (_HEXEL, _LATENCY, _FUNCTION)

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
        _length_mismatch_message = ("Argument length mismatch, please supply one function name and accompanying data, "
                                    "or equal-length sequences of the same.")
        if isinstance(functions, str):
            assert isinstance(data_lh, get_args(_InputDataArray)), _length_mismatch_message
            assert isinstance(data_rh, get_args(_InputDataArray)), _length_mismatch_message
            # Wrap into sequence
            functions = [functions]
            data_lh = [data_lh]
            data_rh = [data_rh]
        assert len(functions) == len(data_lh) == len(data_rh), _length_mismatch_message
        assert len(functions) == len(set(functions)), "Duplicated functions in input"

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

    @classmethod
    def _init_prep_data(cls, data: _InputDataArray) -> COO:
        """Prep data for ExpressionSet.__init__"""
        if isinstance(data, ndarray):
            data = minimise_pmatrix(data)
        elif not isinstance(data, SparseArray):
            raise NotImplementedError()
        data = expand_dims(data, 2)
        return data

    @property
    def functions(self) -> list[_FunctionNameType]:
        """Function names."""
        return self._data.coords[_FUNCTION].values.tolist()

    @property
    def hexels(self) -> NDArray[_HexelType]:
        """Hexels, canonical ID."""
        return self._data.coords[_HEXEL].values

    @property
    def latencies(self) -> NDArray[_LatencyType]:
        """Latencies, in seconds."""
        return self._data.coords[_LATENCY].values

    @property
    def left(self) -> DataArray:
        """Left-hemisphere data."""
        return self._data[_LEFT]

    @property
    def right(self) -> DataArray:
        """Right-hemisphere data."""
        return self._data[_RIGHT]

    def __getitem__(self, functions: str | Sequence[str]) -> ExpressionSet:
        """
        Select data for specified function(s) only.
        Use a function name or list/array of function names
        """
        # Allow indexing by a single function
        if isinstance(functions, str):
            functions = [functions]
        for f in functions:
            if f not in self.functions:
                raise KeyError(f)
        return ExpressionSet(
            functions=functions,
            hexels=self.hexels,
            latencies=self.latencies,
            data_lh=[self._data[_LEFT].sel({_FUNCTION: function}).data for function in functions],
            data_rh=[self._data[_RIGHT].sel({_FUNCTION: function}).data for function in functions],
        )

    def __copy__(self):
        return ExpressionSet(
            functions=self.functions.copy(),
            hexels=self.hexels.copy(),
            latencies=self.latencies.copy(),
            data_lh=self._data[_LEFT].values.copy(),
            data_rh=self._data[_RIGHT].values.copy(),
        )

    def __add__(self, other: ExpressionSet) -> ExpressionSet:
        assert array_equal(self.hexels, other.hexels), "Hexels mismatch"
        assert array_equal(self.latencies, other.latencies), "Latencies mismatch"
        # constructor expects a sequence of function names and sequences of 2d matrices
        functions = []
        data_lh = []
        data_rh = []
        for expr_set in [self, other]:
            for i, function in enumerate(expr_set.functions):
                functions.append(function)
                data_lh.append(expr_set._data[_LEFT].data[:, :, i])
                data_rh.append(expr_set._data[_RIGHT].data[:, :, i])
        return ExpressionSet(
            functions=functions,
            hexels=self.hexels, latencies=self.latencies,
            data_lh=data_lh, data_rh=data_rh,
        )

    def __eq__(self, other: ExpressionSet):
        if not self.functions == other.functions:
            return False
        if not array_equal(self.hexels, other.hexels):
            return False
        if not array_equal(self.latencies, other.latencies):
            return False
        if not COO(self.left.data == other.left.data).all():
            return False
        if not COO(self.right.data == other.right.data).all():
            return False
        return True

    def save(self, to_path: Path | str, overwrite: bool = False):
        warn("Experimental function. "
             "The on-disk data format for ExpressionSet is not yet fixed. "
             "Files saved using .save should not (yet) be treated as stable or future-proof.")
        to_path = Path(to_path)
        if not overwrite and to_path.exists():
            raise FileExistsError(to_path)

        with ZipFile(to_path, "w") as zf:
            zf.writestr("/hexels.txt", "\n".join(str(x) for x in self.hexels))
            zf.writestr("/latencies.txt", "\n".join(str(x) for x in self.latencies))
            zf.writestr("/functions.txt", "\n".join(str(x) for x in self.functions))
            zf.writestr("/left/coo-coords.bytes", self.left.data.coords.tobytes(order="C"))
            zf.writestr("/left/coo-data.bytes", self.left.data.data.tobytes(order="C"))
            # The shape can be inferred, but we save it as an extra validation
            zf.writestr("/left/coo-shape.txt", "\n".join(str(x) for x in self.left.data.shape))
            zf.writestr("/right/coo-coords.bytes", self.right.data.coords.tobytes(order="C"))
            zf.writestr("/right/coo-data.bytes", self.right.data.data.tobytes(order="C"))
            zf.writestr("/right/coo-shape.txt", "\n".join(str(x) for x in self.right.data.shape))

    @classmethod
    def load(cls, from_path: PathLike) -> ExpressionSet:
        from_path = Path(from_path)
        with ZipFile(from_path, "r") as zf:
            with TextIOWrapper(zf.open("/hexels.txt"), encoding="utf-8") as f:
                hexels: list[_HexelType] = [_HexelType(h.strip()) for h in f.readlines()]
            with TextIOWrapper(zf.open("/latencies.txt"), encoding="utf-8") as f:
                latencies: list[_LatencyType] = [_LatencyType(lat.strip()) for lat in f.readlines()]
            with TextIOWrapper(zf.open("/functions.txt"), encoding="utf-8") as f:
                functions: list[_FunctionNameType] = [_FunctionNameType(fun.strip()) for fun in f.readlines()]
            with zf.open("/left/coo-coords.bytes") as f:
                left_coords: ndarray = frombuffer(f.read(), dtype=int).reshape((3, -1))
            with zf.open("/left/coo-data.bytes") as f:
                left_data: ndarray = frombuffer(f.read(), dtype=float)
            with TextIOWrapper(zf.open("/left/coo-shape.txt"), encoding="utf-8") as f:
                left_shape: tuple[int, ...] = tuple(int(s.strip()) for s in f.readlines())
            with zf.open("/right/coo-coords.bytes") as f:
                right_coords: ndarray = frombuffer(f.read(), dtype=int).reshape((3, -1))
            with zf.open("/right/coo-data.bytes") as f:
                right_data: ndarray = frombuffer(f.read(), dtype=float)
            with TextIOWrapper(zf.open("/right/coo-shape.txt"), encoding="utf-8") as f:
                right_shape: tuple[int, ...] = tuple(int(s.strip()) for s in f.readlines())

        left_sparse = COO(coords=left_coords, data=left_data, shape=left_shape, prune=True, fill_value=1.0)
        right_sparse = COO(coords=right_coords, data=right_data, shape=right_shape, prune=True, fill_value=1.0)

        assert left_shape == right_shape

        # In case there was only 1 function and we have a 2-d data matrix
        if len(left_shape) == 2:
            # TODO: does this ever actually happen?
            left_sparse = expand_dims(left_sparse)
            right_sparse = expand_dims(right_sparse)

        assert left_shape == (len(hexels), len(latencies), len(functions))

        return ExpressionSet(
            functions=functions,
            hexels=hexels,
            latencies=latencies,
            data_lh=[left_sparse[:, :, i] for i in range(len(functions))],
            data_rh=[right_sparse[:, :, i] for i in range(len(functions))],
        )

    def best_functions(self) -> Tuple[DataFrame, DataFrame]:
        """
        Return a pair of DataFrames (left, right), containing:
        for each hexel, the best function and latency for that hexel, and the associated p-value
        """
        # Want, for each hexel:
        #  - The name, f, of the function which is best at any latency
        #  - The latency, l, for which f is best
        #  - The p-value, p, for f at l

        # sparse.COO doesn't implement argmin, so we have to do it in a few steps

        data = self._data.copy()
        densify_dataset(data)

        best_latency = data.idxmin(dim=_LATENCY)    # (hexel, function) → l, the best latency
        p_at_best_latency = data.min(dim=_LATENCY)  # (hexel, function) → p of best latency for each function

        p_at_best_function = p_at_best_latency.min(dim=_FUNCTION)  # (hexel) → p of best function (at best latency)
        best_function = p_at_best_latency.idxmin(dim=_FUNCTION)    # (hexel) → f, the best function

        # TODO: shame I have to break into the Left/Right structure here,
        #  but I can't think of a better way to do it
        p_vals_lh = p_at_best_function[_LEFT].data
        p_vals_rh = p_at_best_function[_RIGHT].data

        best_functions_lh = best_function[_LEFT].data
        best_functions_rh = best_function[_RIGHT].data

        best_latencies_lh = best_latency[_LEFT].sel({_HEXEL: self.hexels, _FUNCTION: best_function[_LEFT]}).data
        best_latencies_rh = best_latency[_RIGHT].sel({_HEXEL: self.hexels, _FUNCTION: best_function[_RIGHT]}).data

        # Cut out hexels which have a best p-val of 1
        idxs_lh = p_vals_lh < 1
        idxs_rh = p_vals_rh < 1

        return (
            DataFrame.from_dict({
                _HEXEL: self.hexels[idxs_lh],
                _FUNCTION: best_functions_lh[idxs_lh],
                _LATENCY: best_latencies_lh[idxs_lh],
                "value": p_vals_lh[idxs_lh],
            }),
            DataFrame.from_dict({
                _HEXEL: self.hexels[idxs_rh],
                _FUNCTION: best_functions_rh[idxs_rh],
                _LATENCY: best_latencies_rh[idxs_rh],
                "value": p_vals_rh[idxs_rh],
            })
        )

    # TODO: plotting in here?


def load_matab_expression_files(function_name: str,
                                lh_file: Path | str, flipped_lh_file: Path | str,
                                rh_file: Path | str, flipped_rh_file: Path | str) -> ExpressionSet:
    """Load from a set of MATLAB files."""

    for p in [lh_file, rh_file, flipped_lh_file, flipped_rh_file]:
        if not Path(p).exists():
            raise FileNotFoundError(p)

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
