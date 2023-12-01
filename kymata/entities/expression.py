"""
Classes and functions for storing expression information.
"""

from __future__ import annotations

from io import TextIOWrapper
from pathlib import Path
from typing import Sequence, Union, get_args, Tuple
from warnings import warn
from zipfile import ZipFile, ZIP_LZMA

from numpy import int_, float_, str_, array, array_equal, ndarray, frombuffer
from numpy.typing import NDArray
from sparse import SparseArray, COO
from xarray import DataArray, Dataset, concat
from pandas import DataFrame

from kymata.entities.sparse_data import expand_dims, minimise_pmatrix, densify_dataset
from kymata.io.file import open_or_use, file_type, path_type

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

    def save(self, to_path_or_file: path_type | file_type, compression=ZIP_LZMA, overwrite: bool = False):
        """
        Save the ExpressionSet to a specified path or already open file.

        If an open file is supplied, it should be opened in "wb" mode.

        overwrite flag is ignored if open file is supplied.
        """

        # Format versioning of the saved file. Used to (eventually) ensure old saved files can always be loaded.
        #
        # - Increment MINOR versions when creating a breaking change, but only when we commit to indefinitely supporting
        #   the new version.
        # - Increment MAJOR version when creating a breaking change which makes the format fundamentally incompatible
        #   with previous versions.
        #
        # All distinct versions should be documented.
        #
        # This value should be saved into file called /_metadata/format-version.txt within an archive.
        _VERSION = "0.1"

        warn("Experimental function. "
             "The on-disk data format for ExpressionSet is not yet fixed. "
             "Files saved using .save should not (yet) be treated as stable or future-proof.")

        if isinstance(to_path_or_file, str):
            to_path_or_file = Path(to_path_or_file)
        if isinstance(to_path_or_file, Path) and to_path_or_file.exists() and not overwrite:
            raise FileExistsError(to_path_or_file)

        with open_or_use(to_path_or_file, mode="wb") as f, ZipFile(f, "w", compression=compression) as zf:
            zf.writestr("_metadata/format-version.txt", _VERSION)
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
    def load(cls, from_path_or_file: path_type | file_type) -> ExpressionSet:
        """
        Load an ExpressionSet from an open file, or the file at the specified path.

        If an open file is supplied, it should be opened in "rb" mode.
        """

        if isinstance(from_path_or_file, str):
            from_path_or_file = Path(from_path_or_file)

        with open_or_use(from_path_or_file, mode="rb") as archive, ZipFile(archive, "r") as zf:
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
