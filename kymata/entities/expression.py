"""
Classes and functions for storing expression information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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

from kymata.entities.iterables import all_equal
from kymata.entities.sparse_data import expand_dims, minimise_pmatrix, densify_dataset
from kymata.io.file import open_or_use, file_type, path_type

Hexel = int  # Todo: change this and others to `type Hexel = int` on dropping support for python <3.12
Sensor = str
Latency = float

_InputDataArray = Union[ndarray, SparseArray]  # Type alias for data which can be accepted  # TODO: replace with nicer | syntax when dropping supprot for python <3.12

# Data dimension labels
_HEXEL = "hexel"
_SENSOR = "sensor"
_LATENCY = "latency"
_FUNCTION = "function"

# Layer (e.g. hemisphere)
_LEFT = "left"
_RIGHT = "right"
_SCALP = "scalp"

# Set consistent dtypes
_HexelDType = int_
_SensorDType = str_
_LatencyDType = float_
_FunctionNameDType = str_


class ExpressionSet(ABC):
    """
    Brain data associated with expression of a single function.
    """

    def __init__(self,
                 functions: str | Sequence[str],
                 # Metadata
                 latencies: Sequence[Latency],
                 # In general, we will combine flipped and non-flipped versions
                 # data_arrays contains a dict mapping layer names to data arrays
                 # e.g.:       "left" → [array(), array(), array()],
                 #        and "right" → [array(), array(), array()]
                 #   or:      "scalp" → [array(), array(), array()]
                 # All should be the same size
                 data_layers: dict[str, _InputDataArray | Sequence[_InputDataArray]],
                 # Supply channels already as an array, i.e.
                 channel_coord_name: str,
                 channel_coord_dtype,
                 channel_coord_values: Sequence,
                 # String to identify sub-classes in saved files
                 file_format_identifier: str,
                 ):

        self._layers: list[str] = list(data_layers.keys())
        self._file_format_identifier: str = file_format_identifier

        self._channel_coord_name = channel_coord_name
        self._dims = (channel_coord_name, _LATENCY, _FUNCTION)  # Canonical order of dimensions

        # Validate arguments
        _length_mismatch_message = ("Argument length mismatch, please supply one function name and accompanying data, "
                                    "or equal-length sequences of the same.")
        # TODO: test all of these validations
        if isinstance(functions, str):
            # If only one function
            for layer, data in data_layers.items():
                # Data not a sequence
                assert isinstance(data, get_args(_InputDataArray)), _length_mismatch_message
            # Wrap into sequences
            functions = [functions]
            for layer in data_layers.keys():
                data_layers[layer] = [data_layers[layer]]

        assert len(functions) == len(set(functions)), "Duplicated functions in input"
        for layer, data in data_layers:
            assert len(functions) == len(data), _length_mismatch_message
        assert all_equal([arr.shape for _layer, arrs in data_layers.items() for arr in arrs])

        channels  = array(channel_coord_values, dtype=channel_coord_dtype)
        latencies = array(latencies, dtype=_LatencyDType)
        functions = array(functions, dtype=_FunctionNameDType)

        datasets = []
        for i, f in enumerate(functions):
            dataset_dict = dict()
            for layer, data in data_layers:
                # Get this function's data
                data = data[i]
                data = self._init_prep_data(data)
                # Check validity of input data dimensions
                assert len(channels) == data.shape[0], f"{channel_coord_name} mismatch for {f}"
                assert len(latencies) == data.shape[1], f"Latencies mismatch for {f}"
                dataset_dict[layer] = DataArray(
                    data=data,
                    dims=self._dims,
                )
            datasets.append(
                Dataset(dataset_dict,
                        coords={channel_coord_name: channels, _LATENCY: latencies, _FUNCTION: [f]})
            )
        self._data = concat(datasets, dim=_FUNCTION)

    @classmethod
    def _init_prep_data(cls, data: _InputDataArray) -> COO:
        if isinstance(data, ndarray):
            data = minimise_pmatrix(data)
        elif not isinstance(data, SparseArray):
            raise NotImplementedError()
        data = expand_dims(data, 2)
        return data

    @property
    def _channels(self) -> NDArray:
        return self._data.coords[self._channel_coord_name].values

    @property
    def functions(self) -> list[_FunctionNameDType]:
        """Function names."""
        return self._data.coords[_FUNCTION].values.tolist()

    @property
    def latencies(self) -> NDArray[_LatencyDType]:
        """Latencies, in seconds."""
        return self._data.coords[_LATENCY].values

    @abstractmethod
    def __getitem__(self, functions: str | Sequence[str]) -> ExpressionSet:
        pass

    @abstractmethod
    def __copy__(self) -> ExpressionSet:
        pass

    @abstractmethod
    def __add__(self, other) -> ExpressionSet:
        pass

    @abstractmethod
    def __eq__(self, other: ExpressionSet) -> bool:
        # Override this method and provide additional checks after calling super().__eq__(other)
        if type(self) != type(other):
            return False
        if not self.functions == other.functions:
            return False
        if not array_equal(self.latencies, other.latencies):
            return False
        if not array_equal(self._channels, other._channels):
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
        _VERSION = "0.2"

        warn("Experimental function. "
             "The on-disk data format for ExpressionSet is not yet fixed. "
             "Files saved using .save should not (yet) be treated as stable or future-proof.")

        if isinstance(to_path_or_file, str):
            to_path_or_file = Path(to_path_or_file)
        if isinstance(to_path_or_file, Path) and to_path_or_file.exists() and not overwrite:
            raise FileExistsError(to_path_or_file)

        with open_or_use(to_path_or_file, mode="wb") as f, ZipFile(f, "w", compression=compression) as zf:
            zf.writestr("_metadata/format-version.txt", _VERSION)
            zf.writestr("_metadata/expression-set-type.txt", self._file_format_identifier)
            zf.writestr("/channels.txt",  "\n".join(str(x) for x in self._channels))
            zf.writestr("/latencies.txt", "\n".join(str(x) for x in self.latencies))
            zf.writestr("/functions.txt", "\n".join(str(x) for x in self.functions))
            zf.writestr("/layers.txt",    "\n".join(str(x) for x in self._layers))

            for layer in self._layers:
                zf.writestr(f"/{layer}/coo-coords.bytes", self._data[layer].data.coords.tobytes(order="C"))
                zf.writestr(f"/{layer}/coo-data.bytes", self._data[layer].data.data.tobytes(order="C"))
                # The shape can be inferred, but we save it as an extra validation
                zf.writestr(f"/{layer}/coo-shape.txt", "\n".join(str(x) for x in self._data[layer].data.shape))

    @classmethod
    # TODO: return type
    def _load_data(cls, from_path_or_file: path_type | file_type):
        """
        Load an ExpressionSet from an open file, or the file at the specified path.

        If an open file is supplied, it should be opened in "rb" mode.
        """

        if isinstance(from_path_or_file, str):
            from_path_or_file = Path(from_path_or_file)

        with open_or_use(from_path_or_file, mode="rb") as archive, ZipFile(archive, "r") as zf:
            with TextIOWrapper(zf.open("/_metadata/expression-set-type.txt"), encoding="utf-8") as f:
                es_type: str = str(f.read()).strip()

            with TextIOWrapper(zf.open("/layers.txt"), encoding="utf-8") as f:
                layers: list[str] = [str(l.strip()) for l in f.readlines()]

            with TextIOWrapper(zf.open("/channels.txt"), encoding="utf-8") as f:
                channels: list[str] = [c.strip() for c in f.readlines()]

            with TextIOWrapper(zf.open("/latencies.txt"), encoding="utf-8") as f:
                latencies: list[_LatencyDType] = [_LatencyDType(lat.strip()) for lat in f.readlines()]
            with TextIOWrapper(zf.open("/functions.txt"), encoding="utf-8") as f:
                functions: list[_FunctionNameDType] = [_FunctionNameDType(fun.strip()) for fun in f.readlines()]
            data_dict = dict()
            for layer in layers:
                with zf.open(f"/{layer}/coo-coords.bytes") as f:
                    coords: ndarray = frombuffer(f.read(), dtype=int).reshape((3, -1))
                with zf.open(f"/{layer}/coo-data.bytes") as f:
                    data: ndarray = frombuffer(f.read(), dtype=float)
                with TextIOWrapper(zf.open(f"/{layer}/coo-shape.txt"), encoding="utf-8") as f:
                    shape: tuple[int, ...] = tuple(int(s.strip()) for s in f.readlines())
                data_dict[layer] = COO(coords=coords, data=data, shape=shape, prune=True, fill_value=1.0)
                # In case there was only 1 function and we have a 2-d data matrix
                # TODO: does this ever actually happen?
                if len(shape) == 2:
                    data_dict[layer] = expand_dims(data_dict[layer])
                assert shape == (len(channels), len(latencies), len(functions))

        assert all_equal([sp.shape for _layer, sp in data_dict.items()])

        return (
            functions,
            channels,
            latencies,
            data_dict,
        )

    @classmethod
    @abstractmethod
    def load(cls, from_path_or_file: path_type | file_type) -> ExpressionSet:
        """
        Load an ExpressionSet from an open file, or the file at the specified path.

        If an open file is supplied, it should be opened in "rb" mode.
        """
        # To override this, call _load_data and then call the appropriate constructor
        pass

    def _best_functions_for_layer(self, layer: str) -> DataFrame:
        """
        Return a DataFrame containing:
        for each channel, the best function and latency for that channel, and the associated p-value
        """
        # Want, for each channel:
        #  - The name, f, of the function which is best at any latency
        #  - The latency, l, for which f is best
        #  - The p-value, p, for f at l

        # sparse.COO doesn't implement argmin, so we have to do it in a few steps

        data = self._data.copy()
        densify_dataset(data)

        best_latency = data.idxmin(dim=_LATENCY)    # (channel, function) → l, the best latency
        p_at_best_latency = data.min(dim=_LATENCY)  # (channel, function) → p of best latency for each function

        p_at_best_function = p_at_best_latency.min(dim=_FUNCTION)  # (channel) → p of best function (at best latency)
        best_function = p_at_best_latency.idxmin(dim=_FUNCTION)  # (channel) → f, the best function

        # TODO: shame I have to break into the layer structure here,
        #  but I can't think of a better way to do it
        p_vals = p_at_best_function[layer].data

        best_functions = best_function[layer].data

        best_latencies = best_latency[layer].sel({_HEXEL: self._channels, _FUNCTION: best_function[_LEFT]}).data

        # Cut out channels which have a best p-val of 1
        idxs = p_vals < 1

        return DataFrame.from_dict({
            _HEXEL: self._channels[idxs],
            _FUNCTION: best_functions[idxs],
            _LATENCY: best_latencies[idxs],
            "value": p_vals[idxs],
        })

    @abstractmethod
    def best_functions(self):
        """
        Return a pair of DataFrames (left, right), containing:
        for each hexel, the best function and latency for that hexel, and the associated p-value
        """
        pass


class HexelExpressionSet(ExpressionSet):
    """
    Brain data associated with expression of a single function in hexel space.
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

        super().__init__(
            functions=functions,
            latencies=latencies,
            data_layers={
                _LEFT: data_lh,
                _RIGHT: data_rh,
            },
            channel_coord_name=_HEXEL,
            channel_coord_dtype=_HexelDType,
            channel_coord_values=hexels,
            file_format_identifier="hexel",
        )

    @property
    def hexels(self) -> NDArray[_HexelDType]:
        """Hexels, canonical ID."""
        return self._channels

    @property
    def left(self) -> DataArray:
        """Left-hemisphere data."""
        return self._data[_LEFT]

    @property
    def right(self) -> DataArray:
        """Right-hemisphere data."""
        return self._data[_RIGHT]

    def __getitem__(self, functions: str | Sequence[str]) -> HexelExpressionSet:
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
        return HexelExpressionSet(
            functions=functions,
            hexels=self.hexels,
            latencies=self.latencies,
            data_lh=[self._data[_LEFT].sel({_FUNCTION: function}).data for function in functions],
            data_rh=[self._data[_RIGHT].sel({_FUNCTION: function}).data for function in functions],
        )

    def __copy__(self):
        return HexelExpressionSet(
            functions=self.functions.copy(),
            hexels=self.hexels.copy(),
            latencies=self.latencies.copy(),
            data_lh=self._data[_LEFT].values.copy(),
            data_rh=self._data[_RIGHT].values.copy(),
        )

    def __add__(self, other: HexelExpressionSet) -> HexelExpressionSet:
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
        return HexelExpressionSet(
            functions=functions,
            hexels=self.hexels, latencies=self.latencies,
            data_lh=data_lh, data_rh=data_rh,
        )

    def __eq__(self, other: HexelExpressionSet) -> bool:
        if not super().__eq__(other):
            return False
        if not COO(self.left.data == other.left.data).all():
            return False
        if not COO(self.right.data == other.right.data).all():
            return False
        return True

    @classmethod
    def load(cls, from_path_or_file: path_type | file_type) -> HexelExpressionSet:
        """
        Load an ExpressionSet from an open file, or the file at the specified path.

        If an open file is supplied, it should be opened in "rb" mode.
        """

        functions, channels, latencies, data_dict = cls._load_data(from_path_or_file)

        return HexelExpressionSet(
            functions=functions,
            hexels=[_HexelDType(c) for c in channels],
            latencies=latencies,
            data_lh=[data_dict[_LEFT][:, :, i] for i in range(len(functions))],
            data_rh=[data_dict[_RIGHT][:, :, i] for i in range(len(functions))],
        )

    def best_functions(self) -> Tuple[DataFrame, DataFrame]:
        """
        Return a pair of DataFrames (left, right), containing:
        for each hexel, the best function and latency for that hexel, and the associated p-value
        """
        return (
            super()._best_functions_for_layer(_LEFT),
            super()._best_functions_for_layer(_RIGHT),
        )


class SensorExpressionSet(ExpressionSet):
    """
    Brain data associated with expression of a single function in sensor space.
    Includes lh, rh, flipped, non-flipped.
    """

    def __init__(self,
                 functions: str | Sequence[str],
                 # Metadata
                 sensors: Sequence[Sensor],
                 latencies: Sequence[Latency],
                 # In general, we will combine flipped and non-flipped versions
                 data: _InputDataArray | Sequence[_InputDataArray]):
        # TODO: Docstring

        super().__init__(
            functions=functions,
            latencies=latencies,
            data_layers={
                _SCALP: data
            },
            channel_coord_name=_SENSOR,
            channel_coord_dtype=_SensorDType,
            channel_coord_values=sensors,
            file_format_identifier="sensor",
        )

    @property
    def sensors(self) -> NDArray[_SensorDType]:
        """Channel names."""
        return self._channels

    @property
    def scalp(self) -> DataArray:
        """Left-hemisphere data."""
        return self._data[_SCALP]

    def __eq__(self, other: SensorExpressionSet) -> bool:
        if not super().__eq__(other):
            return False
        if not COO(self.scalp.data == other.scalp.data).all():
            return False
        return True

    def __copy__(self):
        return SensorExpressionSet(
            functions=self.functions.copy(),
            sensors=self.sensors.copy(),
            latencies=self.latencies.copy(),
            data=self._data[_SCALP].values.copy(),
        )

    def __add__(self, other: SensorExpressionSet) -> SensorExpressionSet:
        assert array_equal(self.sensors, other.sensors), "Sensors mismatch"
        assert array_equal(self.latencies, other.latencies), "Latencies mismatch"
        # constructor expects a sequence of function names and sequences of 2d matrices
        functions = []
        data = []
        for expr_set in [self, other]:
            for i, function in enumerate(expr_set.functions):
                functions.append(function)
                data.append(expr_set._data[_SCALP].data[:, :, i])
        return SensorExpressionSet(
            functions=functions,
            sensors=self.sensors, latencies=self.latencies,
            data=data,
        )

    def __getitem__(self, functions: str | Sequence[str]) -> SensorExpressionSet:
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
        return SensorExpressionSet(
            functions=functions,
            sensors=self.sensors,
            latencies=self.latencies,
            data=[self._data[_SCALP].sel({_FUNCTION: function}).data for function in functions],
        )

    def best_functions(self) -> DataFrame:
        """
        Return a pair of DataFrames (left, right), containing:
        for each hexel, the best function and latency for that hexel, and the associated p-value
        """
        return super()._best_functions_for_layer(_SCALP)

    @classmethod
    def load(cls, from_path_or_file: path_type | file_type) -> SensorExpressionSet:
        """
        Load an ExpressionSet from an open file, or the file at the specified path.

        If an open file is supplied, it should be opened in "rb" mode.
        """

        functions, channels, latencies, data_dict = cls._load_data(from_path_or_file)

        return SensorExpressionSet(
            functions=functions,
            sensors=[_SensorDType(c) for c in channels],
            latencies=latencies,
            data=[data_dict[_SCALP][:, :, i] for i in range(len(functions))],
        )
