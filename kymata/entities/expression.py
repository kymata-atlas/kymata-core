"""
Classes and functions for storing expression information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Union, get_args, Tuple

from numpy import array, array_equal, ndarray, log10
from numpy.typing import NDArray, ArrayLike
from pandas import DataFrame
from sparse import SparseArray, COO
from xarray import DataArray, Dataset, concat

from kymata.entities.datatypes import HexelDType, SensorDType, LatencyDType, FunctionNameDType, Hexel, Sensor, \
    Latency
from kymata.entities.iterables import all_equal
from kymata.entities.sparse_data import expand_dims, densify_dataset, sparsify_log_pmatrix

_InputDataArray = Union[ndarray, SparseArray]  # Type alias for data which can be accepted

# Data dimension labels
DIM_HEXEL = "hexel"
DIM_SENSOR = "sensor"
DIM_LATENCY = "latency"
DIM_FUNCTION = "function"

# Layer (e.g. hemisphere)
LAYER_LEFT  = "left"
LAYER_RIGHT = "right"
LAYER_SCALP = "scalp"


class ExpressionSet(ABC):
    """
    Brain data associated with expression of a single function.
    Data is log10 p-values
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
                 data_layers: dict[str, _InputDataArray | Sequence[_InputDataArray]],  # log p-values
                 # Supply channels already as an array, i.e.
                 channel_coord_name: str,
                 channel_coord_dtype,
                 channel_coord_values: Sequence,
                 ):
        """data_layers' values should store log10 p-values"""

        self._layers: list[str] = list(data_layers.keys())

        self._channel_coord_name = channel_coord_name
        self._dims = (channel_coord_name, DIM_LATENCY, DIM_FUNCTION)  # Canonical order of dimensions

        # Validate arguments
        _length_mismatch_message = ("Argument length mismatch, please supply one function name and accompanying data, "
                                    "or equal-length sequences of the same.")
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
        for layer, data in data_layers.items():
            assert len(functions) == len(data), _length_mismatch_message
        assert all_equal([arr.shape for _layer, arrs in data_layers.items() for arr in arrs])

        channels  = array(channel_coord_values, dtype=channel_coord_dtype)
        latencies = array(latencies, dtype=LatencyDType)
        functions = array(functions, dtype=FunctionNameDType)

        datasets = []
        for i, f in enumerate(functions):
            dataset_dict = dict()
            for layer, data in data_layers.items():
                # Get this function's data
                data = data[i]
                data = self._init_prep_data(data)
                # Check validity of input data dimensions
                assert len(channels) == data.shape[0], f"{channel_coord_name} mismatch for {f}: {len(channels)} {channel_coord_name} versus data shape {data.shape}"
                assert len(latencies) == data.shape[1], f"Latencies mismatch for {f}: {len(latencies)} latencies versus data shape {data.shape}"
                dataset_dict[layer] = DataArray(
                    data=data,
                    dims=self._dims,
                )
            datasets.append(
                Dataset(dataset_dict,
                        coords={channel_coord_name: channels, DIM_LATENCY: latencies, DIM_FUNCTION: [f]})
            )
        self._data = concat(datasets, dim=DIM_FUNCTION)

    @classmethod
    def _init_prep_data(cls, data: _InputDataArray) -> COO:
        if isinstance(data, ndarray):
            data = sparsify_log_pmatrix(data)
        elif not isinstance(data, SparseArray):
            raise NotImplementedError()
        data = expand_dims(data, 2)
        return data

    @property
    def _channels(self) -> NDArray:
        return self._data.coords[self._channel_coord_name].values

    @property
    def functions(self) -> list[FunctionNameDType]:
        """Function names."""
        return self._data.coords[DIM_FUNCTION].values.tolist()

    @property
    def latencies(self) -> NDArray[LatencyDType]:
        """Latencies, in seconds."""
        return self._data.coords[DIM_LATENCY].values

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

    def _best_functions_for_layer(self, layer: str) -> DataFrame:
        """
        Return a DataFrame containing:
        for each channel, the best function and latency for that channel, and the associated log p-value
        """
        # Want, for each channel:
        #  - The name, f, of the function which is best at any latency
        #  - The latency, l, for which f is best
        #  - The log p-value, p, for f at l

        # sparse.COO doesn't implement argmin, so we have to do it in a few steps

        data = self._data.copy()
        densify_dataset(data)

        best_latency = data.idxmin(dim=DIM_LATENCY)    # (channel, function) → l, the best latency
        logp_at_best_latency = data.min(dim=DIM_LATENCY)  # (channel, function) → log p of best latency for each function

        logp_at_best_function = logp_at_best_latency.min(dim=DIM_FUNCTION)  # (channel) → log p of best function (at best latency)
        best_function = logp_at_best_latency.idxmin(dim=DIM_FUNCTION)  # (channel) → f, the best function

        # TODO: shame I have to break into the layer structure here,
        #  but I can't think of a better way to do it
        logp_vals = logp_at_best_function[layer].data

        best_functions = best_function[layer].data

        best_latencies = best_latency[layer].sel({self._channel_coord_name: self._channels, DIM_FUNCTION: best_function[layer]}).data

        # Cut out channels which have a best log p-val of 1
        idxs = logp_vals < 1

        return DataFrame.from_dict({
            self._channel_coord_name: self._channels[idxs],
            DIM_FUNCTION: best_functions[idxs],
            DIM_LATENCY: best_latencies[idxs],
            "value": logp_vals[idxs],
        })

    @abstractmethod
    def best_functions(self) -> DataFrame | tuple[DataFrame, ...]:
        pass


class HexelExpressionSet(ExpressionSet):
    """
    Brain data associated with expression of a single function in hexel space.
    Includes lh, rh, flipped, non-flipped.
    Data is log10 p-values
    """

    def __init__(self,
                 functions: str | Sequence[str],
                 # Metadata
                 hexels: Sequence[Hexel],
                 latencies: Sequence[Latency],
                 # log p-values
                 # In general, we will combine flipped and non-flipped versions
                 data_lh: _InputDataArray | Sequence[_InputDataArray],
                 data_rh: _InputDataArray | Sequence[_InputDataArray],
                 ):

        super().__init__(
            functions=functions,
            latencies=latencies,
            data_layers={
                LAYER_LEFT: data_lh,
                LAYER_RIGHT: data_rh,
            },
            channel_coord_name=DIM_HEXEL,
            channel_coord_dtype=HexelDType,
            channel_coord_values=hexels,
        )

    @property
    def hexels(self) -> NDArray[HexelDType]:
        """Hexels, canonical ID."""
        return self._channels

    @property
    def left(self) -> DataArray:
        """Left-hemisphere data."""
        return self._data[LAYER_LEFT]

    @property
    def right(self) -> DataArray:
        """Right-hemisphere data."""
        return self._data[LAYER_RIGHT]

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
            data_lh=[self._data[LAYER_LEFT].sel({DIM_FUNCTION: function}).data for function in functions],
            data_rh=[self._data[LAYER_RIGHT].sel({DIM_FUNCTION: function}).data for function in functions],
        )

    def __copy__(self):
        return HexelExpressionSet(
            functions=self.functions.copy(),
            hexels=self.hexels.copy(),
            latencies=self.latencies.copy(),
            data_lh=self._data[LAYER_LEFT].values.copy(),
            data_rh=self._data[LAYER_RIGHT].values.copy(),
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
                data_lh.append(expr_set._data[LAYER_LEFT].data[:, :, i])
                data_rh.append(expr_set._data[LAYER_RIGHT].data[:, :, i])
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

    def best_functions(self) -> Tuple[DataFrame, DataFrame]:
        """
        Return a pair of DataFrames (left, right), containing:
        for each hexel, the best function and latency for that hexel, and the associated log p-value
        """
        return (
            super()._best_functions_for_layer(LAYER_LEFT),
            super()._best_functions_for_layer(LAYER_RIGHT),
        )


class SensorExpressionSet(ExpressionSet):
    """
    Brain data associated with expression of a single function in sensor space.
    Includes lh, rh, flipped, non-flipped.
    Data is log10 p-values
    """

    def __init__(self,
                 functions: str | Sequence[str],
                 # Metadata
                 sensors: Sequence[Sensor],
                 latencies: Sequence[Latency],
                 # log p-values
                 # In general, we will combine flipped and non-flipped versions
                 data: _InputDataArray | Sequence[_InputDataArray],
                 ):
        # TODO: Docstring

        super().__init__(
            functions=functions,
            latencies=latencies,
            data_layers={
                LAYER_SCALP: data
            },
            channel_coord_name=DIM_SENSOR,
            channel_coord_dtype=SensorDType,
            channel_coord_values=sensors,
        )

    @property
    def sensors(self) -> NDArray[SensorDType]:
        """Channel names."""
        return self._channels

    @property
    def scalp(self) -> DataArray:
        """Left-hemisphere data."""
        return self._data[LAYER_SCALP]

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
            data=self._data[LAYER_SCALP].values.copy(),
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
                data.append(expr_set._data[LAYER_SCALP].data[:, :, i])
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
            data=[self._data[LAYER_SCALP].sel({DIM_FUNCTION: function}).data for function in functions],
        )

    def best_functions(self) -> DataFrame:
        """
        Return a DataFrame containing:
        for each sensor, the best function and latency for that sensor, and the associated log p-value
        """
        return super()._best_functions_for_layer(LAYER_SCALP)


log_base = 10


def p_to_logp(arraylike: ArrayLike) -> ArrayLike:
    """The one-stop-shop for converting from p-values to log p-values."""
    return log10(arraylike)


def logp_to_p(arraylike: ArrayLike) -> ArrayLike:
    """The one-stop-shop for converting from log p-values to p-values."""
    return float(10) ** arraylike
