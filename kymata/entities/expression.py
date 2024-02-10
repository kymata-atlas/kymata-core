"""
Classes and functions for storing expression information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Union, get_args, Tuple

from numpy import array, array_equal, ndarray
from numpy.typing import NDArray
from pandas import DataFrame
from sparse import SparseArray, COO
from xarray import DataArray, concat

from kymata.entities.datatypes import HexelDType, SensorDType, LatencyDType, FunctionNameDType, Hexel, Sensor, \
    Latency
from kymata.entities.iterables import all_equal
from kymata.entities.sparse_data import expand_dims, densify_data_block, sparsify_log_pmatrix

_InputDataArray = Union[ndarray, SparseArray]  # Type alias for data which can be accepted

# Data dimension labels
DIM_HEXEL = "hexel"
DIM_SENSOR = "sensor"
DIM_LATENCY = "latency"
DIM_FUNCTION = "function"

# Block (e.g. hemisphere)
BLOCK_LEFT  = "left"
BLOCK_RIGHT = "right"
BLOCK_SCALP = "scalp"


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
                 # data_blocks contains a dict mapping block names to data arrays
                 # e.g., in the case there are three functions:
                 #             "left" → [array(), array(), array()],
                 #        and "right" → [array(), array(), array()]
                 #   or:      "scalp" → [array(), array(), array()]
                 # All should be the same size.
                 data_blocks: dict[str, _InputDataArray | Sequence[_InputDataArray]],  # log p-values
                 # Supply channels already as an array, i.e.
                 channel_coord_name: str,
                 channel_coord_dtype,
                 channel_coord_values: dict[str, Sequence],
                 ):
        """
        data_blocks' values should store log10 p-values
        """

        # Canonical order of dimensions
        self._dims = (channel_coord_name, DIM_LATENCY, DIM_FUNCTION)

        self._block_names: list[str] = list(data_blocks.keys())
        self._channel_coord_name = channel_coord_name

        # Validate arguments
        assert set(self._block_names) == set(channel_coord_values.keys()), "Ensure data block names match channel block names"
        _length_mismatch_message = ("Argument length mismatch, please supply one function name and accompanying data, "
                                    "or equal-length sequences of the same.")
        if isinstance(functions, str):
            # If only one function
            for data in data_blocks.values():
                # Data should not be a sequence
                assert isinstance(data, get_args(_InputDataArray)), _length_mismatch_message
            # Wrap into sequences
            functions = [functions]
            for bn in self._block_names:
                data_blocks[bn] = [data_blocks[bn]]

        assert len(functions) == len(set(functions)), "Duplicated functions in input"
        for data in data_blocks.values():
            assert len(functions) == len(data), _length_mismatch_message
        assert all_equal([arr.shape[1] for arrs in data_blocks.values() for arr in arrs]), "Not all input data blocks have the same"

        # Channels can vary between blocks (e.g. different numbers of vertices for each hemisphere).
        # But latencies and functions do not
        channels: dict[str, NDArray] = {
            bn: array(channel_coord_values[bn], dtype=channel_coord_dtype)
            for bn in self._block_names
        }
        latencies = array(latencies, dtype=LatencyDType)
        functions = array(functions, dtype=FunctionNameDType)

        # Input value `data_blocks` has type something like dict[str, list[array]].
        #  i.e. a dict mapping block names to a list of 2d data volumes, one for each function
        # We need to eventually get it into `self._data`, which has type dict[str, DataArray]
        # i.e. a dict mapping block names to a DataArray containing data for all functions
        self._data: dict[str, DataArray] = dict()
        for block_name, data_for_functions in data_blocks.items():
            for function_name, data in zip(functions, data_for_functions):
                assert len(channels[block_name]) == data.shape[0], f"{channel_coord_name} mismatch for {function_name}: {len(channels)} {channel_coord_name} versus data shape {data.shape} ({block_name})"
                assert len(latencies) == data.shape[1], f"Latencies mismatch for {function_name}: {len(latencies)} latencies versus data shape {data.shape}"
            data_array: DataArray = concat(
                (
                    DataArray(self._init_prep_data(d),
                              coords={
                                  channel_coord_name: channels[block_name],
                                  DIM_LATENCY: latencies,
                                  DIM_FUNCTION: [function]
                              })
                    for function, d in zip(functions, data_for_functions)
                ),
                dim=DIM_FUNCTION,
                data_vars="all",  # Required by concat of DataArrays
                )
            assert data_array.dims == self._dims
            assert set(data_array.coords.keys()) == set(self._dims)
            assert array_equal(data_array.coords[DIM_FUNCTION].values, functions)

            self._data[block_name] = data_array

    @classmethod
    def _init_prep_data(cls, data: _InputDataArray) -> COO:
        if isinstance(data, ndarray):
            data = sparsify_log_pmatrix(data)
        elif not isinstance(data, SparseArray):
            raise NotImplementedError()
        data = expand_dims(data, 2)
        return data

    @property
    # block → channels
    def _channels(self) -> dict[str, NDArray]:
        return {
            bn: data.coords[self._channel_coord_name].values
            for bn, data in self._data.items()
        }

    @property
    def functions(self) -> list[FunctionNameDType]:
        """Function names."""
        functions = {
            bn: data.coords[DIM_FUNCTION].values.tolist()
            for bn, data in self._data.items()
        }
        # Validate that functions are the same for all data blocks
        assert all_equal(list(functions.values()))
        # Then just return the first one
        return functions[self._block_names[0]]

    @property
    def latencies(self) -> NDArray[LatencyDType]:
        """Latencies, in seconds."""
        latencies = {
            bn: data.coords[DIM_LATENCY].values
            for bn, data in self._data.items()
        }
        # Validate that latencies are the same for all data blocks
        assert all_equal(list(latencies.values()))
        # Then just return the first one
        return latencies[self._block_names[0]]

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
        for bn in self._block_names:
            if not array_equal(self._channels[bn], other._channels[bn]):
                return False
        return True

    def _best_functions_for_block(self, block_name: str) -> DataFrame:
        """
        Return a DataFrame containing:
        for each channel in the block, the best function and latency for that channel, and the associated log p-value
        """
        # Want, for each channel in the block:
        #  - The name, f, of the function which is best at any latency
        #  - The latency, l, for which f is best
        #  - The log p-value, p, for f at l

        # sparse.COO doesn't implement argmin, so we have to do it in a few steps

        data: DataArray = self._data[block_name].copy()
        densify_data_block(data)

        # (channel, function) → l, the best latency
        best_latency = data.idxmin(dim=DIM_LATENCY)
        # (channel, function) → log p of best latency for each function
        logp_at_best_latency = data.min(dim=DIM_LATENCY)

        # (channel) → log p of best function (at best latency)
        logp_at_best_function = logp_at_best_latency.min(dim=DIM_FUNCTION)
        # (channel) → f, the best function
        best_function = logp_at_best_latency.idxmin(dim=DIM_FUNCTION)

        logp_vals = logp_at_best_function.data
        best_functions = best_function.data

        best_latencies = best_latency.sel({
            self._channel_coord_name: self._channels,
            DIM_FUNCTION: best_function[block_name]
        }).data

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
                 hexels_lh: Sequence[Hexel],
                 hexels_rh: Sequence[Hexel],
                 latencies: Sequence[Latency],
                 # log p-values
                 # In general, we will combine flipped and non-flipped versions
                 data_lh: _InputDataArray | Sequence[_InputDataArray],
                 data_rh: _InputDataArray | Sequence[_InputDataArray],
                 ):

        super().__init__(
            functions=functions,
            latencies=latencies,
            data_blocks={
                BLOCK_LEFT: data_lh,
                BLOCK_RIGHT: data_rh,
            },
            channel_coord_name=DIM_HEXEL,
            channel_coord_dtype=HexelDType,
            channel_coord_values={
                BLOCK_LEFT: hexels_lh,
                BLOCK_RIGHT: hexels_rh
            },
        )

    @property
    def hexels_left(self) -> NDArray[HexelDType]:
        """Hexels, canonical ID."""
        return self._channels[BLOCK_LEFT]

    @property
    def hexels_right(self) -> NDArray[HexelDType]:
        """Hexels, canonical ID."""
        return self._channels[BLOCK_RIGHT]

    @property
    def left(self) -> DataArray:
        """Left-hemisphere data."""
        return self._data[BLOCK_LEFT]

    @property
    def right(self) -> DataArray:
        """Right-hemisphere data."""
        return self._data[BLOCK_RIGHT]

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
            hexels_lh=self.hexels_left,
            hexels_rh=self.hexels_right,
            latencies=self.latencies,
            data_lh=[self._data[BLOCK_LEFT].sel({DIM_FUNCTION: function}).data for function in functions],
            data_rh=[self._data[BLOCK_RIGHT].sel({DIM_FUNCTION: function}).data for function in functions],
        )

    def __copy__(self):
        return HexelExpressionSet(
            functions=self.functions.copy(),
            hexels_lh=self.hexels_left.copy(),
            hexels_rh=self.hexels_right.copy(),
            latencies=self.latencies.copy(),
            data_lh=self._data[BLOCK_LEFT].values.copy(),
            data_rh=self._data[BLOCK_RIGHT].values.copy(),
        )

    def __add__(self, other: HexelExpressionSet) -> HexelExpressionSet:
        assert array_equal(self.hexels_left, other.hexels_left), "Hexels mismatch (left)"
        assert array_equal(self.hexels_right, other.hexels_right), "Hexels mismatch (right)"
        assert array_equal(self.latencies, other.latencies), "Latencies mismatch"
        # constructor expects a sequence of function names and sequences of 2d matrices
        functions = []
        data_lh = []
        data_rh = []
        for expr_set in [self, other]:
            for i, function in enumerate(expr_set.functions):
                functions.append(function)
                data_lh.append(expr_set._data[BLOCK_LEFT].data[:, :, i])
                data_rh.append(expr_set._data[BLOCK_RIGHT].data[:, :, i])
        return HexelExpressionSet(
            functions=functions,
            hexels_lh=self.hexels_left, hexels_rh=self.hexels_right,
            latencies=self.latencies,
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
            super()._best_functions_for_block(BLOCK_LEFT),
            super()._best_functions_for_block(BLOCK_RIGHT),
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
            data_blocks={
                BLOCK_SCALP: data
            },
            channel_coord_name=DIM_SENSOR,
            channel_coord_dtype=SensorDType,
            channel_coord_values={BLOCK_SCALP: sensors},
        )

    @property
    def sensors(self) -> NDArray[SensorDType]:
        """Channel names."""
        return self._channels[BLOCK_SCALP]

    @property
    def scalp(self) -> DataArray:
        """Left-hemisphere data."""
        return self._data[BLOCK_SCALP]

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
            data=self._data[BLOCK_SCALP].values.copy(),
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
                data.append(expr_set._data[BLOCK_SCALP].data[:, :, i])
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
            data=[self._data[BLOCK_SCALP].sel({DIM_FUNCTION: function}).data for function in functions],
        )

    def best_functions(self) -> DataFrame:
        """
        Return a DataFrame containing:
        for each sensor, the best function and latency for that sensor, and the associated log p-value
        """
        return super()._best_functions_for_block(BLOCK_SCALP)
