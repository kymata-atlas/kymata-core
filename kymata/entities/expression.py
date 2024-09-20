"""
Classes and functions for storing expression information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Union, get_args, Tuple, TypeVar
from warnings import warn

from numpy import (
    # Can't use NDArray for isinstance checks
    ndarray,
    array, array_equal)
from numpy.typing import NDArray
from pandas import DataFrame
from sparse import SparseArray, COO
from xarray import DataArray, concat

from kymata.entities.constants import HEMI_LEFT, HEMI_RIGHT
from kymata.entities.datatypes import (
    HexelDType, SensorDType, LatencyDType, FunctionNameDType, Hexel, Sensor, Latency)
from kymata.entities.iterables import all_equal
from kymata.entities.sparse_data import (
    expand_dims, densify_data_block, sparsify_log_pmatrix)


_InputDataArray = Union[ndarray, SparseArray]  # Type alias for data which can be accepted

# Data dimension labels
DIM_HEXEL = "hexel"
DIM_SENSOR = "sensor"
DIM_LATENCY = "latency"
DIM_FUNCTION = "function"

# Block (e.g. hemisphere)
BLOCK_LEFT = HEMI_LEFT
BLOCK_RIGHT = HEMI_RIGHT
BLOCK_SCALP = "scalp"

# Column names
COL_LOGP_VALUE = "log-p-value"


class ExpressionSet(ABC):
    """
    Brain data associated with expression of a single function.
    Data is log10 p-values.
    """

    def __init__(
            self,
            functions: str | Sequence[str],
            # Metadata
            latencies: Sequence[Latency],
            # In general, we will combine flipped and non-flipped versions.
            data_blocks: dict[str, _InputDataArray | Sequence[_InputDataArray]],
            channel_coord_name: str,
            channel_coord_dtype,
            channel_coord_values: dict[str, Sequence],
    ):
        """
        Initializes the ExpressionSet with the provided data.

        Args:
            functions (str | Sequence[str]): Function name, or sequence of names.
            latencies (Sequence[Latency]): Latency values.
            data_blocks (dict[str, _InputDataArray | Sequence[_InputDataArray]]):
                Mapping of block names to data arrays (log10 p-values).

                In general there are two possible formats for this argument.

                In the first (safer, more explicit and flexible) format, `data_blocks` contains a dict mapping block
                names to data arrays. E.g., in the case there are three functions in a hexel setting:
                ```
                    {
                        "left":  [array(...), array(...), array(...)],
                        "right": [array(...), array(...), array(...)],
                    }
                ```
                or in a sensor setting:
                ```
                    {
                        "scalp": [array(), array(), array()],
                    }
                ```
                In this format, all data arrays should be the same size.

                In the second (more performant) format, `data_blocks` contains a single data array whose `function`
                dimensions can be concatenated to achieve the desired resultant data block.
            channel_coord_name (str): Name of the channel coordinate.
            channel_coord_dtype: Data type of the channel coordinate.
            channel_coord_values (dict[str, Sequence]): Dictionary mapping block names to channel coordinate values.

        Raises:
            ValueError: when arguments are invalid
        """

        # Canonical order of dimensions
        self._dims = (channel_coord_name, DIM_LATENCY, DIM_FUNCTION)

        self._block_names: list[str] = list(data_blocks.keys())
        self.channel_coord_name = channel_coord_name

        #########################################
        # region Validate and normalise arguments
        #########################################

        if set(self._block_names) != set(channel_coord_values.keys()):
            raise ValueError("Ensure data block names match channel block names")

        data_supplied_as_sequence = self._validate_data_supplied_as_sequence(data_blocks)

        # If only one function supplied, ensure that the other arguments are appropriately specified
        if isinstance(functions, str):
            if data_supplied_as_sequence:
                raise ValueError("Single function name provided but data came as a sequence")
            # Only one function, so no matter what format the data is in, we don't expect a sequence
            # Wrap function names and data blocks to ensure we have sequences from now on
            functions = [functions]

        self._validate_functions_no_duplicates(functions)

        # If data was specified as sequence, ensure all arrays in the sequence have the same shape
        if data_supplied_as_sequence:
            if not all_equal([
                # Only check latencies, as channels can vary between blocks (e.g. different hexels per hemi) and there
                # is only one function per item in the sequence
                block_data_array.shape[1]
                for block_data in data_blocks.values()
                for block_data_array in block_data  # data format sequence
            ]):
                raise ValueError("Not all input data blocks have the same shape.")
        # Otherwise ensure all blocks have the same size
        else:
            if not all_equal([
                # Only check latencies, as channels can vary between blocks (e.g. different hexels per hemi) and there
                # is only one function per item in the sequence
                block_data_array.shape[1]
                for block_data_array in data_blocks.values()
            ]):
                raise ValueError("Not all input data blocks have the same shape")

        # Additional validity checks may happen during construction

        # Channels can vary between blocks (e.g. different numbers of vertices for each hemisphere).
        # But latencies and functions do not
        channels: dict[str, NDArray] = {
            bn: array(channel_coord_values[bn], dtype=channel_coord_dtype)
            for bn in self._block_names
        }
        latencies = array(latencies, dtype=LatencyDType)
        functions = array(functions, dtype=FunctionNameDType)

        for block_name, block_data in data_blocks.items():
            if data_supplied_as_sequence:
                for function_name, data in zip(functions, block_data):
                    if len(channels[block_name]) != data.shape[0]:
                        raise ValueError(f"{channel_coord_name} mismatch for {function_name}: "
                                         f"{len(channels)} {channel_coord_name} versus data shape {data.shape} "
                                         f"({block_name})")
                    if len(latencies) != data.shape[1]:
                        raise ValueError(f"Latencies mismatch for {function_name}: "
                                         f"{len(latencies)} latencies versus data shape {data.shape} "
                                         f"({block_name})")
                    if len(data.shape) > 3:
                        raise ValueError(f"Too many dimensions in data for {function_name}: "
                                         f"shape={data.shape} "
                                         f"({block_name})")
                    if len(data.shape) == 3 and data.shape[2] != 1:
                        raise ValueError(f"Too many dimensions in data for {function_name}: "
                                         f"shape={data.shape} "
                                         f"({block_name})")
            else:
                if len(channels[block_name]) != block_data.shape[0]:
                    raise ValueError(f"{channel_coord_name} mismatch: "
                                     f"{len(channels)} {channel_coord_name} versus data shape {block_data.shape} "
                                     f"({block_name})")
                if len(latencies) != block_data.shape[1]:
                    raise ValueError(f"Latencies mismatch: "
                                     f"{len(latencies)} latencies versus data shape {block_data.shape} "
                                     f"({block_name})")
                if len(block_data.shape) > 3:
                    raise ValueError(f"Too many dimensions in data: "
                                     f"shape={block_data.shape} "
                                     f"({block_name})")
                if len(block_data.shape) == 3 and len(functions) != block_data.shape[2]:
                    raise ValueError(f"Functions mismatch: "
                                     f"{len(functions)} functions verus data shape {block_data.shape} "
                                     f"({block_name})")

        ############################################
        # endregion Validate and normalise arguments
        ############################################

        # Input value `data_blocks` has type something like dict[str, list[array]].
        #  i.e. a dict mapping block names to a list of 2d data volumes, one for each function
        # We need to eventually get it into `self._data`, which has type dict[str, DataArray]
        # i.e. a dict mapping block names to a DataArray containing data for all functions
        self._data: dict[str, DataArray] = dict()
        nan_warning_sent = False
        for block_name, block_data in data_blocks.items():
            if data_supplied_as_sequence:
                # Build DataArray by concatenating sequence
                data_array: DataArray = concat(
                    (
                        DataArray(
                            self._init_prep_data(function_data),
                            coords={
                                channel_coord_name: channels[block_name],
                                DIM_LATENCY: latencies,
                                DIM_FUNCTION: [function],
                            },
                        )
                        for function, function_data in zip(functions, block_data)
                    ),
                    dim=DIM_FUNCTION,
                    data_vars="all",  # Required by concat of DataArrays
                )
            else:
                # Build DataArray in one go
                data_array: DataArray = DataArray(
                    self._init_prep_data(block_data),
                    coords={
                        channel_coord_name: channels[block_name],
                        DIM_LATENCY: latencies,
                        DIM_FUNCTION: functions,
                    }
                )

            # Sometimes the data can contain nans, for example if the MEG hexel currents were set to nan on the medial
            # wall. We can ignore these nans by setting the values to p=1, but because it's not typically expected we
            # warn the user about it.
            if data_array.isnull().any():
                data_array = data_array.fillna(value=0)  # logp = 0 => p = 1
                # Only want to send the warning once, even if there are multiple data blocks with nans
                # (it's likely that if one has them, they all will)
                if not nan_warning_sent:
                    warn("Supplied data contained nans. These will be replaced by p = 1 values.")
                    nan_warning_sent = True
            if data_array.dims != self._dims:
                raise ValueError("DataArray had wrong dimensions")
            if set(data_array.coords.keys()) != set(self._dims):
                raise ValueError("DataArray had wrong coordinates")
            if not array_equal(data_array.coords[DIM_FUNCTION].values, functions):
                raise ValueError("DataArray had wrong functions")

            self._data[block_name] = data_array

    def _validate_data_supplied_as_sequence(self, data_blocks) -> bool:
        # Determine if data is supplied as sequence or contiguous, and that it's the same for each block
        data_supplied_as_sequence = {
            block_name: not isinstance(block_data, get_args(_InputDataArray))
            for block_name, block_data in data_blocks.items()
        }
        if not all_equal(data_supplied_as_sequence.values()):
            ValueError("Not all input data blocks have the same format (sequence or contiguous)")
        # Now we can just use the first one
        return data_supplied_as_sequence[self._block_names[0]]

    # noinspection PyMethodMayBeStatic
    def _validate_functions_no_duplicates(self, functions: Sequence[str]) -> None:
        if not len(functions) == len(set(functions)):
            checked = []
            for f in functions:
                if f in checked:
                    break
                checked.append(f)
            raise ValueError(f"Duplicated functions in input, e.g. {f}")

    @classmethod
    def _init_prep_data(cls, data: _InputDataArray) -> COO:
        # Convert to sparse matrix
        if isinstance(data, ndarray):
            data = sparsify_log_pmatrix(data)
        elif not isinstance(data, SparseArray):
            raise NotImplementedError()

        if data.ndim <= 2:
            data = expand_dims(data, axis=2)
        return data

    @property
    # block → channels
    def _channels(self) -> dict[str, NDArray]:
        return {
            bn: data.coords[self.channel_coord_name].values
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
            bn: data.coords[DIM_LATENCY].values for bn, data in self._data.items()
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

    def _add_compatibility_check(self, other) -> None:
        """
        Checks whether the `other` ExpressionSet is compatible with this one, for purposes of adding them.
        Should return silently if all is well.
        """
        # Type is the same
        if type(self) is not type(other):
            raise ValueError("Can only add ExpressionSets of the same type")
        # Channels are the same
        for bn in self._block_names:
            if not array_equal(self._channels[bn], other._channels[bn]):
                raise ValueError(
                    f"Can only add ExpressionSets with matching {self.channel_coord_name}s"
                )
        # Latencies are the same
        if not array_equal(self.latencies, other.latencies):
            raise ValueError("Can only add ExpressionSets with matching latencies")

    @abstractmethod
    def __eq__(self, other: ExpressionSet) -> bool:
        # Override this method and provide additional checks after calling super().__eq__(other)
        if type(self) is not type(other):
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

        Note that channels for which the best p-value is 1 will be omitted.
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

        # (channel) -> logp of the best function (at the best latency)
        logp_vals = logp_at_best_function.data
        # (channel) -> best function name (at the best latency)
        best_functions = best_function.data
        # (channel) -> latency of the best function
        best_latencies = best_latency.sel(
            {
                # e.g. hexels          -> array([0, ..., 10241])
                self.channel_coord_name: self._channels[block_name],
                #          -> DataArray((hexel) -> function)
                DIM_FUNCTION: best_function,
            }
        ).data

        # Cut out channels which have a best log p-val of 0 (i.e. p = 1)
        idxs = logp_vals < 0

        return DataFrame.from_dict(
            {
                self.channel_coord_name: self._channels[block_name][idxs],
                DIM_FUNCTION: best_functions[idxs],
                DIM_LATENCY: best_latencies[idxs],
                COL_LOGP_VALUE: logp_vals[idxs],
            }
        )

    def rename(self, functions: dict[str, str] = None, channels: dict = None) -> None:
        """
        Renames the functions and channels within an ExpressionSet.

        Supply a dictionary mapping old values to new values.

        Raises KeyError if one of the keys in the renaming dictionary is not a function name in the expression set.
        """
        # Default values
        if functions is None:
            functions = dict()
        if channels is None:
            channels = dict()

        # Validate
        for old, new in functions.items():
            if old not in self.functions:
                raise KeyError(f"{old} is not a function in this expression set")
        for old, new in channels.items():
            for bn in self._block_names:
                if old not in self._channels[bn]:
                    raise KeyError(
                        f"{old} is not a {bn} {self.channel_coord_name} in this expression set"
                    )

        # Replace
        for bn, data in self._data.items():
            # Functions
            new_names = []
            for old_name in self._data[bn][DIM_FUNCTION].values:
                if old_name in functions:
                    new_names.append(functions[old_name])
                else:
                    new_names.append(old_name)
            self._data[bn][DIM_FUNCTION] = new_names

            # Channels
            new_channels = []
            for old_channel in self._data[bn][self.channel_coord_name].values:
                if old_channel in channels:
                    new_channels.append(channels[old_channel])
                else:
                    new_channels.append(old_channel)
            self._data[bn][self.channel_coord_name] = new_channels

    @abstractmethod
    def best_functions(self) -> DataFrame | tuple[DataFrame, ...]:
        """
        Note that channels for which the best p-value is 1 will be omitted.
        """
        pass


class HexelExpressionSet(ExpressionSet):
    """
    Brain data associated with expression of a single function in hexel space.
    Includes lh, rh, flipped, non-flipped.
    Data is log10 p-values
    """

    def __init__(
        self,
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
            channel_coord_values={BLOCK_LEFT: hexels_lh, BLOCK_RIGHT: hexels_rh},
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
            data_lh=[
                self._data[BLOCK_LEFT].sel({DIM_FUNCTION: function}).data
                for function in functions
            ],
            data_rh=[
                self._data[BLOCK_RIGHT].sel({DIM_FUNCTION: function}).data
                for function in functions
            ],
        )

    def __copy__(self):
        data_left: NDArray = self._data[BLOCK_LEFT].data.todense()
        data_right: NDArray = self._data[BLOCK_RIGHT].data.todense()
        return HexelExpressionSet(
            functions=self.functions.copy(),
            hexels_lh=self.hexels_left.copy(),
            hexels_rh=self.hexels_right.copy(),
            latencies=self.latencies.copy(),
            # Slice by function
            data_lh=[data_left[:, :, i].copy() for i in range(data_left.shape[2])],
            data_rh=[data_right[:, :, i].copy() for i in range(data_right.shape[2])],
        )

    def __add__(self, other: HexelExpressionSet) -> HexelExpressionSet:
        self._add_compatibility_check(other)
        if not array_equal(self.hexels_left, other.hexels_left):
            raise ValueError("Hexels mismatch (left)")
        if not array_equal(self.hexels_right, other.hexels_right):
            raise ValueError("Hexels mismatch (right)")
        assert array_equal(self.latencies, other.latencies), "Latencies mismatch"
        return HexelExpressionSet(
            functions=self.functions + other.functions,
            hexels_lh=self.hexels_left,
            hexels_rh=self.hexels_right,
            latencies=self.latencies,
            data_lh=concat([self.left, other.left],
                           dim=DIM_FUNCTION,
                           data_vars="all",  # Required by concat of DataArrays
                           ).data,
            data_rh=concat([self.right, other.right],
                           dim=DIM_FUNCTION,
                           data_vars="all",  # Required by concat of DataArrays
                           ).data,
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

        Note that channels for which the best p-value is 1 will be omitted.
        """
        return (
            super()._best_functions_for_block(BLOCK_LEFT),
            super()._best_functions_for_block(BLOCK_RIGHT),
        )


class SensorExpressionSet(ExpressionSet):
    """
    Brain data associated with the expression of a single function in sensor space.
    Includes left hemisphere (lh), right hemisphere (rh), flipped, and non-flipped data.
    Data is represented as log10 p-values.
    """

    def __init__(
        self,
        functions: str | Sequence[str],
        # Metadata
        sensors: Sequence[Sensor],
        latencies: Sequence[Latency],
        # log p-values
        # In general, we will combine flipped and non-flipped versions
        data: _InputDataArray | Sequence[_InputDataArray],
    ):
        """
        Initialize the SensorExpressionSet with function names, sensor metadata, latency information, and log p-value data.

        Args:
            functions (str | Sequence[str]): The names of the functions being evaluated.
            sensors (Sequence[Sensor]): Metadata about the sensors used in the study.
            latencies (Sequence[Latency]): Latency information corresponding to the data.
            data (_InputDataArray | Sequence[_InputDataArray]): Log p-values representing the data.
        """
        super().__init__(
            functions=functions,
            latencies=latencies,
            data_blocks={BLOCK_SCALP: data},
            channel_coord_name=DIM_SENSOR,
            channel_coord_dtype=SensorDType,
            channel_coord_values={BLOCK_SCALP: sensors},
        )

    @property
    def sensors(self) -> NDArray[SensorDType]:
        """
        Get the sensor metadata.

        Returns:
            NDArray[SensorDType]: Array of sensor metadata.
        """
        return self._channels[BLOCK_SCALP]

    @property
    def scalp(self) -> DataArray:
        """
        Get the left-hemisphere data.
        """
        return self._data[BLOCK_SCALP]

    def __eq__(self, other: SensorExpressionSet) -> bool:
        if not super().__eq__(other):
            return False
        if not COO(self.scalp.data == other.scalp.data).all():
            return False
        return True

    def __copy__(self):
        data: NDArray = self._data[BLOCK_SCALP].data.todense()
        return SensorExpressionSet(
            functions=self.functions.copy(),
            sensors=self.sensors.copy(),
            latencies=self.latencies.copy(),
            # Slice by function
            data=[data[:, :, i].copy() for i in range(data.shape[2])],
        )

    def __add__(self, other: SensorExpressionSet) -> SensorExpressionSet:
        self._add_compatibility_check(other)
        if not array_equal(self.sensors, other.sensors):
            raise ValueError("Sensors mismatch")
        if not array_equal(self.latencies, other.latencies):
            raise ValueError("Latencies mismatch")
        # constructor expects a sequence of function names and sequences of 2d matrices
        return SensorExpressionSet(
            functions=self.functions + other.functions,
            sensors=self.sensors,
            latencies=self.latencies,
            data=concat([self.scalp, other.scalp],
                        dim=DIM_FUNCTION,
                        data_vars="all",  # Required by concat of DataArrays
                        ).data,
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
            data=[
                self._data[BLOCK_SCALP].sel({DIM_FUNCTION: function}).data
                for function in functions
            ],
        )

    def best_functions(self) -> DataFrame:
        """
        Return a DataFrame containing:
        for each sensor, the best function and latency for that sensor, and the associated log p-value

        Note that channels for which the best p-value is 1 will be omitted.
        """
        return super()._best_functions_for_block(BLOCK_SCALP)


T_ExpressionSetSubclass = TypeVar("T_ExpressionSetSubclass", bound=ExpressionSet)


def combine(expression_sets: Sequence[T_ExpressionSetSubclass]) -> T_ExpressionSetSubclass:
    """
    Combines a sequence of `ExpressionSet`s into a single `ExpressionSet`.
    All must be suitable for combination, e.g. same type, same channels, etc.
    """
    if len(expression_sets) == 0:
        raise ValueError("Cannot combine empty collection of ExpressionSets")
    if len(expression_sets) == 1:
        return expression_sets[0]
    return expression_sets[0] + combine(expression_sets[1:])
