"""
Classes and functions for storing expression information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self
from typing import Sequence, Union, get_args, TypeVar, Collection
from warnings import warn

from numpy import (
    # Can't use NDArray for isinstance checks
    ndarray,
    array, array_equal, where, inf, argmax, all as np_all)
from numpy.typing import NDArray
from sparse import SparseArray, COO
from xarray import DataArray, concat

from kymata.entities.constants import HEMI_LEFT, HEMI_RIGHT
from kymata.entities.datatypes import (
    HexelDType, SensorDType, LatencyDType, TransformNameDType, Hexel, Sensor, Latency, Channel)
from kymata.entities.iterables import all_equal
from kymata.entities.sparse_data import (
    expand_dims, densify_data_block, sparsify_log_pmatrix)

_InputDataArray = Union[ndarray, SparseArray]  # Type alias for data which can be accepted

# Data dimension labels
DIM_HEXEL = "hexel"
DIM_SENSOR = "sensor"
DIM_LATENCY = "latency"
DIM_TRANSFORM = "transform"

# Block (e.g. hemisphere)
BLOCK_LEFT = HEMI_LEFT
BLOCK_RIGHT = HEMI_RIGHT
BLOCK_SCALP = "scalp"


@dataclass(frozen=True)
class ExpressionPoint:
    """
    A single point of transform expression evidence.
    """
    channel: Channel
    latency: Latency
    transform: str
    logp_value: float


class ExpressionSet(ABC):
    """
    Brain data associated with expression of a single transform.
    Data is log10 p-values.
    """

    def __init__(
            self,
            transforms: str | Sequence[str],
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
            transforms (str | Sequence[str]): Transform name, or sequence of names.
            latencies (Sequence[Latency]): Latency values.
            data_blocks (dict[str, _InputDataArray | Sequence[_InputDataArray]]):
                Mapping of block names to data arrays (log10 p-values).

                In general there are two possible formats for this argument.

                In the first (safer, more explicit and flexible) format, `data_blocks` contains a dict mapping block
                names to data arrays. E.g., in the case there are three transforms in a hexel setting:
                ```
                    {
                                  # each array is (channel, latency)-shaped
                                  # and there's one for each transform
                                  # ↓
                        "left":  [array(...), array(...), array(...)],
                        "right": [array(...), array(...), array(...)],
                    }
                ```
                or in a sensor setting:
                ```
                    {
                        "scalp": [array(...), array(...), array(...)],
                    }
                ```
                (and where `array(...)` can be a numpy array or a sparse array).
                In this format, all data arrays should be the same size.

                In the second (more performant) format, `data_blocks` contains a single data array whose `transform`
                dimensions can be concatenated to achieve the desired resultant data block. E.g.
                ```
                    {
                                  # each array is (channel, latency, transform)-shaped
                                  # ↓
                        "left":  array(...),
                        "right": array(...),
                    }
                ```
            channel_coord_name (str): Name of the channel coordinate.
            channel_coord_dtype: Data type of the channel coordinate.
            channel_coord_values (dict[str, Sequence]): Dictionary mapping block names to channel coordinate values.

        Raises:
            ValueError: when arguments are invalid
        """

        # Canonical order of dimensions
        self._dims = (channel_coord_name, DIM_LATENCY, DIM_TRANSFORM)

        self._block_names: list[str] = list(data_blocks.keys())
        self.channel_coord_name = channel_coord_name

        #########################################
        # region Validate and normalise arguments
        #########################################

        if set(self._block_names) != set(channel_coord_values.keys()):
            raise ValueError("Ensure data block names match channel block names")

        data_supplied_as_sequence = self._validate_data_supplied_as_sequence(data_blocks)

        # If only one transform supplied, ensure that the other arguments are appropriately specified
        if isinstance(transforms, str):
            if data_supplied_as_sequence:
                raise ValueError("Single transform name provided but data came as a sequence")
            # Only one transform, so no matter what format the data is in, we don't expect a sequence
            # Wrap transform names and data blocks to ensure we have sequences from now on
            transforms = [transforms]

        self._validate_transforms_no_duplicates(transforms)

        # If data was specified as sequence, ensure all arrays in the sequence have the same shape
        if data_supplied_as_sequence:
            if not all_equal([
                # Only check latencies, as channels can vary between blocks (e.g. different hexels per hemi) and there
                # is only one transform per item in the sequence
                block_data_array.shape[1]
                for block_data in data_blocks.values()
                for block_data_array in block_data  # data format sequence
            ]):
                raise ValueError("Not all input data blocks have the same shape.")
        # Otherwise ensure all blocks have the same size
        else:
            if not all_equal([
                # Only check latencies, as channels can vary between blocks (e.g. different hexels per hemi) and there
                # is only one transform per item in the sequence
                block_data_array.shape[1]
                for block_data_array in data_blocks.values()
            ]):
                raise ValueError("Not all input data blocks have the same shape")

        # Additional validity checks may happen during construction

        # Channels can vary between blocks (e.g. different numbers of vertices for each hemisphere).
        # But latencies and transforms do not
        channels: dict[str, NDArray] = {
            bn: array(channel_coord_values[bn], dtype=channel_coord_dtype)
            for bn in self._block_names
        }
        latencies = array(latencies, dtype=LatencyDType)
        transforms = array(transforms, dtype=TransformNameDType)

        for block_name, block_data in data_blocks.items():
            if data_supplied_as_sequence:
                for trans_name, data in zip(transforms, block_data):
                    if len(channels[block_name]) != data.shape[0]:
                        raise ValueError(f"{channel_coord_name} mismatch for {trans_name}: "
                                         f"{len(channels)} {channel_coord_name} versus data shape {data.shape} "
                                         f"({block_name})")
                    if len(latencies) != data.shape[1]:
                        raise ValueError(f"Latencies mismatch for {trans_name}: "
                                         f"{len(latencies)} latencies versus data shape {data.shape} "
                                         f"({block_name})")
                    if not len(transforms) == len(block_data):
                        raise ValueError(f"Transform mismatch for {block_name}: "
                                         f"{len(transforms)} transforms but only {len(block_data)} data blocks")
                    if len(data.shape) > 3:
                        raise ValueError(f"Too many dimensions in data for {trans_name}: "
                                         f"shape={data.shape} "
                                         f"({block_name})")
                    if len(data.shape) == 3 and data.shape[2] != 1:
                        raise ValueError(f"Too many dimensions in data for {trans_name}: "
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
                if len(block_data.shape) == 3 and len(transforms) != block_data.shape[2]:
                    raise ValueError(f"Transforms mismatch: "
                                     f"{len(transforms)} transforms versus data shape {block_data.shape} "
                                     f"({block_name})")

        ############################################
        # endregion Validate and normalise arguments
        ############################################

        # Input value `data_blocks` has type something like dict[str, list[array]].
        #  i.e. a dict mapping block names to a list of 2d data volumes, one for each transform
        # We need to eventually get it into `self._data`, which has type dict[str, DataArray]
        # i.e. a dict mapping block names to a DataArray containing data for all transforms
        self._data: dict[str, DataArray] = dict()
        nan_warning_sent = False
        for block_name, block_data in data_blocks.items():
            if data_supplied_as_sequence:
                # Build DataArray by concatenating sequence
                data_array: DataArray = _concat_dataarrays(
                    [
                        DataArray(
                            self._init_prep_data(transform_data),
                            coords={
                                channel_coord_name: channels[block_name],
                                DIM_LATENCY: latencies,
                                DIM_TRANSFORM: [transform],
                            },
                        )
                        for transform, transform_data in zip(transforms, block_data)
                    ])
            else:
                # Build DataArray in one go
                data_array: DataArray = DataArray(
                    self._init_prep_data(block_data),
                    coords={
                        channel_coord_name: channels[block_name],
                        DIM_LATENCY: latencies,
                        DIM_TRANSFORM: transforms,
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
            if not array_equal(data_array.coords[DIM_TRANSFORM].values, transforms):
                raise ValueError("DataArray had wrong transforms")

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
    def _validate_transforms_no_duplicates(self, transforms: Sequence[str]) -> None:
        if not len(transforms) == len(set(transforms)):
            checked = []
            for f in transforms:
                if f in checked:
                    break
                checked.append(f)
            raise ValueError(f"Duplicated transforms in input, e.g. {f}")

    @classmethod
    def _init_prep_data(cls, data: _InputDataArray) -> COO:
        # Convert to sparse matrix
        if isinstance(data, ndarray):
            data = sparsify_log_pmatrix(data)
        elif not isinstance(data, SparseArray):
            raise NotImplementedError()

        if data.ndim < 3:
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
    def transforms(self) -> list[TransformNameDType]:
        """Transform names."""
        transforms = {
            bn: data.coords[DIM_TRANSFORM].values.tolist()
            for bn, data in self._data.items()
        }
        # Validate that transforms are the same for all data blocks
        assert all_equal(list(transforms.values()))
        # Then just return the first one
        return transforms[self._block_names[0]]

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
    def __getitem__(self, transforms: str | Collection[str]) -> ExpressionSet:
        pass

    def _validate_crop_latency_args(self, start: float, stop: float) -> None:
        """
        Raises IndexError if the start and stop indices are invalid.
        """
        if start >= stop:
            raise IndexError(f"start must be less than stop ({start=}, {stop=})")
        if start > self.latencies.max() or stop < self.latencies.min():
            raise IndexError(f"Crop range lies entirely outside expression data"
                             f" ({start}–{stop} is outside {self.latencies.min()}–{self.latencies.max()})")

        selected_latencies = [lat for lat in self.latencies if start <= lat <= stop]

        if len(selected_latencies) == 0:
            raise IndexError(f"No latencies fell between selected range ({start=}, {stop=})")

    @abstractmethod
    def crop(self, latency_start: float | None, latency_stop: float | None) -> Self:
        """
        Returns a copy of the ExpressionSet with latencies cropped between the two endpoints (inclusive).

        Args:
            latency_start (float | None): Latency in seconds to start the cropped window. Use None for no cropping at
                the start (e.g. half-open crop).
            latency_stop (float | None): Latency in seconds to stop the cropped window. Use None for no cropping at the
                end (e.g. half-open crop).

        Returns:
            Self: A copy of the ExpressionSet with the latencies cropped between the specified start and stop.
        """
        pass

    @abstractmethod
    def __copy__(self) -> Self:
        pass

    @abstractmethod
    def __add__(self, other) -> Self:
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
        if not self.transforms == other.transforms:
            return False
        if not array_equal(self.latencies, other.latencies):
            return False
        for bn in self._block_names:
            if not array_equal(self._channels[bn], other._channels[bn]):
                return False
        return True

    def _best_transforms_for_block(self, block_name: str) -> list[ExpressionPoint]:
        """
        Return a list of expression points:
        for each channel in the block, the best transform and latency for that channel, and the associated log p-value

        Note that channels for which the best p-value is 1 will be omitted.
        """
        # Want, for each channel in the block:
        #  - The name, f, of the transform which is best at any latency
        #  - The latency, l, for which f is best
        #  - The log p-value, p, for f at l

        # sparse.COO doesn't implement argmin, so we have to do it in a few steps

        data: DataArray = self._data[block_name].copy()
        densify_data_block(data)

        # (channel, transform) → l, the best latency
        best_latency = data.idxmin(dim=DIM_LATENCY)
        # (channel, transform) → log p of best latency for each transform
        logp_at_best_latency = data.min(dim=DIM_LATENCY)

        # (channel) → log p of best transform (at best latency)
        logp_at_best_transform = logp_at_best_latency.min(dim=DIM_TRANSFORM)
        # (channel) → f, the best transform
        best_transform = logp_at_best_latency.idxmin(dim=DIM_TRANSFORM)

        # (channel) -> logp of the best transform (at the best latency)
        logp_vals = logp_at_best_transform.data
        # (channel) -> best transform name (at the best latency)
        best_transforms = best_transform.data
        # (channel) -> latency of the best transform
        best_latencies = best_latency.sel(
            {
                # e.g. hexels          -> array([0, ..., 10241])
                self.channel_coord_name: self._channels[block_name],
                #          -> DataArray((hexel) -> transform)
                DIM_TRANSFORM: best_transform,
            }
        ).data

        # Cut out channels which have a best log p-val of 0 (i.e. p = 1)
        idxs = where(logp_vals < 0)[0]

        return [
            ExpressionPoint(channel=self._channels[block_name][idx],
                            transform=best_transforms[idx],
                            latency=best_latencies[idx],
                            logp_value=logp_vals[idx])
            for idx in idxs
        ]

    def rename(self, transforms: dict[str, str] = None, channels: dict = None) -> None:
        """
        Renames the transforms and channels within an ExpressionSet.

        Supply a dictionary mapping old values to new values.

        Raises KeyError if one of the keys in the renaming dictionary is not a transform name in the expression set.
        """
        # Default values
        if transforms is None:
            transforms = dict()
        if channels is None:
            channels = dict()

        # Validate
        for old, new in transforms.items():
            if old not in self.transforms:
                raise KeyError(f"{old} is not a transform in this expression set")
        for old, new in channels.items():
            for bn in self._block_names:
                if old not in self._channels[bn]:
                    raise KeyError(
                        f"{old} is not a {bn} {self.channel_coord_name} in this expression set"
                    )

        # Replace
        for bn, data in self._data.items():
            # Transforms
            new_names = []
            for old_name in self._data[bn][DIM_TRANSFORM].values:
                if old_name in transforms:
                    new_names.append(transforms[old_name])
                else:
                    new_names.append(old_name)
            self._data[bn][DIM_TRANSFORM] = new_names

            # Channels
            new_channels = []
            for old_channel in self._data[bn][self.channel_coord_name].values:
                if old_channel in channels:
                    new_channels.append(channels[old_channel])
                else:
                    new_channels.append(old_channel)
            self._data[bn][self.channel_coord_name] = new_channels

    def _clear_point(self,
                     channel: Channel, latency: Latency,
                     block_name: str):
        if channel not in self._channels[block_name]:
            raise ValueError(f"No {self.channel_coord_name} {channel}")
        if latency not in self.latencies:
            raise ValueError(f"No latency {latency}"
                             + (". Check for floating-point issues?" if not isinstance(latency, LatencyDType) else ""))
        if block_name not in self._block_names:
            raise ValueError(f"Invalid block name {block_name}")

        # Get the coordinates of the value to change in the data array
        channel_i = argmax(self._channels[block_name] == channel)
        latency_i = argmax(self.latencies == latency)
        for transform_i, transform in enumerate(self.transforms):
            coords = array([channel_i, latency_i, transform_i])

            # Get the linear index in the sparse array
            coords_idx = where(np_all(self._data[block_name].data.coords == coords[:, None], axis=0))[0]

            # Clear it if it's there
            if coords_idx:
                self._data[block_name].data.data[coords_idx] = 0

    @abstractmethod
    def best_transforms(self) -> list[ExpressionPoint] | tuple[list[ExpressionPoint], ...]:
        """
        Note that channels for which the best p-value is 1 will be omitted.
        """
        pass


class HexelExpressionSet(ExpressionSet):
    """
    Brain data associated with expression of a single transform in hexel space.
    Includes lh, rh, flipped, non-flipped.
    Data is log10 p-values
    """

    def __init__(
        self,
        transforms: str | Sequence[str],
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
            transforms=transforms,
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

    def __getitem__(self, transforms: str | Collection[str]) -> HexelExpressionSet:
        """
        Select data for specified transform(s) only.
        Use a transform name or collection of transform names
        """
        # Allow indexing by a single transform
        if isinstance(transforms, str):
            transforms = [transforms]
        else:
            # Convert collection to list
            transforms = list(transforms)
        # Get indices of sliced transforms within total transform list
        transform_idxs = []
        for f in transforms:
            try:
                transform_idxs.append(self.transforms.index(f))
            except ValueError:
                raise KeyError(f)
        return HexelExpressionSet(
            transforms=transforms,
            hexels_lh=self.hexels_left,
            hexels_rh=self.hexels_right,
            latencies=self.latencies,
            data_lh=self._data[BLOCK_LEFT].data[:, :, transform_idxs],
            data_rh=self._data[BLOCK_RIGHT].data[:, :, transform_idxs],
        )

    def crop(self, latency_start: float | None, latency_stop: float | None) -> HexelExpressionSet:
        if latency_start is None:
            latency_start = -inf
        if latency_stop is None:
            latency_stop = inf
        self._validate_crop_latency_args(latency_start, latency_stop)

        selected_latencies = [
            (i, lat)
            for i, lat in enumerate(self.latencies)
            if latency_start <= lat <= latency_stop
        ]
        # Unzip idxs and latencies
        new_latency_idxs, selected_latencies = zip(*selected_latencies)

        return HexelExpressionSet(
            transforms=self.transforms.copy(),
            hexels_lh=self.hexels_left.copy(),
            hexels_rh=self.hexels_right.copy(),
            latencies=selected_latencies,
            data_lh=self._data[BLOCK_LEFT].isel({DIM_LATENCY: array(new_latency_idxs)}).data.copy(),
            data_rh=self._data[BLOCK_RIGHT].isel({DIM_LATENCY: array(new_latency_idxs)}).data.copy(),
        )
    
    def __copy__(self):
        return HexelExpressionSet(
            transforms=self.transforms.copy(),
            hexels_lh=self.hexels_left.copy(),
            hexels_rh=self.hexels_right.copy(),
            latencies=self.latencies.copy(),
            # Slice by transform
            data_lh=self._data[BLOCK_LEFT].data.copy(),
            data_rh=self._data[BLOCK_RIGHT].data.copy(),
        )

    def __add__(self, other: HexelExpressionSet) -> HexelExpressionSet:
        self._add_compatibility_check(other)
        if not array_equal(self.hexels_left, other.hexels_left):
            raise ValueError("Hexels mismatch (left)")
        if not array_equal(self.hexels_right, other.hexels_right):
            raise ValueError("Hexels mismatch (right)")
        assert array_equal(self.latencies, other.latencies), "Latencies mismatch"
        return HexelExpressionSet(
            transforms=self.transforms + other.transforms,
            hexels_lh=self.hexels_left,
            hexels_rh=self.hexels_right,
            latencies=self.latencies,
            data_lh=_concat_dataarrays([self.left, other.left]).data,
            data_rh=_concat_dataarrays([self.right, other.right]).data,
        )

    def __eq__(self, other: HexelExpressionSet) -> bool:
        if not super().__eq__(other):
            return False
        if not COO(self.left.data == other.left.data).all():
            return False
        if not COO(self.right.data == other.right.data).all():
            return False
        return True

    def clear_point_left(self, hexel: Hexel, latency: Latency):
        """
        Clears a datapoint in the left hemisphere at the specified hexel and latency.
        """
        self._clear_point(hexel, latency, BLOCK_LEFT)

    def clear_point_right(self, hexel: Hexel, latency: Latency):
        """
        Clears a datapoint in the right hemisphere at the specified hexel and latency.
        """
        self._clear_point(hexel, latency, BLOCK_RIGHT)

    def best_transforms(self) -> tuple[list[ExpressionPoint], list[ExpressionPoint]]:
        """
        Return a pair of DataFrames (left, right), containing:
        for each hexel, the best transform and latency for that hexel, and the associated log p-value

        Note that channels for which the best p-value is 1 will be omitted.
        """
        return (
            super()._best_transforms_for_block(BLOCK_LEFT),
            super()._best_transforms_for_block(BLOCK_RIGHT),
        )


class SensorExpressionSet(ExpressionSet):
    """
    Brain data associated with the expression of a single transform in sensor space.
    Includes left hemisphere (lh), right hemisphere (rh), flipped, and non-flipped data.
    Data is represented as log10 p-values.
    """

    def __init__(
        self,
        transforms: str | Sequence[str],
        # Metadata
        sensors: Sequence[Sensor],
        latencies: Sequence[Latency],
        # log p-values
        # In general, we will combine flipped and non-flipped versions
        data: _InputDataArray | Sequence[_InputDataArray],
    ):
        """
        Initialize the SensorExpressionSet with transform names, sensor metadata, latency information, and log p-value data.

        Args:
            transforms (str | Sequence[str]): The names of the transforms being evaluated.
            sensors (Sequence[Sensor]): Metadata about the sensors used in the study.
            latencies (Sequence[Latency]): Latency information corresponding to the data.
            data (_InputDataArray | Sequence[_InputDataArray]): Log p-values representing the data.
        """
        super().__init__(
            transforms=transforms,
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
        return SensorExpressionSet(
            transforms=self.transforms.copy(),
            sensors=self.sensors.copy(),
            latencies=self.latencies.copy(),
            data=self._data[BLOCK_SCALP].data.copy(),
        )

    def __add__(self, other: SensorExpressionSet) -> SensorExpressionSet:
        self._add_compatibility_check(other)
        if not array_equal(self.sensors, other.sensors):
            raise ValueError("Sensors mismatch")
        if not array_equal(self.latencies, other.latencies):
            raise ValueError("Latencies mismatch")
        # constructor expects a sequence of transform names and sequences of 2d matrices
        return SensorExpressionSet(
            transforms=self.transforms + other.transforms,
            sensors=self.sensors,
            latencies=self.latencies,
            data=_concat_dataarrays([self.scalp, other.scalp]).data,
        )

    def __getitem__(self, transforms: str | Collection[str]) -> SensorExpressionSet:
        """
        Select data for specified transform(s) only.
        Use a transform name or collection of transform names
        """
        # Allow indexing by a single transform
        if isinstance(transforms, str):
            transforms = [transforms]
        else:
            # Convert collection to list
            transforms = list(transforms)
        # Get indices of sliced transforms within total transform list
        transform_idxs = []
        for t in transforms:
            try:
                transform_idxs.append(self.transforms.index(t))
            except ValueError:
                raise KeyError(t)
        return SensorExpressionSet(
            transforms=transforms,
            sensors=self.sensors,
            latencies=self.latencies,
            # Slice data by requested transforms
            data=self._data[BLOCK_SCALP].data[:, :, transform_idxs],
        )

    def crop(self, latency_start: float | None, latency_stop: float | None) -> SensorExpressionSet:
        if latency_start is None:
            latency_start = -inf
        if latency_stop is None:
            latency_stop = inf
        self._validate_crop_latency_args(latency_start, latency_stop)

        selected_latencies = [
            (i, lat)
            for i, lat in enumerate(self.latencies)
            if latency_start <= lat <= latency_stop
        ]
        # Unzip idxs and latencies
        new_latency_idxs, selected_latencies = zip(*selected_latencies)

        return SensorExpressionSet(
            transforms=self.transforms.copy(),
            sensors=self.sensors.copy(),
            latencies=selected_latencies,
            data=self._data[BLOCK_SCALP].isel({DIM_LATENCY: array(new_latency_idxs)}).data.copy(),
        )

    def clear_point(self, sensor: Sensor, latency: Latency):
        """
        Clears a datapoint at the specified sensor and latency.
        """
        self._clear_point(sensor, latency, BLOCK_SCALP)

    def best_transforms(self) -> list[ExpressionPoint]:
        """
        Return a DataFrame containing:
        for each sensor, the best transform and latency for that sensor, and the associated log p-value

        Note that channels for which the best p-value is 1 will be omitted.
        """
        return super()._best_transforms_for_block(BLOCK_SCALP)


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


def _concat_dataarrays(arrays: Sequence[DataArray]) -> DataArray:
    return concat(
        arrays,
        dim=DIM_TRANSFORM,
        data_vars="all",  # Required by concat of DataArrays
    )


def get_n_channels(es: ExpressionSet) -> int:
    if isinstance(es, SensorExpressionSet):
        return len(es.sensors)
    if isinstance(es, HexelExpressionSet):
        return len(es.hexels_left) + len(es.hexels_right)
    raise NotImplementedError()
