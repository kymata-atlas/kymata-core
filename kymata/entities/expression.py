"""
Classes and transforms for storing expression information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Union, get_args, Tuple, TypeVar, Self
from warnings import warn

from numpy import (
    # Can't use NDArray for isinstance checks
    ndarray,
    array, array_equal, inf)
from numpy.typing import NDArray
from pandas import DataFrame
from sparse import SparseArray, COO
from xarray import DataArray, concat

import numpy as np

from kymata.entities.datatypes import HexelDType, SensorDType, LatencyDType, TransformNameDType, Hexel, Sensor, \
    Latency
from kymata.entities.iterables import all_equal
from kymata.entities.sparse_data import expand_dims, densify_data_block, sparsify_log_pmatrix

_InputDataArray = Union[ndarray, SparseArray]  # Type alias for data which can be accepted

# Data dimension labels
DIM_HEXEL = "hexel"
DIM_SENSOR = "sensor"
DIM_LATENCY = "latency"
DIM_TRANSFORM = "transform"

# Block (e.g. hemisphere)
BLOCK_LEFT  = "left"
BLOCK_RIGHT = "right"
BLOCK_SCALP = "scalp"


class ExpressionSet(ABC):
    """
    Brain data associated with expression of a single transform.
    Data is log10 p-values
    """

    def __init__(self,
                 transforms: str | Sequence[str] | Sequence[Sequence[str]],
                 # Metadata
                 latencies: Sequence[Latency],
                 # In general, we will combine flipped and non-flipped versions
                 # data_blocks contains a dict mapping block names to data arrays
                 # e.g., in the case there are three transforms:
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
        self._dims = (channel_coord_name, DIM_LATENCY, DIM_TRANSFORM)

        self._block_names: list[str] = list(data_blocks.keys())
        self.channel_coord_name = channel_coord_name

        # Validate arguments
        assert set(self._block_names) == set(channel_coord_values.keys()), "Ensure data block names match channel block names"
        _length_mismatch_message = ("Argument length mismatch, please supply one transform name and accompanying data, "
                                    "or equal-length sequences of the same.")
        if isinstance(transforms, str):
            # If only one transform
            for data in data_blocks.values():
                # Data should not be a sequence
                assert isinstance(data, get_args(_InputDataArray)), _length_mismatch_message
            # Wrap into sequences
            transforms = [transforms]
            for bn in self._block_names:
                data_blocks[bn] = [data_blocks[bn]]

        if isinstance(transforms[0], str):
            self._validate_transforms_no_duplicates(transforms)
        else:
            self._validate_transforms_no_duplicates(sum(transforms, []))

        assert all_equal([arr.shape[1] for arrs in data_blocks.values() for arr in arrs]), "Not all input data blocks have the same"

        # Channels can vary between blocks (e.g. different numbers of vertices for each hemisphere).
        # But latencies and transforms do not
        channels: dict[str, NDArray] = {
            bn: array(channel_coord_values[bn], dtype=channel_coord_dtype)
            for bn in self._block_names
        }
        latencies = array(latencies, dtype=LatencyDType)
        if isinstance(transforms[0], str):
            transforms = array(transforms, dtype=TransformNameDType)
        else:
            transforms = [array(transform_block, dtype=TransformNameDType) for transform_block in transforms]

        # Input value `data_blocks` has type something like dict[str, list[array]].
        #  i.e. a dict mapping block names to a list of 2d data volumes, one for each transform
        # We need to eventually get it into `self._data`, which has type dict[str, DataArray]
        # i.e. a dict mapping block names to a DataArray containing data for all transforms
        self._data: dict[str, DataArray] = dict()
        nan_warning_sent = False
        for data in data_blocks.values():
            if isinstance(data, list) and data[0].ndim == 2:
                for block_name, data_for_transforms in data_blocks.items():
                    for transform_name, data in zip(transforms, data_for_transforms):
                        assert len(channels[block_name]) == data.shape[0], f"{channel_coord_name} mismatch for {transform_name}: {len(channels)} {channel_coord_name} versus data shape {data.shape} ({block_name})"
                        assert len(latencies) == data.shape[1], f"Latencies mismatch for {transform_name}: {len(latencies)} latencies versus data shape {data.shape}"
                    data_array: DataArray = concat(
                        (
                            DataArray(self._init_prep_data(d),
                                    coords={
                                        channel_coord_name: channels[block_name],
                                        DIM_LATENCY: latencies,
                                        DIM_TRANSFORM: [transform]
                                    })
                            for transform, d in zip(transforms, data_for_transforms)
                        ),
                        dim=DIM_TRANSFORM,
                        data_vars="all",  # Required by concat of DataArrays
                        )

                    # Sometimes the data can contain nans, for example if the MEG hexel currents were set to nan on the medial
                    # wall. We can ignore these nans by setting the values to p=1, but because it's not typically expected we
                    # warn the user about it.
                    if data_array.isnull().any():
                        data_array = data_array.fillna(value=0)  # logp = 0 => p = 1
                        if not nan_warning_sent:  # Only want to send the warning once, even if there are multiple data blocks with nans.
                            warn("Supplied data contained nans. These will be replaced by p = 1 values.")
                            nan_warning_sent = True

                    assert data_array.dims == self._dims
                    assert set(data_array.coords.keys()) == set(self._dims)
                    assert array_equal(data_array.coords[DIM_TRANSFORM].values, transforms)

                    self._data[block_name] = data_array
            
            else:
                for block_name, data_for_transforms in data_blocks.items():
                    if not isinstance(data_for_transforms, list):
                        data_array = DataArray(data_for_transforms,
                                        coords={
                                            channel_coord_name: channels[block_name],
                                            DIM_LATENCY: latencies,
                                            DIM_TRANSFORM: transforms
                                        })
                    else:
                        data_array: DataArray = concat(
                            (
                                DataArray(d,
                                        coords={
                                            channel_coord_name: channels[block_name],
                                            DIM_LATENCY: latencies,
                                            DIM_TRANSFORM: transform
                                        })
                                for transform, d in zip(transforms, data_for_transforms)
                            ),
                            dim=DIM_TRANSFORM,
                            data_vars="all",  # Required by concat of DataArrays
                            )

                    # Sometimes the data can contain nans, for example if the MEG hexel currents were set to nan on the medial
                    # wall. We can ignore these nans by setting the values to p=1, but because it's not typically expected we
                    # warn the user about it.
                    if data_array.isnull().any():
                        data_array = data_array.fillna(value=0)  # logp = 0 => p = 1
                        if not nan_warning_sent:  # Only want to send the warning once, even if there are multiple data blocks with nans.
                            warn("Supplied data contained nans. These will be replaced by p = 1 values.")
                            nan_warning_sent = True

                    assert data_array.dims == self._dims
                    assert set(data_array.coords.keys()) == set(self._dims)
                    if isinstance(transforms, list):
                        assert array_equal(data_array.coords[DIM_TRANSFORM].values, np.concatenate(transforms))
                    else:
                        assert array_equal(data_array.coords[DIM_TRANSFORM].values, transforms)

                    self._data[block_name] = data_array
            

    def _validate_transforms_no_duplicates(self, transforms):
        if not len(transforms) == len(set(transforms)):
            checked = []
            for f in transforms:
                if f in checked:
                    break
                checked.append(f)
            raise ValueError(f"Duplicated transforms in input, e.g. {f}")

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
            bn: data.coords[self.channel_coord_name].values
            for bn, data in self._data.items()
        }

    @property
    def transforms(self) -> list[TransformNameDType]:
        """transform names."""
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
            bn: data.coords[DIM_LATENCY].values
            for bn, data in self._data.items()
        }
        # Validate that latencies are the same for all data blocks
        assert all_equal(list(latencies.values()))
        # Then just return the first one
        return latencies[self._block_names[0]]

    @abstractmethod
    def __getitem__(self, transforms: str | Sequence[str]) -> ExpressionSet:
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

    def _best_transforms_for_block(self, block_name: str) -> DataFrame:
        """
        Return a DataFrame containing:
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
        best_latencies = best_latency.sel({
            # e.g. hexels          -> array([0, ..., 10241])
            self.channel_coord_name: self._channels[block_name],
            #          -> DataArray((hexel) -> transform)
            DIM_TRANSFORM: best_transform
        }).data

        # Cut out channels which have a best log p-val of 0 (i.e. p = 1)
        idxs = logp_vals < 0

        return DataFrame.from_dict({
            self.channel_coord_name: self._channels[block_name][idxs],
            DIM_TRANSFORM: best_transforms[idxs],
            DIM_LATENCY: best_latencies[idxs],
            "value": logp_vals[idxs],
        })

    def rename(self, transforms: dict[str, str]) -> None:
        """
        Renames the transforms within an ExpressionSet.

        Supply a dictionary mapping old transform names to new transform names.

        Raises KeyError if one of the keys in the renaming dictionary is not a transform name in the expression set.
        """
        for old, new in transforms.items():
            if old not in self.transforms:
                raise KeyError(f"{old} is not a transform in this expression set")
        for bn, data in self._data.items():
            new_names = []
            for old_name in self._data[bn][DIM_TRANSFORM].values:
                if old_name in transforms:
                    new_names.append(transforms[old_name])
                else:
                    new_names.append(old_name)
            self._data[bn][DIM_TRANSFORM] = new_names

    @abstractmethod
    def best_transforms(self) -> DataFrame | tuple[DataFrame, ...]:
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

    def __init__(self,
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

    def __getitem__(self, transforms: str | Sequence[str]) -> HexelExpressionSet:
        """
        Select data for specified transform(s) only.
        Use a transform name or list/array of transform names
        """
        # Allow indexing by a single transform
        if isinstance(transforms, str):
            transforms = [transforms]
        for f in transforms:
            if f not in self.transforms:
                raise KeyError(f)
        return HexelExpressionSet(
            transforms=transforms,
            hexels_lh=self.hexels_left,
            hexels_rh=self.hexels_right,
            latencies=self.latencies,
            data_lh=[self._data[BLOCK_LEFT].sel({DIM_TRANSFORM: transform}).data for transform in transforms],
            data_rh=[self._data[BLOCK_RIGHT].sel({DIM_TRANSFORM: transform}).data for transform in transforms],
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
            data_lh=self._data[BLOCK_LEFT].values.copy(),
            data_rh=self._data[BLOCK_RIGHT].values.copy(),
        )

    def __add__(self, other: HexelExpressionSet) -> HexelExpressionSet:
        assert array_equal(self.hexels_left, other.hexels_left), "Hexels mismatch (left)"
        assert array_equal(self.hexels_right, other.hexels_right), "Hexels mismatch (right)"
        assert array_equal(self.latencies, other.latencies), "Latencies mismatch"
        # constructor expects a sequence of transform names and sequences of 2d matrices
        transforms = []
        data_lh = []
        data_rh = []
        for expr_set in [self, other]:
            for i, transform in enumerate(expr_set.transforms):
                transforms.append(transform)
                data_lh.append(expr_set._data[BLOCK_LEFT].data[:, :, i])
                data_rh.append(expr_set._data[BLOCK_RIGHT].data[:, :, i])
        return HexelExpressionSet(
            transforms=transforms,
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

    def best_transforms(self) -> Tuple[DataFrame, DataFrame]:
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

    def __init__(self,
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
            data_blocks={
                BLOCK_SCALP: data
            },
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
            data=self._data[BLOCK_SCALP].values.copy(),
        )

    def __add__(self, other: SensorExpressionSet) -> SensorExpressionSet:
        assert array_equal(self.sensors, other.sensors), "Sensors mismatch"
        assert array_equal(self.latencies, other.latencies), "Latencies mismatch"
        # constructor expects a sequence of transform names and sequences of 2d matrices
        return SensorExpressionSet(
            transforms=[self.transforms, other.transforms],
            sensors=self.sensors, 
            latencies=self.latencies,
            data=[self._data[BLOCK_SCALP].data, other._data[BLOCK_SCALP].data],
        )

    def __getitem__(self, transforms: str | Sequence[str]) -> SensorExpressionSet:
        """
        Select data for specified transform(s) only.
        Use a transform name or list/array of transform names
        """
        # Allow indexing by a single transform
        if isinstance(transforms, str):
            transforms = [transforms]
        for f in transforms:
            if f not in self.transforms:
                raise KeyError(f)
        return SensorExpressionSet(
            transforms=transforms,
            sensors=self.sensors,
            latencies=self.latencies,
            data=[self._data[BLOCK_SCALP].sel({DIM_TRANSFORM: transform}).data for transform in transforms],
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

    def best_transforms(self) -> DataFrame:
        """
        Return a DataFrame containing:
        for each sensor, the best transform and latency for that sensor, and the associated log p-value

        Note that channels for which the best p-value is 1 will be omitted.
        """
        return super()._best_transforms_for_block(BLOCK_SCALP)
