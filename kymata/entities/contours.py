from collections import OrderedDict
from math import floor
from typing import Mapping, Optional, Iterator

from numpy import squeeze, float32, floor as np_floor, arange, zeros, zeros_like
from numpy.typing import NDArray


class Contour:
    """
    A class representing a single transform contour with a name, values, and a sample rate.

    Attributes:
        name (str): The name of the transform.
        values (NDArray): The sampled values of the transform.
        sample_rate_hz (float): The sample rate in Hertz.
    """
    def __init__(self, name: str, values: NDArray, sample_rate_hz: Optional[float] = None):
        self.name = name
        self.values = _normalise_contour_format(values)
        self.sample_rate_hz = sample_rate_hz if sample_rate_hz is not None else 1000

    def resampled(self, rate_hz: float):
        """
        Resamples the transform's values to a new sample rate, using the most recent sampling method.

        Args:
            rate_hz (float): The new sample rate in Hertz.

        Returns:
            Contour: A new Transform instance with resampled values and the specified sample rate.
        """
        ratio = self.sample_rate_hz / rate_hz
        if ratio > 1:  # Downsampling
            if ratio.is_integer():
                # Take every n samples
                resampled_values = self.values[::int(ratio)]
            else:
                # Use most recent sampling
                idxs = np_floor(arange(0, len(self.values), ratio)).astype(int)
                resampled_values = self.values[idxs]

        else:  # Upsampling
            new_length = floor(len(self.values) / ratio)

            resampled_values = zeros(new_length)
            for i in range(new_length):
                # Use most recent sampling
                resampled_values[i] = self.values[floor(i * ratio)]

        return Contour(
            name=self.name,
            values=resampled_values,
            sample_rate_hz=rate_hz)

    @property
    def time_step(self) -> float:
        """
        Computes the time step between samples based on the sample rate.

        Returns:
            float: The time step in seconds.
        """
        return 1 / self.sample_rate_hz

    @property
    def duration_samples(self) -> int:
        """Duration of the contour in samples."""
        return len(self)

    @property
    def duration_seconds(self) -> float:
        """Duration of the contour in seconds"""
        return self.duration_samples / self.sample_rate_hz

    def __len__(self) -> int:
        return len(self.values)


class ContourSet(Mapping):
    """
    Represents a collection of transform contours with associated metadata.
    """

    def __init__(
            self,
            contour_dict: dict[str, NDArray],
            sample_rate_hz: Optional[float] = None,
            description: Optional[str] = None,
            author: Optional[str] = None,
    ):
        if sample_rate_hz is None:
            sample_rate_hz = 1000

        # Validate and store contour dict
        self._contour_dict: OrderedDict[str, NDArray] = OrderedDict()
        validation_length = None
        for contour_name, contour_array in contour_dict.items():
            contour: NDArray = _normalise_contour_format(contour_array)

            if contour.ndim > 1:
                raise ValueError(f"Only supports vectors; supplied contour {contour_name} had shape {contour_array.shape}")
            if validation_length is None:
                validation_length = len(contour)
            elif validation_length != len(contour):
                raise ValueError(f"Unequal lengths: contour {contour_name} had length {len(contour)}, others had length {validation_length}")

            self._contour_dict[contour_name] = contour

        self.sample_rate_hz: float = sample_rate_hz
        self.description: str | None = description
        self.author: str | None = author

    @property
    def names(self) -> list[str]:
        """The transform names."""
        return list(self._contour_dict.keys())

    @property
    def duration_samples(self) -> int:
        """Duration of the contour in samples."""
        return len(self._contour_dict[self.names[0]])

    @property
    def duration_seconds(self) -> float:
        """Duration of the contour in seconds."""
        return self.duration_samples / self.sample_rate_hz

    def __getitem__(self, item: str) -> Contour:
        if item not in self._contour_dict:
            raise KeyError(item)
        return Contour(name=item, values=self._contour_dict[item], sample_rate_hz=self.sample_rate_hz)

    def __len__(self):
        return len(self._contour_dict)

    def __iter__(self) -> Iterator:
        return iter(self._contour_dict)

    def __str__(self) -> str:
        sep = ",\n    "
        return (f"Contours({self.description}): [\n"
                f"    {sep.join(self.names)}"
                f") duration {self.duration_seconds:.2f} seconds")

    def add_contour(self, transform_name: str, transform_contour: NDArray) -> None:
        """Adds a contour with the specified name to the list."""
        transform_contour = _normalise_contour_format(transform_contour)
        if len(transform_contour) != self.duration_samples:
            raise ValueError(f"Couldn't add contour of duration {len(transform_contour)} samples to contours with duration {self.duration_samples}")
        if transform_name in self._contour_dict.keys():
            raise ValueError(f"Contour named {transform_name} already exists")
        self._contour_dict[transform_name] = transform_contour

    def remove_contour(self, transform_name: str):
        """Removes the contour named `name`."""
        if transform_name not in self._contour_dict:
            raise KeyError(transform_name)
        del self._contour_dict[transform_name]


def _normalise_contour_format(contour: NDArray) -> NDArray:
    """Shared format normalisation function to ensure low-precision vectors."""
    return squeeze(contour).astype(float32)


def shift_by(data: NDArray, samples: int) -> NDArray:
    """
    Shift signal in array by a number of samples.

    Args:
        data:
        samples:
    Returns:

    """
    if samples == 0:
        return data

    shifted = zeros_like(data)

    if abs(samples) >= len(data):
        return shifted

    if samples > 0:
        shifted[samples:] = data[:-samples]
    if samples < 0:
        shifted[:len(data) + samples] = data[-samples:]

    return shifted
