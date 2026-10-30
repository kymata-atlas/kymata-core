from collections import OrderedDict
from typing import Mapping, Optional, Iterator

from numpy import squeeze, float32
from numpy.typing import NDArray


class Contours(Mapping):
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

    def __getitem__(self, item):
        return self._contour_dict[item]

    def __len__(self):
        return len(self._contour_dict)

    def __iter__(self) -> Iterator:
        return iter(self._contour_dict)

    def add_contour(self, transform_name: str, transform_contour: NDArray):
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
