from dataclasses import dataclass
from math import floor

from numpy import floor as np_floor, arange, zeros
from numpy.typing import NDArray


@dataclass
class Function:
    """
    A class representing a function with a name, values, and a sample rate.

    Attributes:
        name (str): The name of the function.
        values (NDArray): The sampled values of the function.
        sample_rate (float): The sample rate in Hertz.
    """
    name: str
    values: NDArray
    sample_rate: float  # Hertz

    def resampled(self, rate_hz: float):
        """
        Resamples the function's values to a new sample rate, using the most recent sampling method.

        Args:
            rate_hz (float): The new sample rate in Hertz.

        Returns:
            Function: A new Function instance with resampled values and the specified sample rate.
        """
        ratio = self.sample_rate / rate_hz
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

        return Function(
            name=self.name,
            values=resampled_values,
            sample_rate=rate_hz)

    @property
    def time_step(self) -> float:
        """
        Computes the time step between samples based on the sample rate.

        Returns:
            float: The time step in seconds.
        """
        return 1 / self.sample_rate
