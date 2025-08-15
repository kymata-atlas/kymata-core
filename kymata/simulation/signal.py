from numpy.typing import NDArray
from numpy.random import normal
from scipy.signal import butter, filtfilt


def bandpass_filter(data,
                    low_cutoff: float, high_cutoff: float,  fs: float,
                    order: int = 4,
                    ) -> NDArray:
    """
    Apply a bandpass filter to a 1d input signal.

    Args:
        data: 1d input signal
        low_cutoff (float): low cutoff frequency for the bandpass
        high_cutoff (float): high cutoff frequency for the bandpass
        fs (float): sampling frequency (Hz)
        order (int): The filter order

    Returns:
        NDArray: filtered data
    """
    if not 0 < low_cutoff < high_cutoff < fs / 2:
        raise ValueError(f"Require 0 < lowcut < highcut < fs / 2")
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def white_noise(n_samples: int, low: float = 0, high: float = 1):
    """White noise within specified freq limits"""
    return normal(low, high, n_samples)
