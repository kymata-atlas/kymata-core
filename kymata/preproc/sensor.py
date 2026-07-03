import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.signal import hilbert, resample


def fit_drift_delay(
        stimulus: NDArray,
        misc: NDArray,
        stimulus_sr: float,
        misc_sr: float,
        reference_drift: float = 0,
        reference_delay: float = 0,
        drift_range: float = 0.001,
        delay_range: float = 0.01,
        grid_n: int = 101,
) -> tuple[float, float]:
    """
    Compare the recorded stimulus (via the EMEG misc channel) and the original transform to automatically determine the
    delay and drift of the EMEG recording.

    Args:
        stimulus (NDArray): a (samples,)-sized array of the actual stimulus
        misc (NDArray): a (samples,)-sized array of the misc channel
        stimulus_sr (float): sample rate of the stimulus (Hz)
        misc_sr (float): sample rate of the misc channel (Hz)
        reference_drift (reference): a reference drift (s/s), usually taken from the config file
        reference_delay (reference): a reference delay (s), usually taken from the config file
        drift_range (float): allowed values of actual drift are reference ± drift_range (seconds).
        delay_range (float): allowed values of actual delay are reference ± delay_range (seconds).
        grid_n (int): Number of gridpoints to use in search.

    Returns:
        tuple[float, float}:
            [0]: The actual drift, relative to the reference (actual - reference).
            [1]: The actual delay, relative to the reference (actual - reference).
    """

    stimulus_envelope = _hilbert_envolope(stimulus)
    misc_envelope = _hilbert_envolope(misc)

    # Remove DC and normalise
    stimulus_envelope -= stimulus_envelope.mean()
    misc_envelope -= misc_envelope.mean()
    with np.errstate(divide="raise"):
        stimulus_envelope /= norm(stimulus_envelope)
        misc_envelope /= norm(misc_envelope)

    # Resample to same sample rate
    if misc_sr > stimulus_sr:
        raise ValueError(f"Highly unexpected for misc sample rate ({misc_sr}) to be higher than stimulus sample rate ({stimulus_sr})")
    if misc_sr < stimulus_sr:
        resampled_n = round(len(stimulus_envelope) * misc_sr / stimulus_sr)
        stimulus_envelope = resample(stimulus_envelope, resampled_n)
        stimulus_sr = misc_sr

    candidate_drifts = np.linspace(start=reference_drift - drift_range, stop=reference_drift + drift_range, num=grid_n)
    candidate_delays = np.linspace(start=reference_delay - delay_range, stop=reference_delay + delay_range, num=grid_n)

    # Grid search over drifts and delays
    best_score = -np.inf
    best_drift = reference_drift
    best_delay = reference_delay
    for drift in candidate_drifts:
        for delay in candidate_delays:

            stretched_stimulus_envelope = _stretch_signal(stimulus_envelope, sample_rate=stimulus_sr, stretch=1 + drift, delay=delay)

            # Score is correlation in overlapping region
            overlap_length = min(len(stretched_stimulus_envelope), len(misc_envelope))
            score = np.corrcoef(
                stretched_stimulus_envelope[:overlap_length],
                misc_envelope[:overlap_length],
            )[0][1]

            if score > best_score:
                best_score = score
                best_drift = drift
                best_delay = delay

    return (
        best_drift - reference_drift,
        best_delay - reference_delay,
    )


def _hilbert_envolope(signal: NDArray) -> NDArray:
    return np.abs(hilbert(np.asarray(signal, dtype=float)))


def _stretch_signal(signal: NDArray, sample_rate: float, stretch: float, delay: float) -> NDArray:
    signal = np.asarray(signal, dtype=float)
    n_samples = len(signal)

    time_axis = np.arange(n_samples) / sample_rate
    time_axis_stretched = (time_axis - delay) / stretch

    interpolator = interp1d(time_axis, signal, kind="linear", bounds_error=False, fill_value=0.0, assume_sorted=True)

    stretched_signal = interpolator(time_axis_stretched)
    return stretched_signal
