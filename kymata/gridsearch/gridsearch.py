import numpy as np
from numpy.typing import NDArray
from scipy import stats

from kymata.entities.functions import Function
from kymata.math.combinatorics import generate_derangement
from kymata.math.vector import normalize
from kymata.entities.expression import SensorExpressionSet


def do_gridsearch(
        emeg_values: NDArray,
        function: Function,
        sensor_names: list[str],
        start_latency: float,  # seconds
        emeg_t_start: float,
        emeg_sample_rate: int = 1000,  # Hertz
        # TODO: what are good default values?
        n_derangements: int = 20,
        seconds_per_split: float = 0.5,
        n_splits: int = 800,
        audio_shift_correction: float = 0.0005375,  # second/second  # TODO: describe in which direction?
        ) -> SensorExpressionSet:
    """
    Do the Kymata gridsearch over all hexels for all latencies.
    """

    # We'll need to downsample the EMEG to match the function's sample rate
    downsample_rate = emeg_sample_rate / function.sample_rate

    n_samples_per_split = int(
        (seconds_per_split * emeg_sample_rate
         * 2)  # We need double the length so we can do the full cross-correlation overlap
        # TODO: does this need to be integer div if we're also coercing to int?
        #  Now that downsample_rate is computed (and is a float), this div will
        #  result in a float anyway.
        // downsample_rate)

    func = function.values.reshape(n_splits, n_samples_per_split // 2)
    n_channels = emeg_values.shape[0]

    # Reshape EMEG into splits of `seconds_per_split` s
    split_initial_timesteps = [
        start_latency
        + round(i * seconds_per_split
                # Correct for audio drift in delivery equipment
                * (1 + audio_shift_correction)
                )
        - emeg_t_start
        for i in range(n_splits)
    ]
    emeg_reshaped = np.zeros((n_channels, n_splits, n_samples_per_split))
    for split_i in range(n_splits):
        emeg_reshaped[:, split_i, :] = emeg_values[
            :, split_initial_timesteps[split_i]
               :split_initial_timesteps[split_i] + int(2 * emeg_sample_rate * seconds_per_split)
               :downsample_rate]

    # Derangements for null distribution
    derangements = np.zeros((n_derangements, n_splits), dtype=int)
    for der_i in range(n_derangements):
        derangements[der_i, :] = generate_derangement(n_splits)
    derangements = np.vstack((np.arange(n_splits), derangements))  # Include the identity on top

    # FFT cross-corr
    emeg_reshaped = np.fft.rfft(normalize(emeg_reshaped), n=n_samples_per_split, axis=-1)
    f_func = np.conj(np.fft.rfft(normalize(func), n=n_samples_per_split, axis=-1))
    corrs = np.zeros((n_channels, n_derangements + 1, n_splits, n_samples_per_split // 2))
    for der_i, derangement in enumerate(derangements):
        deranged_emeg = emeg_reshaped[:, derangement, :]
        corrs[:, der_i] = np.fft.irfft(deranged_emeg * f_func)[:, :, :n_samples_per_split//2]

    p_values = _ttest(corrs)

    latencies = np.linspace(start_latency, start_latency + seconds_per_split, n_samples_per_split // 2) / 1000

    es = SensorExpressionSet(
        functions=function.name,
        latencies=latencies,
        sensors=sensor_names,
        data=p_values,
    )

    return es


def _ttest(corrs: NDArray, f_alpha: float = 0.001, use_all_lats: bool = True):
    """
    Vectorised Welch's t-test.
    """

    # Fisher Z-Transformation
    corrs = 0.5 * np.log((1 + corrs) / (1 - corrs))
    n_channels, n_derangements, n_splits, t_steps = corrs.shape

    true_mean = np.mean(corrs[:, 0, :, :], axis=1)
    true_var = np.var(corrs[:, 0, :, :], axis=1, ddof=1)
    true_n = n_splits
    if use_all_lats:
        rand_mean = np.mean(corrs[:, 1:, :, :].reshape(n_channels, -1), axis=1).reshape(n_channels, 1)
        rand_var = np.var(corrs[:, 1:, :, :].reshape(n_channels, -1), axis=1, ddof=1).reshape(n_channels, 1)
        rand_n = n_splits * n_derangements * t_steps
    else:
        rand_mean = np.mean(corrs[:, 1:, :, :].reshape(n_channels, -1, t_steps), axis=1)
        rand_var = np.var(corrs[:, 1:, :, :].reshape(n_channels, -1, t_steps), axis=1, ddof=1)
        rand_n = n_splits * n_derangements

    # Vectorized two-sample t-tests for all channels and time steps
    numerator = true_mean - rand_mean
    denominator = np.sqrt(true_var / true_n + rand_var / rand_n)
    df = ((true_var / true_n + rand_var / rand_n) ** 2 /
          ((true_var / true_n) ** 2 / (true_n - 1) +
           (rand_var / rand_n) ** 2 / (rand_n - 1)))

    t_stat = numerator / denominator
    p = stats.t.sf(np.abs(t_stat), df) * 2  # two-tailed p-value

    # Adjust p-values for multiple comparisons (Bonferroni correction) [NOT SURE ABOUT THIS]
    pvalues_adj = p #np.minimum(1, p * t_steps * n_channels / (1 - f_alpha))

    return pvalues_adj
