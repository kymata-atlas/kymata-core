import numpy as np
from numpy.typing import NDArray
from scipy import stats

from kymata.entities.functions import Function
from kymata.math.combinatorics import generate_derangement
from kymata.math.vector import normalize
from kymata.entities.expression import SensorExpressionSet


def do_gridsearch(function: Function,
                  function_name: str,
                  sensor_names: list[str],
                  emeg_values: NDArray,
                  n_derangements: int,
                  downsample_rate: int,
                  seconds_per_split: float,
                  n_splits: int,
                  start_latency_ms: float,
                  audio_shift_correction: float = 0.5375,
                  ) -> SensorExpressionSet:
    """
    Do the Kymata gridsearch over all hexels for all latencies
    """

    # TODO: 2000?
    n_timesteps = int((2000 * seconds_per_split) // downsample_rate)

    func = function.values.reshape(n_splits, n_timesteps // 2)
    n_channels = emeg_values.shape[0]

    # Reshape EMEG into splits of 'seconds_per_split' s
    second_start_points = [
        # TODO: 200?
        start_latency_ms + 200 + round((1000 + audio_shift_correction)  # correcting for audio shift in delivery
                                       * seconds_per_split * i)
        for i in range(n_splits)
    ]
    r_emeg = np.zeros((n_channels, n_splits, n_timesteps))
    for i in range(n_splits):
        r_emeg[:, i, :] = emeg_values[:, second_start_points[i]
                                         :second_start_points[i]
                                          + int(2000 * seconds_per_split)
                                         :downsample_rate]

    # Get derangement for null dist:
    derangements = np.zeros((n_derangements, n_splits), dtype=int)
    for i in range(n_derangements):
        derangements[i, :] = generate_derangement(n_splits)
    # Include the identity on top
    derangements = np.vstack((np.arange(n_splits), derangements))

    # FFT cross-corr
    r_emeg = np.fft.rfft(normalize(r_emeg), n=n_timesteps, axis=-1)
    f_func = np.conj(np.fft.rfft(normalize(func), n=n_timesteps, axis=-1))
    corrs = np.zeros((n_channels, n_derangements + 1, n_splits, n_timesteps // 2))
    for i, order in enumerate(derangements):
        deranged_emeg = r_emeg[:, order, :]
        corrs[:, i] = np.fft.irfft(deranged_emeg * f_func)[:, :, :n_timesteps//2]

    p_values = _ttest(corrs)

    latencies = np.linspace(start_latency_ms, start_latency_ms + 1000 * seconds_per_split, n_timesteps // 2) / 1000

    es = SensorExpressionSet(
        functions=function_name,
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
