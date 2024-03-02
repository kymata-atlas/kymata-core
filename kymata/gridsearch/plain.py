from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import stats

from kymata.entities.functions import Function
from kymata.math.combinatorics import generate_derangement
from kymata.math.vector import normalize, get_stds
from kymata.entities.expression import ExpressionSet, SensorExpressionSet, HexelExpressionSet
from kymata.math.p_values import log_base, p_to_logp
from kymata.plot.plot import plot_top_five_channels_of_gridsearch


def do_gridsearch(
        emeg_values: NDArray,  # chan x time
        function: Function,
        channel_names: list,
        channel_space: str,
        start_latency: float,   # ms
        emeg_t_start: float,    # ms
        plot_location: Optional[Path] = None,
        emeg_sample_rate: int = 1000,  # Hertz
        audio_shift_correction: float = 0.000_537_5,  # seconds/second
        n_derangements: int = 1,
        seconds_per_split: float = 0.5,
        n_splits: int = 800,
        n_reps: int = 1,
        overwrite: bool = True,
) -> ExpressionSet:
    """
    Do the Kymata gridsearch over all hexels for all latencies.
    """

    channel_space = channel_space.lower()
    if channel_space not in {"sensor", "source"}:
        raise NotImplementedError(channel_space)

    # We'll need to downsample the EMEG to match the function's sample rate
    downsample_rate: int = int(emeg_sample_rate / function.sample_rate)

    n_samples_per_split = int(seconds_per_split * emeg_sample_rate * 2 // downsample_rate)


    if ave_mode == 'add':
        # commented this out to fix EMEG_paths undefined error. Assuming it is defined as # of EMEG chans?
        #n_reps = len(EMEG_paths)
        n_reps = 1
    else:
        n_reps = 1

    func_length = n_splits * n_samples_per_split // 2
    if func_length < function.values.shape[0]:
        func = function.values[:func_length].reshape(n_splits, n_samples_per_split // 2)
        print(f'WARNING: not using full 400s of the file (only using {round(n_splits * seconds_per_split, 2)}s)')
    else:
        func = function.values.reshape(n_splits, n_samples_per_split // 2)
    n_channels = emeg_values.shape[0]

    # Reshape EMEG into splits of `seconds_per_split` s
    split_initial_timesteps = [
        int(start_latency + round(i * 1000 * seconds_per_split * (1 + audio_shift_correction)) - emeg_t_start)
        for i in range(n_splits)
    ]

    emeg_reshaped = np.zeros((n_channels, n_splits * n_reps, n_samples_per_split))
    for j in range(n_reps):
        for split_i in range(n_splits):
            split_start = split_initial_timesteps[split_i]
            split_stop = split_start + int(2 * emeg_sample_rate * seconds_per_split)
            emeg_reshaped[:, split_i + (j * n_splits), :] = emeg_values[:, j, split_start:split_stop:downsample_rate]

    del emeg_values

    # Derangements for null distribution
    derangements = np.zeros((n_derangements, n_splits * n_reps), dtype=int)
    for der_i in range(n_derangements):
        derangements[der_i, :] = generate_derangement(n_splits * n_reps, n_splits)
    derangements = np.vstack((np.arange(n_splits * n_reps), derangements))  # Include the identity on top

    # Fast cross-correlation using FFT
    emeg_reshaped = normalize(emeg_reshaped)
    emeg_stds = get_stds(emeg_reshaped, n_samples_per_split // 2)
    emeg_reshaped = np.fft.rfft(emeg_reshaped, n=n_samples_per_split, axis=-1)
    F_func = np.conj(np.fft.rfft(normalize(func), n=n_samples_per_split, axis=-1))
    corrs = np.zeros((n_channels, n_derangements + 1, n_splits * n_reps, n_samples_per_split // 2))
    for der_i, derangement in enumerate(derangements):
        deranged_emeg = emeg_reshaped[:, derangement, :]
        corrs[:, der_i] = np.fft.irfft(deranged_emeg * F_func)[:, :, :n_samples_per_split//2] / emeg_stds[:, derangement]

    # work out autocorrelation for channel-by-channel plots
    noise = normalize(np.random.randn(func.shape[0], func.shape[1])) * 0
    noisy_func = normalize(np.copy(func)) + noise
    nn = n_samples_per_split // 2

    F_noisy_func = np.fft.rfft(normalize(noisy_func), n=nn, axis=-1)
    F_func = np.conj(np.fft.rfft(normalize(func), n=nn, axis=-1))

    auto_corrs = np.fft.irfft(F_noisy_func * F_func)

    del F_func, deranged_emeg, emeg_reshaped

    # derive pvalues
    log_pvalues = _ttest(corrs)

    latencies_ms = np.linspace(start_latency, start_latency + (seconds_per_split * 1000), n_samples_per_split // 2 + 1)[:-1]

    plot_top_five_channels_of_gridsearch(
        corrs=corrs,
        auto_corrs=auto_corrs,
        function=function,
        n_reps=n_reps,
        n_splits=n_splits,
        n_samples_per_split=n_samples_per_split,
        latencies=latencies_ms,
        save_to=plot_location,
        log_pvalues=log_pvalues,
        overwrite=overwrite,
        )

    if channel_space == "sensor":
        es = SensorExpressionSet(
            functions=function.name,
            latencies=latencies_ms / 1000,  # seconds
            sensors=channel_names,
            data=log_pvalues,
        )
    elif channel_space == "source":
        es = HexelExpressionSet(
            functions=function.name,
            latencies=latencies_ms / 1000,  # seconds
            hexels_lh=channel_names[0],
            hexels_rh=channel_names[1],
            # Unstack the data
            data_lh=log_pvalues[:len(channel_names[0]), :],
            data_rh=log_pvalues[len(channel_names[0]):, :],
        )
    else:
        raise NotImplementedError(channel_space)

    return es


def _ttest(corrs: NDArray, use_all_lats: bool = True) -> ArrayLike:

    """
    Vectorised Welch's t-test.
    """
    n_channels, n_derangements, n_splits, t_steps = corrs.shape

    # Fisher Z-Transformation
    corrs_z = 0.5 * np.log((1 + corrs) / (1 - corrs))

    # Non-deranged values are on top
    true_mean = np.mean(corrs_z[:, 0, :, :], axis=1)
    true_var = np.var(corrs_z[:, 0, :, :], axis=1, ddof=1)
    true_n = n_splits

    # Recompute mean and var for null correlations
    if use_all_lats:
        rand_mean = np.mean(corrs_z[:, 1:, :, :].reshape(n_channels, -1), axis=1).reshape(n_channels, 1)
        rand_var = np.var(corrs_z[:, 1:, :, :].reshape(n_channels, -1), axis=1, ddof=1).reshape(n_channels, 1)
        rand_n = n_splits * n_derangements * t_steps
    else:
        rand_mean = np.mean(corrs_z[:, 1:, :, :].reshape(n_channels, -1, t_steps), axis=1)
        rand_var = np.var(corrs_z[:, 1:, :, :].reshape(n_channels, -1, t_steps), axis=1, ddof=1)
        rand_n = n_splits * n_derangements

    # Vectorized two-sample t-tests for all channels and time steps
    numerator = true_mean - rand_mean
    denominator = np.sqrt(true_var / true_n + rand_var / rand_n)
    df = ((true_var / true_n + rand_var / rand_n) ** 2 /
          ((true_var / true_n) ** 2 / (true_n - 1) +
           (rand_var / rand_n) ** 2 / (rand_n - 1)))

    t_stat = numerator / denominator

    if np.min(df) <= 300:
        p = stats.t.sf(np.abs(t_stat), df) * 2  # two-tailed p-value
        log_p = p_to_logp(p)
    else:
        # norm v good approx for this, (logsf for t not implemented in logspace)
        log_p = stats.norm.logsf(np.abs(t_stat)) + np.log(2)
        log_p /= np.log(log_base)  # log base correction

    return log_p
