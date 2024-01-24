import numpy as np
from numpy.typing import NDArray
from scipy import stats

from kymata.entities.functions import Function
from kymata.math.combinatorics import generate_derangement
from kymata.math.vector import normalize, get_stds
#from kymata.entities.expression import SensorExpressionSet, p_to_logp
import matplotlib.pyplot as plt

import sys

def do_gridsearch(
        emeg_values: NDArray,  # chan x time
        function: Function,
        sensor_names: list[str],
        start_latency: float,   # ms
        emeg_t_start: float,    # ms
        emeg_sample_rate: int = 1000,  # Hertz
        audio_shift_correction: float = 0.000_537_5,  # seconds/second  # TODO: describe in which direction?
        n_derangements: int = 1,
        seconds_per_split: float = 0.5,
        n_splits: int = 800,
        ave_mode: str = 'ave', # either ave or add, for averaging over input files or adding in as extra evidence
        add_autocorr: bool = True,
        plot_name: str = 'example',
        part_name: str = '',
        ):
    """
    Do the Kymata gridsearch over all hexels for all latencies.
    """

    # We'll need to downsample the EMEG to match the function's sample rate
    downsample_rate: int = int(emeg_sample_rate / function.sample_rate)

    n_samples_per_split = int(seconds_per_split * emeg_sample_rate * 2 // downsample_rate)

    if ave_mode == 'add':
        n_reps = len(EMEG_paths)
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
    split_initial_timesteps = [int(start_latency + round(i * 1000 * seconds_per_split * (1 + audio_shift_correction)) - emeg_t_start)
        for i in range(n_splits)]

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

    if add_autocorr:
        auto_corrs = np.zeros((n_splits, n_samples_per_split//2))
        noise = normalize(np.random.randn(func.shape[0], func.shape[1])) * 0
        noisy_func = normalize(np.copy(func)) + noise
        nn = n_samples_per_split // 2

        F_noisy_func = np.fft.rfft(normalize(noisy_func), n=nn, axis=-1)
        F_func = np.conj(np.fft.rfft(normalize(func), n=nn, axis=-1))

        auto_corrs = np.fft.irfft(F_noisy_func * F_func)

    del F_func, deranged_emeg, emeg_reshaped

    log_pvalues = _ttest(corrs)

    latencies = np.linspace(start_latency, start_latency + (seconds_per_split * 1000), n_samples_per_split // 2 + 1)[:-1]

    if plot_name is not None:
        plt.figure(1)
        corr_avrs = np.mean(corrs[:, 0], axis=-2) ** 2 # (n_chans, n_derangs, n_splits, t_steps) -> (n_chans, t_steps)
        maxs = np.max(corr_avrs, axis=1)
        n_amaxs = 5
        amaxs = np.argpartition(maxs, -n_amaxs)[-n_amaxs:]
        amax = np.argmax(corr_avrs) // (n_samples_per_split // 2)
        amaxs = [i for i in amaxs if i != amax] # + [209]

        plt.plot(latencies, np.mean(corrs[amax, 0], axis=-2).T, 'r-', label=amax)
        plt.plot(latencies, np.mean(corrs[amaxs, 0], axis=-2).T, label=amaxs)
        std_null = np.mean(np.std(corrs[:, 1], axis=-2), axis=0).T * 3 / np.sqrt(n_reps * n_splits) # 3 pop. std.s
        std_real = np.std(corrs[amax, 0], axis=-2).T * 3  / np.sqrt(n_reps * n_splits)
        av_real = np.mean(corrs[amax, 0], axis=-2).T
        plt.fill_between(latencies, -std_null, std_null, alpha=0.5, color='grey')
        plt.fill_between(latencies, av_real - std_real, av_real + std_real, alpha=0.25, color='red')

        if add_autocorr:
            peak_lat_ind = np.argmax(corr_avrs) % (n_samples_per_split // 2)
            peak_lat = latencies[peak_lat_ind]
            peak_corr = np.mean(corrs[amax, 0], axis=-2)[peak_lat_ind]
            print(f'{part_name}: {function.name}: \tpeak lat, peak corr, ind, -log(pval):', peak_lat, peak_corr, amax, -log_pvalues[amax][peak_lat_ind])
            sys.stdout.flush()

            auto_corrs = np.mean(auto_corrs, axis=0)
            plt.plot(latencies, np.roll(auto_corrs, peak_lat_ind) * peak_corr / np.max(auto_corrs), 'k--', label='func auto-corr')

        plt.axvline(0, color='k')
        plt.legend()
        plt.xlabel('latencies (ms)')
        plt.ylabel('Corr coef.')
        plt.savefig(f'{plot_name}_1.png')
        plt.clf()

        plt.figure(2)
        plt.plot(latencies, -log_pvalues[amax].T, 'r-', label=amax)
        plt.plot(latencies, -log_pvalues[amaxs].T, label=amaxs)
        plt.axvline(0, color='k')
        plt.legend()
        plt.xlabel('latencies (ms)')
        plt.ylabel('p-values')
        plt.savefig(f'{plot_name}_3.png')
        plt.clf()

    sensor_pvalues = np.max(-log_pvalues, axis=1)
    # sensor_corrs = corrs[:, np.argmax(corrs**2, axis=1)]
    sensor_lats = latencies[np.argmax(-log_pvalues, axis=1)]

    return sensor_pvalues, sensor_lats  # , sensor_corrs

    """es = SensorExpressionSet(
        functions=function.name,
        latencies=latencies / 1000,
        sensors=sensor_names,
        data=log_pvalues,
    )"""

    return es


def _ttest(corrs: NDArray, use_all_lats: bool = True):
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
    # TODO: why looking at only 1 in the n_derangements dimension?
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
        log_p = np.log(stats.t.sf(np.abs(t_stat), df) * 2)  # two-tailed p-value
    else:
        # norm v good approx for this, (logsf for t not implemented in logspace)
        log_p = stats.norm.logsf(np.abs(t_stat)) + np.log(2) 

    return log_p / np.log(10)  # log base correction
