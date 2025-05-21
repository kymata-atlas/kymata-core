from logging import getLogger
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import stats

from kymata.entities.transform import Transform
from kymata.math.combinatorics import generate_derangement
from kymata.io.layouts import SensorLayout
from kymata.math.vector import normalize, get_stds
from kymata.entities.expression import ExpressionSet, SensorExpressionSet, HexelExpressionSet
from kymata.math.probability import LOGP_BASE, p_to_logp
from kymata.plot.gridsearch import plot_top_five_channels_of_gridsearch

from tqdm import tqdm
import gc

_logger = getLogger(__name__)


def do_gridsearch(
    emeg_values: NDArray | list[NDArray],  # chans x reps x time
    transform: Transform,
    channel_names: list,
    channel_space: str,
    start_latency: float,  # seconds
    emeg_t_start: float,  # seconds
    stimulus_shift_correction: float,  # seconds/second
    stimulus_delivery_latency: float,  # seconds
    emeg_sample_rate: float,  # Hertz
    invsol_npy_file_name: list,
    emeg_layout: Optional[SensorLayout] = None,
    plot_location: Optional[Path] = None,
    n_derangements: int = 1,
    seconds_per_split: float = 1,
    n_splits: int = 400,
    n_reps: int = 1,
    plot_top_five_channels: bool = False,
    overwrite: bool = True,
    speedup: bool = False,
    premorphed_inverse_operator_dir: Optional[Path] = None,
    emeg_std_source: Optional[NDArray] = None,
) -> ExpressionSet:
    """
    Perform a grid search over all hexels for all latencies using EMEG data and a given transform.

    This function processes EMEG data to compute the correlation between sensor or source signals
    and a specified transform across multiple latencies. The results include statistical significance
    testing and optional plotting.

    Args:
        emeg_values (NDArray): A 2D array of EMEG values with shape (channels, reps, time).
        transform (Transform): The transform against which the EMEG data will be correlated. It should
            have a `values` attribute representing the transform's values and a `sample_rate`
            attribute indicating its sample rate.
        channel_names (list): List of channel names corresponding to the EMEG data. For 'sensor' space,
            it is a flat list of sensor names. For 'source' space, it is a list containing two lists:
            left hemisphere and right hemisphere hexel names.
        channel_space (str): The type of channel space used, either 'sensor' or 'source'.
        start_latency (float): The starting latency for the grid search in seconds.
        emeg_t_start (float): The starting time of the EMEG data in seconds.
        stimulus_shift_correction (float): Correction factor for stimulus shift in seconds per second.
        stimulus_delivery_latency (float): Correction offset for stimulus delivery in seconds.
        plot_location (Optional[Path], optional): Path to save the plot of the top five channels of the
            grid search. If None, plotting is skipped. Default is None.
        emeg_sample_rate (float, optional): The sample rate of the EMEG data in Hertz.
        n_derangements (int, optional): Number of derangements (random permutations) used to create the
            null distribution. Default is 1.
        seconds_per_split (float, optional): Duration of each split in seconds. Default is 0.5 seconds.
        n_splits (int, optional): Number of splits used for analysis. Default is 400.
        n_reps (int, optional): Number of repetitions for each split. Default is 1.
        plot_top_five_channels (bool, optional): Plots the p-values and correlation values of the top
            five channels in the gridsearch. Default is False.
        overwrite (bool, optional): Whether to overwrite existing plot files. Default is True.

    Returns:
        ExpressionSet: An ExpressionSet object (either SensorExpressionSet or HexelExpressionSet)
        containing the log p-values for each channel/hexel and latency.

    Notes:
        - The function down-samples the EMEG data to match the transform's sample rate.
        - The EMEG data is reshaped into segments of the specified duration (`seconds_per_split`).
        - Cross-correlations between the EMEG data and the transform are computed using FFT.
        - Statistical significance is assessed using a vectorized Welch's t-test.
        - If specified, the results are plotted and saved to the given location.
    """

    # Set random seed to keep derangement orderings
    # deterministic between runs
    np.random.seed(17)

    premorphed_inverse_operator_path = [Path(premorphed_inverse_operator_dir, premorphed_inverse_file) for premorphed_inverse_file in invsol_npy_file_name]

    channel_space = channel_space.lower()
    if channel_space not in {"sensor", "source"}:
        raise NotImplementedError(channel_space)

    # We'll need to downsample the EMEG to match the transform's sample rate
    if emeg_sample_rate != transform.sample_rate:
        _logger.warning(f"Data sample rate ({emeg_sample_rate} Hz) and "
                        f"transform sample rate ({transform.sample_rate} Hz) differ. "
                        f"Data will be down-sampled.")
    downsample_ratio = emeg_sample_rate / transform.sample_rate
    if downsample_ratio.is_integer():
        downsample_rate: int = int(emeg_sample_rate / transform.sample_rate)
    else:
        raise ValueError(f"Data sample rate ({emeg_sample_rate} Hz) and "
                         f"transform sample rate ({transform.sample_rate} Hz) are incompatible.")

    n_samples_per_split = int(seconds_per_split * emeg_sample_rate * 2 // downsample_rate)

    # the number of samples in the transform 'trial' which is half that needed for the EMEG
    n_trans_samples_per_split = n_samples_per_split // 2

    if speedup:
        _logger.info(f"Tianyi's speedup mode is on. ")
        _logger.info(f"Total EMEG length is {emeg_values[0].shape[2] / emeg_sample_rate:.2f} s"
                    f" @ {emeg_sample_rate} Hz")
    else:
        _logger.info(f"Total EMEG length is {emeg_values.shape[2] / emeg_sample_rate:.2f} s"
                    f" @ {emeg_sample_rate} Hz")

    _logger.info(f"Total transform length is {transform.values.shape[0] / transform.sample_rate:.2f} s"
                 f" @ {transform.sample_rate} Hz")

    trans_length = n_splits * n_trans_samples_per_split
    if trans_length < transform.values.shape[0]:
        _logger.warning(f"WARNING: not using full length of the file (only using {n_splits * seconds_per_split:.2f}s)")
        trans = transform.values[:trans_length].reshape(n_splits, n_trans_samples_per_split)
    else:
        trans = transform.values.reshape(n_splits, n_trans_samples_per_split)

    # In case trans contains a fully constant split, normalize will involve a divide by zero error, resulting in a nan
    # which will infect everything downstream. Rather than try and catch and fix that, we instead kick it back to the
    # invoker to say, just ensure that this can't happen.
    try:
        trans = normalize(trans)
    except (ZeroDivisionError, FloatingPointError) as ex:
        _logger.error("Could not normalize transform.")
        _logger.error(
            f"It's possible that the {transform.name} transform contains a constant {seconds_per_split}-second "
            "segment, which is invalid for gridsearch. Try increasing the seconds-per-split to greater than "
            f"{seconds_per_split} seconds, and adjust `n_splits` accordingly"
        )
        raise ex
    
    if speedup:

        n_channels = emeg_values[0].shape[0]

        # Reshape EMEG into splits of `seconds_per_split` s
        split_initial_timesteps = [
            int(
                (start_latency * emeg_sample_rate)
                - (emeg_t_start * emeg_sample_rate)
                + round(
                    i
                    * emeg_sample_rate
                    * seconds_per_split #seconds
                    * (1 + stimulus_shift_correction) #seconds
                )  # splits, stretched by the shift correction
                + round(stimulus_delivery_latency * emeg_sample_rate)  # correct for stimulus delivery latency delay
            )
            for i in range(n_splits)
        ]

        emeg_reshaped_list = []
        for emeg in emeg_values:

            emeg_reshaped = np.zeros((n_channels, n_splits * n_reps, n_samples_per_split))
            for j in range(n_reps):
                for split_i in range(n_splits):
                    split_start = split_initial_timesteps[split_i]
                    split_stop = split_start + int(2 * emeg_sample_rate * seconds_per_split)
                    emeg_reshaped[:, split_i + (j * n_splits), :] = emeg[
                        :, j, split_start:split_stop:downsample_rate
                    ]

            del emeg

            # Fast cross-correlation using FFT
            normalize(emeg_reshaped, inplace=True)
            emeg_reshaped = np.fft.rfft(emeg_reshaped, n=n_samples_per_split, axis=-1)
            emeg_reshaped_list.append(emeg_reshaped)

        # Derangements for null distribution
        derangements = np.zeros((n_derangements, n_splits * n_reps), dtype=int)
        for der_i in range(n_derangements):
            derangements[der_i, :] = generate_derangement(n_splits * n_reps, n_splits)
        derangements = np.vstack(
            (np.arange(n_splits * n_reps), derangements)
        )  # Include the identity on top

        F_trans = np.conj(np.fft.rfft(trans, n=n_samples_per_split, axis=-1))
        if n_reps > 1:
            F_trans = np.tile(F_trans, (n_reps, 1))

        F_trans = np.ascontiguousarray(F_trans)

        n_sources = 20484
        corrs = np.empty((n_sources, n_derangements + 1, n_splits * n_reps, n_trans_samples_per_split))

        premorphed_inverse_operator_all = []
        for i in range(len(emeg_reshaped_list)):
            premorphed_inverse_operator_all.append(
                np.load(premorphed_inverse_operator_path[i])
            )

        for der_i, derangement in tqdm(enumerate(derangements)):
            # for par_i, emeg_reshaped in tqdm(enumerate(emeg_reshaped_list)):
            #     # deranged_emeg = emeg_reshaped[:, derangement, :]
            #     # premorphed_inverse_operator = np.load(premorphed_inverse_operator_path[par_i])
            #     premorphed_inverse_operator = premorphed_inverse_operator_all[par_i]
            #     reshaped_irfft = np.fft.irfft(emeg_reshaped[:, derangement, :] * F_trans)[:, :, :n_trans_samples_per_split].reshape(n_channels, -1)
            #     # corrs_ind = np.matmul(premorphed_inverse_operator, reshaped_irfft).reshape(n_sources, deranged_emeg.shape[1], n_trans_samples_per_split)
            #     corrs_ind = np.matmul(premorphed_inverse_operator, reshaped_irfft).reshape(n_sources, derangement.shape[0], n_trans_samples_per_split)
            #     corrs_list.append(corrs_ind)
            #     # del deranged_emeg, reshaped_irfft
            #     del reshaped_irfft
            # del emeg_reshaped
            # reshaped_irfft = np.fft.irfft(np.ascontiguousarray(emeg_reshaped_list)[:, :, derangement, :] * F_trans)[:, :, :, :n_trans_samples_per_split].reshape(len(emeg_reshaped_list), n_channels, -1)
            reshaped_irfft = np.fft.irfft(np.ascontiguousarray(emeg_reshaped_list)[:, :, derangement, :] * F_trans)[:, :, :, :n_trans_samples_per_split]
            # corrs_ind = np.matmul(premorphed_inverse_operator, reshaped_irfft).reshape(n_sources, deranged_emeg.shape[1], n_trans_samples_per_split)
            # corrs[:, der_i] = np.mean(np.einsum('bij, bjk -> bik', np.array(premorphed_inverse_operator_all), reshaped_irfft).reshape(len(emeg_reshaped_list), n_sources, derangement.shape[0], n_trans_samples_per_split), axis = 0) / emeg_std_source[:, derangement]
            # corrs[:, der_i] = np.einsum('bij, bjkl -> ikl', np.ascontiguousarray(premorphed_inverse_operator_all), reshaped_irfft) / (emeg_std_source[:, derangement] * len(emeg_reshaped_list))
            corrs[:, der_i] = np.einsum('bij, bjkl -> ikl', np.ascontiguousarray(premorphed_inverse_operator_all), reshaped_irfft) / (emeg_std_source[:, derangement] * len(emeg_reshaped_list))
            # del deranged_emeg, reshaped_irfft
            del reshaped_irfft
            # _logger.info(f"Size of corrs_ind is {corrs_ind.nbytes / 1024**3:.2f} GB")
            # import ipdb; ipdb.set_trace()
            # # Process in smaller chunks to avoid memory issues
            # chunk_size = 1000  # Adjust this value based on available memory
            # for start_idx in tqdm(range(0, corrs.shape[0], chunk_size)):
            #     end_idx = min(start_idx + chunk_size, corrs.shape[0])
            #     slices = np.empty((len(corrs_list), ) + corrs_list[0][start_idx:end_idx].shape)
            #     for chunk_i, chunk in enumerate(corrs_list):
            #         slices[chunk_i] = chunk[start_idx:end_idx]
            #     # slices = np.array([chunk[start_idx:end_idx] for chunk in corrs_list])
            #     corrs[start_idx:end_idx, der_i] = np.mean(slices, axis=0) / emeg_std_source[start_idx:end_idx, derangement]
            # corrs[:, der_i] = np.mean(np.array(corrs_list), axis=0) / emeg_std_source[:, derangement]
            # Allocate output mean array
            # mean_corr = np.empty_like(corrs_list[0])

            # # Incrementally compute the mean to avoid 4D allocation
            # for arr in tqdm(corrs_list):
            #     mean_corr += arr
            #     del arr
            # mean_corr /= len(corrs_list)

            # Now safely divide and store in the final tensor
            # corrs[:, der_i] = np.mean(corrs_ind, axis=0) / emeg_std_source[:, derangement]
            # np.divide(
            #             np.mean(corrs_ind, axis=0),          # shape: (20484, 400, 200)
            #             emeg_std_source[:, derangement],     # must be broadcastable to (20484, 400, 200)
            #             out=corrs[:, der_i]                  # must also be shape (20484, 400, 200)
            #         )

            # del corrs_ind

    else:

        n_channels = emeg_values.shape[0]

        # Reshape EMEG into splits of `seconds_per_split` s
        split_initial_timesteps = [
            int(
                (start_latency * emeg_sample_rate)
                - (emeg_t_start * emeg_sample_rate)
                + round(
                    i
                    * emeg_sample_rate
                    * seconds_per_split #seconds
                    * (1 + stimulus_shift_correction) #seconds
                )  # splits, stretched by the shift correction
                + round(stimulus_delivery_latency * emeg_sample_rate)  # correct for stimulus delivery latency delay
            )
            for i in range(n_splits)
        ]

        emeg_reshaped = np.zeros((n_channels, n_splits * n_reps, n_samples_per_split))
        for j in range(n_reps):
            for split_i in range(n_splits):
                split_start = split_initial_timesteps[split_i]
                split_stop = split_start + int(2 * emeg_sample_rate * seconds_per_split)
                emeg_reshaped[:, split_i + (j * n_splits), :] = emeg_values[
                    :, j, split_start:split_stop:downsample_rate
                ]

        del emeg_values

        # Derangements for null distribution
        derangements = np.zeros((n_derangements, n_splits * n_reps), dtype=int)
        for der_i in range(n_derangements):
            derangements[der_i, :] = generate_derangement(n_splits * n_reps, n_splits)
        derangements = np.vstack(
            (np.arange(n_splits * n_reps), derangements)
        )  # Include the identity on top

        # Fast cross-correlation using FFT
        normalize(emeg_reshaped, inplace=True)
        emeg_stds = get_stds(emeg_reshaped, n_trans_samples_per_split)
        emeg_reshaped = np.fft.rfft(emeg_reshaped, n=n_samples_per_split, axis=-1)
        F_trans = np.conj(np.fft.rfft(trans, n=n_samples_per_split, axis=-1))
        if n_reps > 1:
            F_trans = np.tile(F_trans, (n_reps, 1))
        corrs = np.zeros(
            (n_channels, n_derangements + 1, n_splits * n_reps, n_trans_samples_per_split)
        )
        for der_i, derangement in enumerate(derangements):
            deranged_emeg = emeg_reshaped[:, derangement, :]
            corrs[:, der_i] = (
                np.fft.irfft(deranged_emeg * F_trans)[:, :, :n_trans_samples_per_split]
                / emeg_stds[:, derangement]
            )     

    del deranged_emeg, emeg_reshaped

    # In case there was a large part of the transform which was constant, the corr will be undefined (nan).
    # We want p-vals here to be 1.

    # derive pvalues
    log_pvalues = _ttest(corrs)

    latencies_ms = np.linspace(
        (start_latency * emeg_sample_rate),
        (start_latency * emeg_sample_rate) + (seconds_per_split * 1000),
        n_trans_samples_per_split + 1,
    )[:-1]

    if plot_top_five_channels:
        # work out autocorrelation for channel-by-channel plots
        noise = normalize(np.random.randn(trans.shape[0], trans.shape[1])) * 0
        noisy_trans = trans + noise
        normalize(noisy_trans, inplace=True)

        F_noisy_trans = np.fft.rfft(noisy_trans, n=n_trans_samples_per_split, axis=-1)
        F_trans = np.conj(np.fft.rfft(trans, n=n_trans_samples_per_split, axis=-1))

        auto_corrs = np.fft.irfft(F_noisy_trans * F_trans)

        del F_trans

        plot_top_five_channels_of_gridsearch(
            corrs=corrs,
            auto_corrs=auto_corrs,
            transform=transform,
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
            transforms=transform.name,
            latencies=latencies_ms / 1000,  # seconds
            sensors=channel_names,
            data=log_pvalues,
            sensor_layout=emeg_layout,
        )
    elif channel_space == "source":
        es = HexelExpressionSet(
            transforms=transform.name,
            latencies=latencies_ms / 1000,  # seconds
            hexels_lh=channel_names[0],
            hexels_rh=channel_names[1],
            # Unstack the data
            data_lh=log_pvalues[: len(channel_names[0]), :],
            data_rh=log_pvalues[len(channel_names[0]) :, :],
        )
    else:
        raise NotImplementedError(channel_space)

    return es


def _ttest(corrs: NDArray, use_all_lats: bool = True) -> ArrayLike:
    """
    Perform a vectorized Welch's t-test on correlation matrices.

    This function calculates the two-sample Welch's t-test statistics and their corresponding
    log p-values for given correlation matrices. The test compares true correlation values against
    null (random) correlation values after applying Fisher's Z-transformation.

    Parameters:
    -----------
    corrs : NDArray
        A 4D array of correlation values with shape (n_channels, n_derangements, n_splits, t_steps).
        - n_channels: Number of channels.
        - n_derangements: Number of derangements (random permutations).
        - n_splits: Number of splits or repetitions.
        - t_steps: Number of time steps.

    use_all_lats : bool, optional
        If True, use all latencies for computing the mean and variance of null correlations.
        If False, compute mean and variance per time step. Default is True.

    Returns:
    --------
    ArrayLike
        A 2D array of log p-values with shape (n_channels, t_steps), representing the log p-values
        of the t-tests for each channel and time step.

    Notes:
    ------
    - The correlation values are first transformed using Fisher's Z-transformation.
    - The function computes the mean and variance of the transformed true and null correlations.
    - Welch's t-test is then performed in a vectorized manner to obtain the t-statistics.
    - Depending on the degrees of freedom, either the t-distribution or normal distribution is used
      to compute the log p-values.
    - The function ensures numerical stability and precision by applying log transformations where necessary.

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
        rand_mean = np.mean(
            corrs_z[:, 1:, :, :].reshape(n_channels, -1), axis=1
        ).reshape(n_channels, 1)
        rand_var = np.var(
            corrs_z[:, 1:, :, :].reshape(n_channels, -1), axis=1, ddof=1
        ).reshape(n_channels, 1)
        rand_n = n_splits * n_derangements * t_steps
    else:
        rand_mean = np.mean(
            corrs_z[:, 1:, :, :].reshape(n_channels, -1, t_steps), axis=1
        )
        rand_var = np.var(
            corrs_z[:, 1:, :, :].reshape(n_channels, -1, t_steps), axis=1, ddof=1
        )
        rand_n = n_splits * n_derangements

    # Vectorized two-sample t-tests for all channels and time steps
    numerator = true_mean - rand_mean
    denominator = np.sqrt(true_var / true_n + rand_var / rand_n)
    df = (true_var / true_n + rand_var / rand_n) ** 2 / (
        (true_var / true_n) ** 2 / (true_n - 1)
        + (rand_var / rand_n) ** 2 / (rand_n - 1)
    )

    t_stat = numerator / denominator

    if np.min(df) <= 300:
        p = stats.t.sf(np.abs(t_stat), df) * 2  # two-tailed p-value
        log_p = p_to_logp(p)
    else:
        # norm v good approx for this, (logsf for t not implemented in logspace)
        log_p = stats.norm.logsf(np.abs(t_stat)) + np.log(2)
        log_p /= np.log(LOGP_BASE)  # log base correction

    return log_p
