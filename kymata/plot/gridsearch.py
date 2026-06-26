from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot
from numpy._typing import NDArray

from kymata.entities.transform import Transform


def plot_top_five_channels_of_gridsearch(
    latencies: NDArray,
    corrs: NDArray,
    transform: Transform,
    logp_values: NDArray,
    auto_corr: NDArray,
    top_n_to_plot: int = 5,
    error_band_scaling_factor: float = 1.0,
    # I/O args
    save_to: Optional[Path] = None,
    overwrite: bool = True,
):
    """
    Generates correlation and p-value plots showing the top five channels of the gridsearch.

    Args:
        latencies (NDArray): Array of latency values (e.g., time points in milliseconds) for the x-axis of the plots.
        corrs (NDArray): Correlation coefficients array with shape (n_channels, n_derangements, n_splits, n_time_steps).
        transform (Transform): The transform object whose name attribute will be used in the plot title.
        logp_values (NDArray): Array of log-transformed p-values for each channel and time point with shape
            (n_channels, t_steps).
        auto_corr (NDArray): Auto-correlation values array used for plotting the transform auto-correlation.
        top_n_to_plot (int): The top-n channels to plot. Default is 5.
        error_band_scaling_factor (float): Manually scale width of error regions. Default scale is 1.
        save_to (Optional[Path], optional): Path to save the generated plot. If None (the default), the plot is not
            saved.
        overwrite (bool, optional): If True, overwrite the existing file if it exists. Default is True.

    Raises:
        FileExistsError: If the file already exists at save_to and overwrite is set to False.

    Notes:
        The function generates two subplots:

        - The first subplot shows the correlation coefficients over latencies for the top five channels.
        - The second subplot shows the corresponding p-values for these channels.
    """

    if error_band_scaling_factor != 1.0:
        error_band_message = f" (scaled by {error_band_scaling_factor:.2f})"
    else:
        error_band_message = ""

    # Best logp val for each channel
    #                                                ↓ latency axis
    channel_min_logp_vals = np.min(logp_values, axis=1)
    
    # Select top-n channel
    #
    # argpartition(arr, idx) rearranges arr such that the value at idx is in the correct position when sorted ascending,
    # and all values before that are lower (though not necessarily sorted ascending) and all values after it are higher
    # (though not necessarily sorted ascending). It returns the permutation indices into arr rather than the permuted
    # arr itself.
    #
    # Therefore, argpartition(arr, idx)[:idx] gives indices to the (unsorted) lowest values in the array. These
    # correspond to the (indices of) the best channels.
    best_chan_idxs = np.argpartition(channel_min_logp_vals, top_n_to_plot)[:top_n_to_plot]
    
    # Select the best channel to highlight and remove from remaining-top list
    best_chan_idx, peak_latency_idx = np.unravel_index(np.argmin(logp_values), shape=logp_values.shape)
    other_chan_idxs = [i for i in best_chan_idxs if i != best_chan_idx]
    del best_chan_idxs

    # Find peak
    peak_latency = latencies[peak_latency_idx]

    # Select best channel and derangement 0 for shape (n_splits, n_time_steps)
    best_chan_corrs = corrs[best_chan_idx, 0]

    # Timecourse of mean (over splits) correlation for best channel
    best_chan_corr_mean = np.mean(best_chan_corrs, axis=0)
    best_chan_corr_std_scaled  = np.std(best_chan_corrs, axis=0) * error_band_scaling_factor

    # Timecourse of mean (over splits) correlation for other channels
    other_chan_corr_means = np.mean(corrs[other_chan_idxs, 0], axis=1)

    # Null dist envelope for best channel (n-sigma STD to SEM)
    null_corrs = corrs[best_chan_idx, 1:]  # shape (n_derangements, n_splits, n_timesteps)
    null_corrs_mean = np.mean(np.mean(null_corrs, axis=1), axis=0)  # Average over splits and derangements
    # compute null std timecourse for each channel
    #                                            splits dim ↓        ↓ derangements dim
    null_corrs_std_scaled = np.mean(np.std(null_corrs, axis=1), axis=0) * error_band_scaling_factor

    mean_auto_corr = np.mean(auto_corr, axis=0)
    peak_corr = best_chan_corr_mean[peak_latency_idx]
    scaled_auto_corr = np.roll(mean_auto_corr, peak_latency_idx) * peak_corr / np.max(mean_auto_corr)

    print(f"{transform.name}:\t"
          f"peak lat: {peak_latency:.1f},\t"
          f"peak corr: {peak_corr:.4f},\t"
          f"[sensor] ind: {best_chan_idx},\t"
          f"-log(pval): {-logp_values[best_chan_idx][peak_latency_idx]:.4f}")

    # Create figure
    figure, axis = pyplot.subplots(1, 2, figsize=(15, 7))
    figure.suptitle(f"{transform.name}: Plotting corrs and pvalues for top five channels")

    # Plot overall best chan
    axis[0].plot(latencies, best_chan_corr_mean, "r-", label=best_chan_idx)
    # Plot remaining best chans
    axis[0].plot(latencies, other_chan_corr_means.T, label=other_chan_idxs)

    # Plot null mean
    axis[0].plot(latencies, null_corrs_mean, "k--", label="null distribution for best channel")

    # Plot error regions
    axis[0].fill_between(latencies, null_corrs_mean - null_corrs_std_scaled, null_corrs_mean + null_corrs_std_scaled,
                         alpha=0.5, color="grey", label="Null distribution STD" + error_band_message)
    axis[0].fill_between(latencies, best_chan_corr_mean - best_chan_corr_std_scaled, best_chan_corr_mean + best_chan_corr_std_scaled,
                         alpha=0.25, color="red", label="Best channel STD"+ error_band_message)

    # Plot autocorr
    axis[0].plot(latencies, scaled_auto_corr, "g--", label="trans auto-corr")

    axis[0].axvline(0, color="k")
    axis[0].legend()
    axis[0].set_title("Corr coef.")
    axis[0].set_xlabel("latencies (ms)")
    axis[0].set_ylabel("Corr coef.")

    axis[1].plot(latencies, -logp_values[best_chan_idx].T, "r-", label=best_chan_idx)
    axis[1].plot(latencies, -logp_values[other_chan_idxs].T, label=other_chan_idxs)
    axis[1].axvline(0, color="k")
    axis[1].legend()
    axis[1].set_title("p-values")
    axis[1].set_xlabel("latencies (ms)")
    axis[1].set_ylabel("p-values")

    # Handle figure
    if save_to is not None:
        pyplot.rcParams["savefig.dpi"] = 300
        save_to = Path(save_to, transform.name + f"_gridsearch_top_{top_n_to_plot}_channels.png")

        if overwrite or not save_to.exists():
            pyplot.savefig(Path(save_to))
        else:
            raise FileExistsError(save_to)
    pyplot.clf()
    pyplot.close()
