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
    n_samples_per_split: int,
    n_reps: int,
    n_splits: int,
    top_n_to_plot = 5,
    # I/O args
    save_to: Optional[Path] = None,
    overwrite: bool = True,
):
    """
    Generates correlation and p-value plots showing the top five channels of the gridsearch.

    Args:
        latencies (NDArray): Array of latency values (e.g., time points in milliseconds) for the x-axis of the plots.
        corrs (NDArray): Correlation coefficients array with shape (n_channels, n_conditions, n_splits, n_time_steps).
        transform (Transform): The transform object whose name attribute will be used in the plot title.
        logp_values (NDArray): Array of log-transformed p-values for each channel and time point with shape
            (n_channels, t_steps).
        auto_corr (NDArray): Auto-correlation values array used for plotting the transform auto-correlation.
        n_samples_per_split (int): Number of samples per split used in the grid search.
        n_reps (int): Number of repetitions in the grid search.
        n_splits (int): Number of splits in the grid search.
        top_n_to_plot (int): The top-n channels to plot. Default is 5.
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

    # Select the best channels
    channel_min_logp_vals = np.min(logp_values, axis=1)
    
    # Select top-n channels
    best_chan_idxs = np.argpartition(channel_min_logp_vals, -top_n_to_plot)[-top_n_to_plot:]
    
    # Select the best 1 to highlight and remove from remaining-top list
    best_chan_idx = np.argmin(logp_values) // (n_samples_per_split // 2)
    best_chan_idxs = [i for i in best_chan_idxs if i != best_chan_idx]  # + [209]

    best_chan_mean = np.mean(corrs[best_chan_idx, 0], axis=-2).T

    # Area distribution
    # Create null dist envelopes (3 std)
    std_null = np.mean(np.std(corrs[:, 1], axis=-2), axis=0).T * 3 / np.sqrt(n_reps * n_splits)
    std_real = np.std(corrs[best_chan_idx, 0], axis=-2).T * 3 / np.sqrt(n_reps * n_splits)

    # Find peak
    peak_latency_idx = np.argmin(logp_values) % (n_samples_per_split // 2)
    peak_latency = latencies[peak_latency_idx]
    peak_corr = np.mean(corrs[best_chan_idx, 0], axis=-2)[peak_latency_idx]

    mean_auto_corr = np.mean(auto_corr, axis=0)
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
    axis[0].plot(latencies, np.mean(corrs[best_chan_idx, 0], axis=-2).T, "r-", label=best_chan_idx)
    # Plot remaining best chans
    axis[0].plot(latencies, np.mean(corrs[best_chan_idxs, 0], axis=-2).T, label=best_chan_idxs)

    axis[0].fill_between(latencies, -std_null, std_null, alpha=0.5, color="grey")
    axis[0].fill_between(latencies, best_chan_mean - std_real, best_chan_mean + std_real, alpha=0.25, color="red")

    axis[0].plot(latencies, scaled_auto_corr, "k--", label="trans auto-corr", )

    axis[0].axvline(0, color="k")
    axis[0].legend()
    axis[0].set_title("Corr coef.")
    axis[0].set_xlabel("latencies (ms)")
    axis[0].set_ylabel("Corr coef.")

    axis[1].plot(latencies, -logp_values[best_chan_idx].T, "r-", label=best_chan_idx)
    axis[1].plot(latencies, -logp_values[best_chan_idxs].T, label=best_chan_idxs)
    axis[1].axvline(0, color="k")
    axis[1].legend()
    axis[1].set_title("p-values")
    axis[1].set_xlabel("latencies (ms)")
    axis[1].set_ylabel("p-values")

    # Handle figure
    if save_to is not None:
        pyplot.rcParams["savefig.dpi"] = 300
        save_to = Path(save_to, transform.name + "_gridsearch_top_five_channels.png")

        if overwrite or not save_to.exists():
            pyplot.savefig(Path(save_to))
        else:
            raise FileExistsError(save_to)
    pyplot.clf()
    pyplot.close()
