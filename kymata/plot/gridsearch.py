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
    n_samples_per_split: int,
    n_reps: int,
    n_splits: int,
    auto_corrs: NDArray,
    log_pvalues: any,
    # I/O args
    save_to: Optional[Path] = None,
    overwrite: bool = True,
):
    """
    Generates correlation and p-value plots showing the top five channels of the gridsearch.

    Args:
        latencies (NDArray[any]): Array of latency values (e.g., time points in milliseconds) for the x-axis of the plots.
        corrs (NDArray[any]): Correlation coefficients array with shape (n_channels, n_conditions, n_splits, n_time_steps).
        transform (Transform): The transform object whose name attribute will be used in the plot title.
        n_samples_per_split (int): Number of samples per split used in the grid search.
        n_reps (int): Number of repetitions in the grid search.
        n_splits (int): Number of splits in the grid search.
        auto_corrs (NDArray[any]): Auto-correlation values array used for plotting the transform auto-correlation.
        log_pvalues (any): Array of log-transformed p-values for each channel and time point.
        save_to (Optional[Path], optional): Path to save the generated plot. If None, the plot is not saved. Default is None.
        overwrite (bool, optional): If True, overwrite the existing file if it exists. Default is True.

    Raises:
        FileExistsError: If the file already exists at save_to and overwrite is set to False.

    Notes:
        The function generates two subplots:

        - The first subplot shows the correlation coefficients over latencies for the top five channels.
        - The second subplot shows the corresponding p-values for these channels.
    """

    figure, axis = pyplot.subplots(1, 2, figsize=(15, 7))
    figure.suptitle(
        f"{transform.name}: Plotting corrs and pvalues for top five channels"
    )

    maxs = np.min(log_pvalues, axis=1)
    n_amaxs = 5
    amaxs = np.argpartition(maxs, -n_amaxs)[-n_amaxs:]
    amax = np.argmin(log_pvalues) // (n_samples_per_split // 2)
    amaxs = [i for i in amaxs if i != amax]  # + [209]

    axis[0].plot(latencies, np.mean(corrs[amax, 0], axis=-2).T, "r-", label=amax)
    axis[0].plot(latencies, np.mean(corrs[amaxs, 0], axis=-2).T, label=amaxs)
    std_null = (
        np.mean(np.std(corrs[:, 1], axis=-2), axis=0).T * 3 / np.sqrt(n_reps * n_splits)
    )  # 3 pop std.s
    std_real = np.std(corrs[amax, 0], axis=-2).T * 3 / np.sqrt(n_reps * n_splits)
    av_real = np.mean(corrs[amax, 0], axis=-2).T

    axis[0].fill_between(latencies, -std_null, std_null, alpha=0.5, color="grey")
    axis[0].fill_between(
        latencies, av_real - std_real, av_real + std_real, alpha=0.25, color="red"
    )

    peak_lat_ind = np.argmin(log_pvalues) % (n_samples_per_split // 2)
    peak_lat = latencies[peak_lat_ind]
    peak_corr = np.mean(corrs[amax, 0], axis=-2)[peak_lat_ind]
    print(
        f"{transform.name}: peak lat: {peak_lat:.1f},   peak corr: {peak_corr:.4f}   [sensor] ind: {amax},   -log(pval): {-log_pvalues[amax][peak_lat_ind]:.4f}"
    )

    auto_corrs = np.mean(auto_corrs, axis=0)
    axis[0].plot(
        latencies,
        np.roll(auto_corrs, peak_lat_ind) * peak_corr / np.max(auto_corrs),
        "k--",
        label="trans auto-corr",
    )

    axis[0].axvline(0, color="k")
    axis[0].legend()
    axis[0].set_title("Corr coef.")
    axis[0].set_xlabel("latencies (ms)")
    axis[0].set_ylabel("Corr coef.")

    axis[1].plot(latencies, -log_pvalues[amax].T, "r-", label=amax)
    axis[1].plot(latencies, -log_pvalues[amaxs].T, label=amaxs)
    axis[1].axvline(0, color="k")
    axis[1].legend()
    axis[1].set_title("p-values")
    axis[1].set_xlabel("latencies (ms)")
    axis[1].set_ylabel("p-values")

    if save_to is not None:
        pyplot.rcParams["savefig.dpi"] = 300
        save_to = Path(save_to, transform.name + "_gridsearch_top_five_channels.png")

        if overwrite or not save_to.exists():
            pyplot.savefig(Path(save_to))
        else:
            raise FileExistsError(save_to)

    pyplot.clf()
    pyplot.close()
