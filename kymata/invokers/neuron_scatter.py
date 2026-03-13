from logging import basicConfig, INFO, getLogger
from pathlib import Path
from typing import Any
import re
import argparse
from statistics import NormalDist

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import linregress

from kymata.io.logging import log_message, date_format
from kymata.math.probability import p_to_logp, sidak_correct


_logger = getLogger(__file__)


def neuron_scatter(log_dir: Path, output_dir: Path, x_axis: str, dataset: str):
    """Recreate the original neuron-level scatter plot from the per-layer log files.

    Reads slurm logs per layer, extracts (peak latency, peak corr, sensor ind, -log10(pval)),
    filters significant neurons, then produces a scatter plot in the original style:

    x = peak latency (ms) OR neuron index
      y = layer index
      color = -log10(pval)

    Output is saved under further_analysis_results/.
    """

    _logger.info(f"Creating {x_axis} scatter for {dataset} from {log_dir!s}")

    layer = 33  # 41 # 66 64 34 33
    neuron = 4096  # 4096 5120
    if dataset == "eeg":
        n_sensors = 64
    elif dataset == "meg":
        n_sensors = 306
    elif dataset == "emeg":
        n_sensors = 370
    elif dataset == "ecog":
        n_sensors = 370  # really?
    else:
        raise NotImplementedError()

    # Keep same thresholding approach as the other script
    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = -p_to_logp(sidak_correct(alpha, 200 * n_sensors * neuron * layer))

    #                                   ↓ was 6 until I removed peak_corr
    lat_sig = np.zeros((layer * neuron, 5), dtype=float)
    # columns: peak lat, ~peak corr~, ind, -log(pval), layer_no, neuron_no

    line_re = re.compile(
        r"^(?P<prefix>\S+):\s+"
        r"peak\s+lat:\s*(?P<lat>-?\d+(?:\.\d+)?),\s+"
        # r"peak\s+corr:\s*(?P<corr>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s+|$)"
        r"(?:\[sensor]\s+ind:\s*(?P<ind>\d+),\s+)?"
        r"-log\(pval\):\s*(?P<logp>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
    )

    row = 0
    for li in range(layer):
        log_file = log_dir / f'layer{li}_{neuron-1}_gridsearch_{dataset}_results.txt'
        with log_file.open("r") as f:
            all_text = [line for line in f.readlines() if '[sensor] ind' in line]

        if len(all_text) != neuron:
            raise ValueError(f'Length mismatch in layer {li}: expected {neuron}, got {len(all_text)}')

        for k in range(neuron):
            line = all_text[k].strip()
            m = line_re.match(line)
            if m is None:
                raise ValueError(f"Could not parse log line (layer={li}, neuron={k}): {line!r}")

            peak_lat = float(m.group('lat'))
            # peak_corr = float(m.group('corr'))
            sensor_ind = float(m.group('ind')) if m.group('ind') is not None else 0.0
            logp = float(m.group('logp'))
            neuron_no = float(m.group('prefix').split('_')[-1].rstrip(':'))

            # lat_sig[row] = [peak_lat, peak_corr, sensor_ind, logp, li, neuron_no]
            lat_sig[row] = [peak_lat, sensor_ind, logp, li, neuron_no]
            row += 1

    # significant neurons only                       ↓ was 3 until I removed peak_corr
    sig = lat_sig[(lat_sig[:, 0] != 0) & (lat_sig[:, 2] > thres)]

    # import ipdb; ipdb.set_trace()

    fig, ax = plt.subplots()

    # Scatter: neuron index vs layer, colored by -log10(p)
    logp_norm = colors.Normalize(vmin=float(thres), vmax=70.0, clip=True)

    if x_axis == 'latency':
        x = sig[:, 0]
        x_label = 'Latency (ms)'
    elif x_axis in ('neuron', 'neuron_index', 'index'):
        #          ↓ was 5 until I removed peak_corr
        x = sig[:, 4]
        x_label = 'Neuron index'
    else:
        raise ValueError("x_axis must be one of: 'latency', 'neuron' (aka 'neuron_index'/'index')")
    y = sig[:, 3]

    if dataset == "emeg":
        color = '#79b15b'
    elif dataset == "ecog":
        color = '#b1835b'
    elif dataset == "meg":
        color = "#D71815"
    elif dataset == "eeg":
        color = "#FF9800"
    else:
        raise NotImplementedError()
    ax.scatter(
        x,
        y,
        c=color,
        norm=logp_norm,
        s=4,
        alpha=0.9,
        linewidths=0,
    )

    # cbar = plt.colorbar(sc, ax=ax)
    # cbar.set_label('-log10(p-value)')
    # Use rounded, evenly spaced ticks (looks more natural than hard-coding)
    # cbar.locator = MaxNLocator(nbins=5)
    # cbar.update_ticks()

    if x_axis == "latency":
        ax.set_xlim(-250, 850)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Layer number')
    ax.set_ylim(-1, layer)

    plt.tight_layout()

    save_loc = output_dir / f"{dataset}_neuron_scatter_{x_axis}.png"

    plt.savefig(save_loc, dpi=600)
    plt.close(fig)

    # Plot line of best fit
    if x_axis == "latency":
        _plot_line_of_best_fit(layer, sig, output_dir, dataset, axlim_ms=(-250, 850),
                               min_count_for_average=0)


def _plot_line_of_best_fit(layer: int, sig: np.ndarray[Any, np.dtype[Any]], output_dir: Path, dataset: str,
                           axlim_ms=(None, None), min_count_for_average: int = 5) -> None:
    fig, ax = plt.subplots()

    # Use only layers with at least `min_count_for_average` significant entries.
    # Per-layer aggregation: x = layer number, y = mean latency
    mean_lat_by_layer = np.full(layer, np.nan, dtype=float)
    n_sig_by_layer = np.zeros(layer, dtype=int)
    for li in range(layer):
        #                      ↓ was 3 until I removed peak_corr
        latencies = sig[sig[:, 3] == li, 0]
        n_sig_by_layer[li] = int(latencies.size)
        if latencies.size >= min_count_for_average:
            mean_lat_by_layer[li] = float(np.mean(latencies))

    layers = np.arange(layer)

    # Color-code points by number of significant neurons per layer
    count_norm = colors.Normalize(vmin=0, vmax=300, clip=True)

    mask = np.isfinite(mean_lat_by_layer)
    x_fit = layers[mask]
    y_fit = mean_lat_by_layer[mask]

    if x_fit.size >= 2:
        ax.scatter(
            mean_lat_by_layer,
            layers,
            c=n_sig_by_layer,
            cmap='turbo',
            norm=count_norm,
            marker='o',
            s=25,
            edgecolors='black',
            linewidths=0.3,
        )

        lr = linregress(x_fit, y_fit)
        x_pred = lr.slope * layers + lr.intercept
        ax.plot(x_pred, layers, linestyle=':', linewidth=2, color='black')

        # Pearson correlation (same as lr.rvalue) + p-value
        r = float(lr.rvalue)
        p = float(lr.pvalue)
        ax.text(
            0.02,
            0.98,
            f"Pearson r = {r:.3g}\np = {p:.3g}",
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
        )
    else:
        ax.text(
            0.02,
            0.98,
            "Not enough layers with data for regression",
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
        )

    ax.set_xlabel('Mean latency (ms)')
    ax.set_ylabel('Layer number')
    ax.set_xlim(axlim_ms)
    ax.set_ylim(-1, layer)

    plt.tight_layout()

    save_loc = output_dir / f"{dataset}_best_fit.png"

    plt.savefig(save_loc, dpi=600)

    plt.close(fig)


if __name__ == '__main__':

    basicConfig(format=log_message, datefmt=date_format, level=INFO)

    parser = argparse.ArgumentParser(description='Neuron-level scatter plot from slurm logs')
    parser.add_argument('--log-dir', '-i', type=Path, help="Path to logs")
    parser.add_argument('--output-dir', '-o', type=Path, help="Path to figures")
    parser.add_argument(
        '--x-axis',
        choices=['latency', 'neuron'],
        default='neuron',
        help="X axis to plot: 'latency' (ms) or 'neuron' (neuron index)",
    )
    parser.add_argument(
        '--dataset',
        choices=['emeg', 'ecog', 'eeg', 'meg'],
        default='emeg',
        help="Dataset to use: 'emeg' or 'ecog'",
    )
    args = parser.parse_args()

    neuron_scatter(log_dir=args.log_dir, output_dir=args.output_dir, x_axis=args.x_axis, dataset=args.dataset)
