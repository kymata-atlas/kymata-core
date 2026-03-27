import argparse
import re
from collections import defaultdict
from logging import basicConfig, INFO, getLogger
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray
from pandas import DataFrame
from seaborn import heatmap

from kymata.io.logging import log_message, date_format
from kymata.math.probability import p_to_logp, sidak_correct

_logger = getLogger(__file__)


def _get_data_shape(dataset: str, log_dir: Path) -> tuple[int, int, int]:
    if 'qwen' in str(log_dir) and 'encoder' not in str(log_dir):
        layer = 29  # 41 # 66 64 34 33
        neuron = 3584  # 4096 5120
    elif 'encoder' in str(log_dir) or 'large-v2' in str(log_dir):
        layer = 32  # 41 # 66 64 34 33
        neuron = 1280  # 4096 5120
    else:
        layer = 33  # 41 # 66 64 34 33
        neuron = 4096  # 4096 5120

    if dataset == "eeg":
        n_sensors = 64
    elif dataset == "meg":
        n_sensors = 306
    elif dataset == "emeg":
        n_sensors = 370
    elif dataset == "ecog":
        n_sensors = 300  # number of regions
    else:
        raise NotImplementedError()
    return layer, n_sensors, neuron


def _get_significant_neurons(log_dir: Path, dataset: str, thres: float) -> NDArray:

    layer, n_sensors, neuron = _get_data_shape(dataset, log_dir)

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

    line_emeg_ecog_re = re.compile(
        r"^(?P<prefix>\S+):\s+"
        r"peak\s+lat:\s*(?P<lat>-?\d+(?:\.\d+)?),\s+"
        r"peak\s+corr:\s*(?P<corr>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s+|$)"
        r"(?:\[sensor\]\s+ind:\s*(?P<ind>\d+),\s+)?"
        r"-log\(pval\):\s*(?P<logp>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
    )

    row = 0
    for li in range(layer):
        if dataset == "emeg" or dataset == "ecog":
            log_file = log_dir / f'slurm_log_{li}.txt'
        else:
            log_file = log_dir / f'layer{li}_{neuron - 1}_gridsearch_{dataset}_results.txt'
        with log_file.open("r") as f:
            all_text = [line for line in f.readlines() if '[sensor] ind' in line]

        if len(all_text) != neuron:
            raise ValueError(f'Length mismatch in layer {li}: expected {neuron}, got {len(all_text)}')

        for k in range(neuron):
            line = all_text[k].strip()
            if dataset in {"emeg", "ecog"}:
                m = line_emeg_ecog_re.match(line)
            else:
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
    return sig


def _dataset_colour(dataset: str) -> str:
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
    return color


def _get_threshold(layer: int, n_sensors: int, neuron: int) -> float:
    # Keep same thresholding approach as the other script
    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = -p_to_logp(sidak_correct(alpha, 200 * n_sensors * neuron * layer))
    return thres


def latency_scatter(log_dir: Path, output_dir: Path, dataset: str):
    """Recreate the original neuron-level scatter plot from the per-layer log files.

    Reads slurm logs per layer, extracts (peak latency, peak corr, sensor ind, -log10(pval)),
    filters significant neurons, then produces a scatter plot in the original style:

    x = peak latency (ms) OR neuron index
      y = layer index
      color = -log10(pval)

    Output is saved under further_analysis_results/.
    """

    _logger.info(f"Creating latency scatter for {dataset} from {log_dir!s}")

    layer, n_sensors, neuron = _get_data_shape(dataset, log_dir)
    thres = _get_threshold(layer, n_sensors, neuron)
    # Rows are (peak_lat, sensor_ind, logp, layer, neuron_no)
    sig = _get_significant_neurons(log_dir, dataset, thres)

    # Save dataset for this modality
    results_out_loc = output_dir / f"{dataset}_sig_neurons_layer{layer}_neuron{neuron}.npy"
    with results_out_loc.open("wb") as f:
        np.save(f, sig)
        print(f"Saved {results_out_loc.name}")

    fig, ax = plt.subplots()

    # Scatter: neuron index vs layer, colored by -log10(p)
    logp_norm = colors.Normalize(vmin=float(thres), vmax=70.0, clip=True)

    x_label = 'Latency (ms)'
    x = sig[:, 0]

    y_label = "Layer number"
    y = sig[:, 3]

    ax.scatter(x, y, c=_dataset_colour(dataset), norm=logp_norm, s=4, alpha=0.9, linewidths=0)

    ax.set_ylim(-1, layer)
    ax.set_xlim(0, 400)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_title(dataset.upper())
    ax.set_box_aspect(1)
    plt.tight_layout()

    plt.savefig(save_loc = output_dir / f"{dataset}_neuron_scatter_latency.png", dpi=600)
    plt.close(fig)


def _convert_to_heatmap(x, y, neuron_cutoff, total_layers):

    data_vals = defaultdict(lambda: defaultdict(int))
    for neuron, layer in zip(x, y):
        data_vals[layer][neuron] = 1

    heatmap_df = DataFrame([
        {"Neuron": neuron, "Layer": layer, "Value": data_vals[layer][neuron]}
        for neuron in range(*neuron_cutoff)
        for layer in range(total_layers)
    ])

    return heatmap_df.pivot(index="Layer", columns="Neuron", values="Value")


def neuron_scatter(log_dir: Path, output_dir: Path, dataset: str, draw_mode: str, neuron_cutoff: tuple[int, int]):
    """Recreate the original neuron-level scatter plot from the per-layer log files.

    Reads slurm logs per layer, extracts (peak latency, peak corr, sensor ind, -log10(pval)),
    filters significant neurons, then produces a scatter plot in the original style:

    x = peak latency (ms) OR neuron index
      y = layer index
      color = -log10(pval)

    Output is saved under further_analysis_results/.
    """
    _logger.info(f"Creating neuron scatter for {dataset} from {log_dir!s}")

    layer, n_sensors, neuron = _get_data_shape(dataset, log_dir)
    thres = _get_threshold(layer, n_sensors, neuron)
    # Rows are (peak_lat, sensor_ind, logp, layer, neuron_no)
    sig = _get_significant_neurons(log_dir, dataset, thres)

    fig, ax = plt.subplots(figsize=(20, 5))

    x_label = 'Neuron index'
    x = sig[:, 4]

    y_label = "Layer number"
    y = sig[:, 3]

    if neuron_cutoff is not None and min(neuron_cutoff) >= 0:
        # Cap to actual number of neurons
        neuron_cutoff = (neuron_cutoff[0], min(neuron_cutoff[1], neuron))
        # Apply the cutoff
        old_x, old_y = x, y
        x, y = [], []
        for i, xx in enumerate(old_x):
            if neuron_cutoff[0] <= xx < neuron_cutoff[1]:
                x.append(xx)
                y.append(old_y[i])
    else:
        neuron_cutoff = (0, neuron)

    if draw_mode == "dots":
        ax.scatter(x, y, c=_dataset_colour(dataset), s=4, linewidths=0)

        ax.set_xlim(neuron_cutoff[0] - 1, neuron_cutoff[1])
        ax.set_ylim(-1, layer)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    elif draw_mode == "dotgrid":
        # Plot grey dots for all neurons
        grid_x, grid_y = np.meshgrid(range(*neuron_cutoff), range(layer))
        ax.scatter(grid_x, grid_y, c="grey", s=50, linewidths=0)

        # Plot colour for significant neurons
        ax.scatter(x, y, c=_dataset_colour(dataset), s=100)

        ax.set_xlim(neuron_cutoff[0] - 1, neuron_cutoff[1])
        ax.set_ylim(-1, layer)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    elif draw_mode == "tickgrid":

        # Plot grey dots for all neurons
        grid_x, grid_y = np.meshgrid(range(*neuron_cutoff), range(layer))
        ax.scatter(grid_x, grid_y, c="grey", s=6, linewidths=2, marker="|")

        # Plot colour for significant neurons
        ax.scatter(x, y, c=_dataset_colour(dataset), s=4, linewidths=0)

        ax.set_xlim(neuron_cutoff[0] - 1, neuron_cutoff[1])
        ax.set_ylim(-1, layer)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    elif draw_mode == "heatmap":

        heatmap_data = _convert_to_heatmap(x, y, neuron_cutoff, layer)
        heatmap(heatmap_data,
                ax=ax,
                cmap=LinearSegmentedColormap.from_list(f"{dataset}_gradient", ["grey", _dataset_colour(dataset)]),
                cbar=False,
                linewidths=0.5,
                )
        ax.invert_yaxis()

    else:
        raise NotImplementedError()

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(output_dir / f"{dataset}_neuron_scatter_neuron_{draw_mode}.png", dpi=600)
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
    parser.add_argument(
        "--draw-mode",
        type=str,
        choices=["dots", "dotgrid", "tickgrid", "heatmap"],
        default="dots",
        help="How to draw the neuron scatter"
    )
    parser.add_argument(
        "--neuron-cutoff",
        nargs=2,
        type=int,
        default=(-1, -1),
        help="Only show this range of neurons (first, last); supply -1 to either to skip cutoff"
    )
    args = parser.parse_args()

    if args.x_axis == "latency":
        latency_scatter(log_dir=args.log_dir, output_dir=args.output_dir, dataset=args.dataset)
    elif args.x_axis == "neuron":
        neuron_scatter(log_dir=args.log_dir, output_dir=args.output_dir, dataset=args.dataset,
                       draw_mode=args.draw_mode, neuron_cutoff=args.neuron_cutoff)
    else:
        raise NotImplementedError()
