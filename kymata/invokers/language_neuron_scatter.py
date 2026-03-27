import argparse
import re
from logging import basicConfig, INFO, getLogger
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

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
            log_file = log_dir / f'layer{li}_{neuron-1}_gridsearch_{dataset}_results.txt'
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

    # Save dataset for this modality
    results_out_loc = output_dir / f"{dataset}_sig_neurons_layer{layer}_neuron{neuron}.npy"
    with results_out_loc.open("wb") as f:
        np.save(f, sig)
        print(f"Saved {results_out_loc.name}")

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

    ax.set_ylabel('Layer number')
    ax.set_ylim(-1, layer)

    ax.set_xlabel(x_label)
    if x_axis == "latency":
        ax.set_xlim(0, 400)
        ax.set_aspect("equal")

    plt.tight_layout()

    save_loc = output_dir / f"{dataset}_neuron_scatter_{x_axis}.png"

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
