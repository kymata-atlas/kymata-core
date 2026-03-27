import argparse
import re
from logging import basicConfig, INFO, getLogger
from pathlib import Path
from typing import Any, Literal

import numpy as np
from matplotlib import colors, pyplot as plt
from statsmodels.regression.linear_model import RegressionResults, OLS
from statsmodels.tools import add_constant

from kymata.io.logging import log_message, date_format


_logger = getLogger(__file__)


def plot_line_of_best_fit(layer: int, sig: np.ndarray[Any, np.dtype[Any]], output_dir: Path, dataset_name: str,
                          degree: Literal[1, 2],
                          axlim_ms=(None, None), min_count_for_average: int = 5) -> RegressionResults:
    """
    Plot a line of best fit for a set of significant neurons
    """

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

    inf_mask = np.isfinite(mean_lat_by_layer)
    layers_to_fit = layers[inf_mask]
    latency_to_fit = mean_lat_by_layer[inf_mask]

    fig, ax = plt.subplots()
    if layers_to_fit.size >= 2:
        if degree == 1:
            fit_results = OLS(
                latency_to_fit,
                add_constant(layers_to_fit)
            ).fit()
            intercept, slope = fit_results.params
            latency_prediction = intercept + slope * layers
        elif degree == 2:
            fit_results = OLS(
                latency_to_fit,
                add_constant(np.column_stack([
                    # We won't include the linear term, as we want to force the stationary point to be at layer 0
                    # layers_to_fit,
                    layers_to_fit ** 2,
                ]))
            ).fit()
            intercept, quadratic = fit_results.params
            slope = 0  # Stationary point forced to be at layer 0
            latency_prediction = intercept + slope * layers + quadratic * (layers ** 2)
        else:
            raise NotImplementedError()

        ax.plot(latency_prediction, layers, linestyle=':', linewidth=2, color='black')

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

        # Pearson correlation and p-value
        if degree == 1:
            r = float(np.sqrt(fit_results.rsquared) * np.sign(slope))
            p = float(fit_results.pvalues[1])
            bic = float(fit_results.bic)
            ax.text(
                0.02,
                0.98,
                f"Pearson r = {r:.3g}\n"
                f"p = {p:.3g}\n"
                f"R² = {fit_results.rsquared:.2g}\n"
                f"BIC = {bic:.2g}",
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
            )
        elif degree == 2:
            p = float(fit_results.pvalues[1])
            bic = float(fit_results.bic)
            ax.text(
                0.02,
                0.98,
                f"p = {p:.3g}\n"
                f"R² = {fit_results.rsquared:.2g}\n"
                f"BIC = {bic:.2g}",
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
        fit_results = None

    ax.set_xlabel('Mean latency (ms)')
    ax.set_ylabel('Layer number')
    ax.set_xlim(axlim_ms)
    ax.set_ylim(-1, layer)
    ax.set_aspect("equal")

    plt.tight_layout()

    model_name = "linear" if degree == 1 else "quadratic"
    save_loc = output_dir / f"{dataset_name}_best_{model_name}_fit.png"

    plt.savefig(save_loc, dpi=600)

    plt.close(fig)

    return fit_results


if __name__ == '__main__':

    basicConfig(format=log_message, datefmt=date_format, level=INFO)

    parser = argparse.ArgumentParser(description='Neuron-level scatter plot from slurm logs')
    parser.add_argument('--neuron-sig', '-i', nargs="+", type=Path, help="Path to neuron")
    parser.add_argument('--output-dir', '-o', type=Path, help="Path to figures")
    parser.add_argument(
        '--dataset',
        choices=['emeg', 'ecog', 'eeg', 'meg'],
        default='emeg',
        help="Dataset to use: 'emeg' or 'ecog'",
    )
    args = parser.parse_args()
    
    dataset: str = args.dataset
    output_dir: Path = args.output_dir

    sig_loc: Path
    layer = None
    neuron = None
    sigs = []
    datasets = []
    for sig_loc in args.neuron_sig:

        # Parse layer and neuron out of filename
        filename_re = re.compile(r"(?P<dataset>[^_]+)"
                                 r"_sig_neurons"
                                 r"_layer(?P<layer>[0-9]+)"
                                 r"_neuron(?P<neuron>[0-9]+)"
                                 r"\.npy")
        match = filename_re.match(sig_loc.name)
        this_layer = int(match.group("layer"))
        this_neuron = int(match.group("neuron"))
        this_dataset = str(match.group("dataset"))

        if layer is not None and this_layer != layer:
            raise ValueError(f"Inconsistent layer in files: {layer} vs {this_layer}")
        if neuron is not None and this_neuron != neuron:
            raise ValueError(f"Inconsistent neuron in files: {neuron} vs {this_neuron}")

        layer = this_layer
        neuron = this_neuron
        datasets.append(this_dataset)

        with sig_loc.open("rb") as sig_f:
            # each row is (peak_lat, sensor_idx, logp, layer, neuron_no)
            sig = np.load(sig_f)
        sigs.append(sig)

    # Concatenate rows from all sigs
    sig_unified = np.concatenate(sigs, axis=0)

    linear_model = plot_line_of_best_fit(
        degree=1,
        min_count_for_average=0, axlim_ms=(0, 400),
        layer=layer, sig=sig_unified, output_dir=output_dir, dataset_name="+".join(datasets),
    )
    quadratic_model = plot_line_of_best_fit(
        degree=2,
        min_count_for_average=0, axlim_ms=(0, 400),
        layer=layer, sig=sig_unified, output_dir=output_dir, dataset_name="+".join(datasets),
    )
    print(f"Linear BIC = {linear_model.bic:.2f}")
    print(f"Quadratic BIC = {quadratic_model.bic:.2f}")
    delta_bic = quadratic_model.bic - linear_model.bic
    favoured = "quadratic" if delta_bic < 0 else "linear"
    if abs(delta_bic) > 10:
        strength = "very strong"
    elif abs(delta_bic) > 6:
        strength = "strong"
    elif abs(delta_bic) > 2:
        strength = "positive"
    else:
        strength = "weak"
    print(f"The {favoured} model is {strength}ly preferred")
