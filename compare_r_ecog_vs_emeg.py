"""Compare per-neuron latency-vs-layer correlation between two datasets.

This script loads per-layer log files from two model-output directories, extracts the
significant hits, computes for each neuron index the regression p-value (testing
non-zero correlation / slope) from a linear regression of latency vs layer, and plots
p(A) vs p(B) as a 2D scatter.

It intentionally mirrors the data selection logic used in `kymata/invokers/neuron_scatter.py`:
- Parse each layer log
- Threshold on Sidak-corrected -log(p)
- For each neuron index, select ONE point per layer (the max -log(p) if multiple exist)
- Run SciPy `linregress(layer, latency)` to get the p-value

Output:
- PNG scatter saved under the given --output-dir

Example:
  python compare_r_ecog_vs_emeg.py \
    --log-dir-a /path/to/emeg/log \
    --log-dir-b /path/to/ecog/log \
    --dataset-a emeg \
    --dataset-b ecog \
    --output-dir /path/to/ecog_vs_emeg
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import NormalDist
from typing import Final

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from kymata.math.probability import p_to_logp, sidak_correct


LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<prefix>\S+):\s+"
    r"peak\s+lat:\s*(?P<lat>-?\d+(?:\.\d+)?),\s*"
    # Some logs (ecog/emeg) include peak corr, some don't; spacing is inconsistent.
    r"(?:peak\s+corr:\s*(?P<corr>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s+|\s*,\s*))?"
    r"(?:\[sensor\]\s+ind:\s*(?P<ind>\d+),\s*)?"
    r"-log\(pval\):\s*(?P<logp>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def _model_geometry_from_logdir(log_dir: Path) -> tuple[int, int]:
    """Infer (n_layers, n_neurons) in the same way as neuron_scatter.py."""
    s = str(log_dir)
    if "qwen" in s and "encoder" not in s:
        return 29, 3584
    if "encoder" in s or "large-v2" in s:
        return 32, 1280
    return 33, 4096


def _n_sensors_for_dataset(dataset: str) -> int:
    if dataset == "eeg":
        return 64
    if dataset == "meg":
        return 306
    if dataset == "emeg":
        return 370
    if dataset == "ecog":
        return 300
    raise NotImplementedError(dataset)


def _sidak_logp_threshold(n_layers: int, n_neurons: int, dataset: str) -> float:
    n_sensors = _n_sensors_for_dataset(dataset)
    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    # Same multiplicity factor as neuron_scatter.py
    return float(-p_to_logp(sidak_correct(alpha, 200 * n_sensors * n_neurons * n_layers)))


def _iter_layer_log_lines(log_file: Path) -> list[str]:
    with log_file.open("r") as f:
        return [line for line in f.readlines() if "[sensor] ind" in line]


def load_sig_matrix(log_dir: Path, dataset: str) -> tuple[np.ndarray, int, int, float]:
    """Load significant entries.

    Returns (sig, n_layers, n_neurons, thres)

    sig has columns: [peak_lat, sensor_ind, logp, layer_no, neuron_no]
    """

    n_layers, n_neurons = _model_geometry_from_logdir(log_dir)
    thres = _sidak_logp_threshold(n_layers, n_neurons, dataset)

    lat_sig = np.zeros((n_layers * n_neurons, 5), dtype=float)

    row = 0
    for li in range(n_layers):
        if dataset in ("emeg", "ecog"):
            log_file = log_dir / f"slurm_log_{li}.txt"
        else:
            log_file = log_dir / f"layer{li}_{n_neurons-1}_gridsearch_{dataset}_results.txt"

        all_text = _iter_layer_log_lines(log_file)
        if len(all_text) != n_neurons:
            raise ValueError(
                f"Length mismatch in layer {li}: expected {n_neurons}, got {len(all_text)} ({log_file})"
            )

        for k in range(n_neurons):
            line = all_text[k].strip()
            m = LINE_RE.match(line)
            if m is None:
                raise ValueError(f"Could not parse log line (layer={li}, neuron_row={k}): {line!r}")

            peak_lat = float(m.group("lat"))
            sensor_ind = float(m.group("ind")) if m.group("ind") is not None else 0.0
            logp = float(m.group("logp"))
            neuron_no = float(m.group("prefix").split("_")[-1].rstrip(":"))

            lat_sig[row] = [peak_lat, sensor_ind, logp, li, neuron_no]
            row += 1

    sig = lat_sig[(lat_sig[:, 0] != 0) & (lat_sig[:, 2] > thres)]
    return sig, n_layers, n_neurons, thres


def continuous_run_indices(sig: np.ndarray, *, min_run_len: int = 5) -> set[int]:
    """Return indices that have a run of >=min_run_len consecutive layers in `sig`.

    Mirrors the logic in `kymata/invokers/neuron_scatter.py`:
    - collect layers per neuron index
    - de-dup + sort
    - detect any consecutive run of length >=min_run_len
    """

    layers_by_index: dict[int, list[int]] = {}
    for idx_i, layer_i in zip(sig[:, 4], sig[:, 3]):
        idx_int = int(idx_i)
        layer_int = int(layer_i)
        layers_by_index.setdefault(idx_int, []).append(layer_int)

    for idx_int, ls in layers_by_index.items():
        layers_by_index[idx_int] = sorted(set(ls))

    ok: set[int] = set()
    for idx_int, ls in layers_by_index.items():
        if len(ls) < min_run_len:
            continue
        run_len = 1
        for i in range(1, len(ls)):
            if ls[i] == ls[i - 1] + 1:
                run_len += 1
                if run_len >= min_run_len:
                    ok.add(int(idx_int))
                    break
            else:
                run_len = 1
    return ok


def per_index_pvalues(
    sig: np.ndarray,
    n_layers: int,
    *,
    min_layers_for_regression: int = 2,
    allowed_indices: set[int] | None = None,
) -> dict[int, float]:
    """Compute regression p-value for each neuron index present in sig.

    Mirrors the per-layer point selection: for each layer, keep the point with max -log(p).
    """

    idxs = np.unique(np.asarray(sig[:, 4], dtype=int))
    out: dict[int, float] = {}

    for idx in idxs.tolist():
        if allowed_indices is not None and int(idx) not in allowed_indices:
            continue
        sig_idx = sig[np.asarray(sig[:, 4], dtype=int) == int(idx)]
        if sig_idx.size == 0:
            continue

        layer_int = np.asarray(sig_idx[:, 3], dtype=int)
        lat = np.asarray(sig_idx[:, 0], dtype=float)
        logp = np.asarray(sig_idx[:, 2], dtype=float)

        best_lat_by_layer = np.full(n_layers, np.nan, dtype=float)
        best_logp_by_layer = np.full(n_layers, np.nan, dtype=float)

        for li in range(n_layers):
            m = layer_int == li
            if not np.any(m):
                continue
            j_local = int(np.argmax(logp[m]))
            best_lat_by_layer[li] = float(lat[m][j_local])
            best_logp_by_layer[li] = float(logp[m][j_local])

        mask = np.isfinite(best_lat_by_layer) & np.isfinite(best_logp_by_layer)
        layers_used = np.arange(n_layers, dtype=float)[mask]
        lat_used = best_lat_by_layer[mask].astype(float)

        if layers_used.size < min_layers_for_regression:
            continue
        if float(np.nanstd(lat_used)) == 0.0:
            continue

        lr = linregress(layers_used, lat_used)
        p = float(lr.pvalue)
        if not np.isfinite(p):
            continue

        out[int(idx)] = p

    return out


def plot_p_scatter(
    p_a: dict[int, float],
    p_b: dict[int, float],
    *,
    label_a: str,
    label_b: str,
    output_png: Path,
) -> None:
    common = sorted(set(p_a.keys()) & set(p_b.keys()))
    if not common:
        raise ValueError("No overlapping neuron indices between datasets after filtering.")

    x = np.asarray([p_a[i] for i in common], dtype=float)
    y = np.asarray([p_b[i] for i in common], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # Plot -log10(p) on linear axes.
    # Guard: clip p to a tiny positive to avoid log(0).
    eps = 1e-300
    x_clip = np.clip(x, eps, 1.0)
    y_clip = np.clip(y, eps, 1.0)
    x_logp = -np.log10(x_clip)
    y_logp = -np.log10(y_clip)

    ax.scatter(x_logp, y_logp, s=10, alpha=0.7, linewidths=0)

    # Axis limits with padding, synced so y=x is meaningful.
    lo = float(min(np.min(x_logp), np.min(y_logp)))
    hi = float(max(np.max(x_logp), np.max(y_logp)))
    pad = 0.5
    lo = max(0.0, lo - pad)
    hi = hi + pad
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # y=x reference
    ax.plot([lo, hi], [lo, hi], linestyle=":", color="black", linewidth=1, alpha=0.6)

    # Threshold guide lines at p=0.05
    thr = float(-np.log10(0.05))
    ax.axvline(thr, linestyle=":", color="gray", linewidth=1.5, alpha=0.8)
    ax.axhline(thr, linestyle=":", color="gray", linewidth=1.5, alpha=0.8)

    ax.set_xlabel(f"-log10(p) (latency vs layer) — {label_a}")
    ax.set_ylabel(f"-log10(p) (latency vs layer) — {label_b}")
    ax.set_title(f"Per-neuron -log10(p) comparison (n={len(common)})")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir-a", type=Path, required=True)
    p.add_argument("--log-dir-b", type=Path, required=True)
    p.add_argument("--dataset-a", choices=["emeg", "ecog", "eeg", "meg"], required=True)
    p.add_argument("--dataset-b", choices=["emeg", "ecog", "eeg", "meg"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--min-layers",
        type=int,
        default=2,
        help="Minimum number of layers with data required per neuron to run regression.",
    )
    p.add_argument(
        "--min-consecutive-layers",
        type=int,
        default=5,
        help=(
            "Require a run of at least this many consecutive layers with significant hits for a neuron "
            "to be eligible. Use <=0 to disable."
        ),
    )
    p.add_argument("--label-a", type=str, default=None)
    p.add_argument("--label-b", type=str, default=None)
    args = p.parse_args()

    sig_a, n_layers_a, _n_neurons_a, _thres_a = load_sig_matrix(args.log_dir_a, args.dataset_a)
    sig_b, n_layers_b, _n_neurons_b, _thres_b = load_sig_matrix(args.log_dir_b, args.dataset_b)

    min_run = int(args.min_consecutive_layers)
    allowed_a = None
    allowed_b = None
    if min_run > 0:
        allowed_a = continuous_run_indices(sig_a, min_run_len=min_run)
        allowed_b = continuous_run_indices(sig_b, min_run_len=min_run)

    p_a = per_index_pvalues(
        sig_a,
        n_layers_a,
        min_layers_for_regression=args.min_layers,
        allowed_indices=allowed_a,
    )
    p_b = per_index_pvalues(
        sig_b,
        n_layers_b,
        min_layers_for_regression=args.min_layers,
        allowed_indices=allowed_b,
    )

    label_a = args.label_a or args.dataset_a
    label_b = args.label_b or args.dataset_b

    out_png = args.output_dir / "pval_scatter.png"
    plot_p_scatter(p_a, p_b, label_a=label_a, label_b=label_b, output_png=out_png)

    # Lightweight console summary
    common = sorted(set(p_a.keys()) & set(p_b.keys()))
    x = np.asarray([p_a[i] for i in common], dtype=float)
    y = np.asarray([p_b[i] for i in common], dtype=float)

    # Significance categories at p<0.05 (no filtering; purely descriptive)
    alpha = 0.05
    sig_a = x < alpha
    sig_b = y < alpha
    n_both = int(np.sum(sig_a & sig_b))
    n_either = int(np.sum(sig_a | sig_b))
    n_none = int(np.sum(~(sig_a | sig_b)))
    n_emeg = int(np.sum(sig_a & ~sig_b))
    n_ecog = int(np.sum(sig_b & ~sig_a))

    print(
        "p-value comparison done\n"
        f"  n_common={len(common)}\n"
        f"  saved={out_png}\n"
        + (f"  constraint: >= {min_run} consecutive layers\n" if min_run > 0 else "  constraint: not applied\n")
        + f"  significance (p<{alpha}): both={n_both}, either={n_either}, emeg={n_emeg}, ecog={n_ecog}, none={n_none}\n"
        + f"  median_p({label_a})={float(np.median(x)):.3g}, median_p({label_b})={float(np.median(y)):.3g}\n"
        + f"  corr(log10 p_a, log10 p_b)={float(np.corrcoef(np.log10(np.clip(x, 1e-300, 1.0)), np.log10(np.clip(y, 1e-300, 1.0)))[0, 1]):.3g}"
    )


if __name__ == "__main__":
    main()
