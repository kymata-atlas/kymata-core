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
from scipy.stats import binomtest
import torch


_logger = getLogger(__file__)


def _is_emeg_like(dataset: str) -> bool:
    """Datasets whose logs use the slurm_log_{layer}.txt format."""

    return dataset in {"emeg", "ecog"}


def _load_activation_trace(pt_path: str, idx: int, *, T_expected: int = 115) -> np.ndarray | None:
    """Load one activation trace y(t) from a saved tensor file.

    Expected tensor shape: (1, T, n_units). Returns y(t) as float array of shape (T,).
    Returns None if the file is missing or the tensor/index is invalid.
    """

    try:
        t = torch.load(pt_path, map_location="cpu")
    except FileNotFoundError:
        return None

    if hasattr(t, "detach"):
        t = t.detach()
    t_np = np.asarray(t)
    if t_np.ndim != 3 or t_np.shape[0] != 1 or t_np.shape[1] != T_expected:
        return None
    if idx < 0 or idx >= t_np.shape[2]:
        return None
    y = t_np[0, :, idx].astype(float)
    if not np.isfinite(y).all():
        return None
    return y


def _pearsonr_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (r, p) or (nan, nan) if not computable."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan"), float("nan")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan"), float("nan")
    if float(np.nanstd(x)) == 0.0 or float(np.nanstd(y)) == 0.0:
        return float("nan"), float("nan")
    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return float("nan"), float("nan")


def _cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine distance (1 - cosine similarity).

    Returns NaN if either vector has near-zero norm.
    """

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return float("nan")
    return float(1.0 - (float(np.dot(a, b)) / (na * nb)))


def neuron_scatter(
    log_dir: Path,
    output_dir: Path,
    x_axis: str,
    dataset: str,
    *,
    per_layer_scatter_color: str = "layer",
):
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

    line_emeg_re = re.compile(
        r"^(?P<prefix>\S+):\s+"
        r"peak\s+lat:\s*(?P<lat>-?\d+(?:\.\d+)?),\s+"
        r"peak\s+corr:\s*(?P<corr>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s+|$)"
        r"(?:\[sensor\]\s+ind:\s*(?P<ind>\d+),\s+)?"
        r"-log\(pval\):\s*(?P<logp>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
    )

    emeg_like = _is_emeg_like(dataset)

    row = 0
    for li in range(layer):
        if emeg_like:
            log_file = log_dir / f'slurm_log_{li}.txt'
        else:
            log_file = log_dir / f'layer{li}_{neuron-1}_gridsearch_{dataset}_results.txt'
        with log_file.open("r") as f:
            all_text = [line for line in f.readlines() if '[sensor] ind' in line]

        if len(all_text) != neuron:
            raise ValueError(f'Length mismatch in layer {li}: expected {neuron}, got {len(all_text)}')

        for k in range(neuron):
            line = all_text[k].strip()
            if emeg_like:
                m = line_emeg_re.match(line)
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

        # Find neuron indices that are significant in >=5 continuous layers, and return the indices as a list
        layers_by_index: dict[int, list[int]] = {}
        for idx_i, layer_i in zip(sig[:, 4], sig[:, 3]):
            idx_int = int(idx_i)
            layer_int = int(layer_i)
            layers_by_index.setdefault(idx_int, []).append(layer_int)

        # De-dup + sort layers per index
        for idx_int, ls in layers_by_index.items():
            layers_by_index[idx_int] = sorted(set(ls))

        hits: dict[int, list[int]] = {}
        for idx_int, ls in layers_by_index.items():
            if len(ls) < 5:
                continue

            run: list[int] = [ls[0]]
            best_run: list[int] = []
            for v in ls[1:]:
                if v == run[-1] + 1:
                    run.append(v)
                else:
                    if len(run) >= 5:
                        best_run = run[:]  # first qualifying run
                        break
                    run = [v]
            if not best_run and len(run) >= 5:
                best_run = run[:]

            if best_run:
                hits[idx_int] = best_run

        continuous_5plus_indices: list[int] = sorted(hits.keys())

        # --- Statistical test: is this proportion above chance? ---
        # Null: each layer is independently "significant" for a given neuron index with probability p0,
        # where p0 is the overall per-(layer,index) significance rate observed in `sig`.
        # Under null, for each index we simulate `layer` Bernoulli trials and ask whether there exists
        # a run of >=5 consecutive successes. This yields p_run = P(run>=5), and then we test whether
        # the observed count of indices with such a run is higher than Binomial(total_indices, p_run).
        #
        # This is a one-sided Monte-Carlo randomization test for p_run, followed by an exact binomial tail
        # for the observed count. (SciPy is already a dependency in this file.)
        total_indices = int(np.unique(np.asarray(sig[:, 4], dtype=int)).size)
        category_indices = int(len(continuous_5plus_indices))
        if total_indices > 0:
            # Overall per-cell significance rate (index-layer) from observed data
            denom = float(layer * total_indices)
            if denom <= 0:
                print("Skipping chance test: invalid denominator for p0.")
            else:
                p0 = float(sig.shape[0]) / denom
                # Numerical guard: SciPy expects probabilities in [0, 1]
                p0 = min(max(p0, 0.0), 1.0)

                # Monte Carlo estimate of chance probability of a >=5 run for an index
                rng = np.random.default_rng(0)
                n_mc = 50_000

                def _has_run_ge_k(b: np.ndarray, k: int = 5) -> bool:
                    # b: boolean array of length `layer`
                    run = 0
                    for v in b:
                        run = (run + 1) if v else 0
                        if run >= k:
                            return True
                    return False

                hits_mc = 0
                for _ in range(n_mc):
                    b = rng.random(layer) < p0
                    hits_mc += int(_has_run_ge_k(b, k=5))
                p_run = hits_mc / n_mc
                p_run = min(max(float(p_run), 0.0), 1.0)

                print(
                    f"Chance model: p0 (per-layer significant) = {p0:.6g}; "
                    f"estimated P(run>=5 in {layer} layers) = {p_run:.6g} (MC n={n_mc})"
                )

                bt = binomtest(category_indices, total_indices, p_run, alternative="greater")
                pval = float(bt.pvalue)

                # SciPy can underflow extremely small p-values to 0.0 in float64; clamp so reporting
                # stays finite and any downstream -log10(p) won't error/divide by zero.
                pval_safe = max(pval, np.finfo(float).tiny)

                # Use full-precision formatting so extremely small p-values don't appear as 0.
                # Also report -log10(p) for readability when p is tiny.
                pval_str = format(pval_safe, ".17g")
                neglog10_pval_str = format(-np.log10(pval_safe), ".6g")

                if pval == 0.0:
                    pval_str = f"< {pval_str}"

                print(
                    "Binomial test (greater): "
                    f"k={category_indices}, n={total_indices}, p0={p_run:.6g}, "
                    f"p-value={pval_str}, -log10(p)={neglog10_pval_str}"
                )
                # if total_indices > 0:
                #     # Print all eligible indices (those with a >=5 consecutive-layer run) and their run layers
                #     if not continuous_5plus_indices:
                #         print("No indices found with >=5 consecutive significant layers.")
                #     else:
                #         print(
                #             f"Eligible indices with >=5 consecutive significant layers (n={len(continuous_5plus_indices)}):"
                #         )
                #         for idx_int in continuous_5plus_indices:
                #             run_layers = hits.get(idx_int, [])
                #             print(f"  idx={idx_int}, run_len={len(run_layers)}, run_layers={run_layers}")
        else:
            print("No indices available for chance test (total_indices=0).")
        # Plot activation contours over time for a fixed neuron index across layers

        # --- Plot latency vs layer for selected neuron index/indices ---
        # If `idx_to_plot` is an int, we do the plot as before.
        # If it is a list of ints, we skip plotting and instead compute summary statistics
        # for Pearson r and its p-value across those indices.
        # By default we run list-mode stats over all qualifying indices.
        idx_to_plot: int | list[int] = continuous_5plus_indices  # an int like 2624 or continuous_5plus_indices

        def _per_index_series_and_regression(
            idx: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float] | None:
            """Single source of truth.

            Returns:
              (layers_used, lat_used, logp_used, slope, intercept, r, p)
            Where regression is latency = slope*layer + intercept.
            """

            sig_idx = sig[np.asarray(sig[:, 4], dtype=int) == int(idx)]
            if sig_idx.size == 0:
                return None

            layer_int = np.asarray(sig_idx[:, 3], dtype=int)
            lat = np.asarray(sig_idx[:, 0], dtype=float)
            logp = np.asarray(sig_idx[:, 2], dtype=float)

            best_lat_by_layer = np.full(layer, np.nan, dtype=float)
            best_logp_by_layer = np.full(layer, np.nan, dtype=float)

            for li in range(layer):
                m = layer_int == li
                if not np.any(m):
                    continue
                # Take the point with the maximum -log(p) within the layer.
                # (If you believe there can only be 1 hit/layer, this still behaves identically.)
                m_ix = np.flatnonzero(m)
                j_local = int(np.argmax(logp[m]))
                j = int(m_ix[j_local])
                best_lat_by_layer[li] = float(lat[j])
                best_logp_by_layer[li] = float(logp[j])

            mask = np.isfinite(best_lat_by_layer) & np.isfinite(best_logp_by_layer)
            layers_used = np.arange(layer, dtype=int)[mask]
            lat_used = best_lat_by_layer[mask]
            logp_used = best_logp_by_layer[mask]
            if layers_used.size < 2:
                return None

            if not np.isfinite(lat_used).all():
                return None
            if float(np.nanstd(lat_used)) == 0.0:
                return None

            lr = linregress(layers_used.astype(float), lat_used.astype(float))
            slope = float(lr.slope)
            intercept = float(lr.intercept)
            r = float(lr.rvalue)
            p = float(lr.pvalue)
            if not (np.isfinite(slope) and np.isfinite(intercept) and np.isfinite(r) and np.isfinite(p)):
                return None

            return layers_used, lat_used, logp_used, slope, intercept, r, p

        if isinstance(idx_to_plot, (list, tuple, np.ndarray)):
            indices = [int(x) for x in idx_to_plot]
            rs: list[float] = []
            ps: list[float] = []
            slopes: list[float] = []
            used_indices: list[int] = []
            skipped: list[int] = []
            layers_used_by_idx: dict[int, np.ndarray] = {}
            r_by_idx: dict[int, float] = {}
            p_by_idx: dict[int, float] = {}

            for idx in indices:
                out = _per_index_series_and_regression(idx)
                if out is None:
                    skipped.append(idx)
                    continue
                _layers_used, _lat_used, _logp_used, slope, _intercept, r, p = out
                used_indices.append(idx)
                rs.append(r)
                ps.append(p)
                slopes.append(slope)
                layers_used_by_idx[int(idx)] = np.asarray(_layers_used, dtype=int)
                r_by_idx[int(idx)] = float(r)
                p_by_idx[int(idx)] = float(p)

            if len(rs) == 0:
                print(f"No indices had enough layers with data for regression (n_total={len(indices)}).")
            else:
                r_arr = np.asarray(rs, dtype=float)
                p_arr = np.asarray(ps, dtype=float)
                slope_arr = np.asarray(slopes, dtype=float)

                # --- Summary across indices (unadjusted p-values) ---
                # You're interested in: across indices, how many show a meaningful correlation, and how significant.
                # Since you're using one regression test per neuron index, we keep this unadjusted.
                # We report:
                #   1) counts below common alpha thresholds
                #   2) directionality breakdown (positive vs negative r among significant indices)
                #   3) effect size distribution via |r| percentiles
                #   4) top-k smallest p with their r

                def _count_le(x: np.ndarray, thr: float) -> int:
                    return int(np.sum(x <= thr))

                alpha_levels = [0.05, 0.01, 0.001]
                p_counts = {a: _count_le(p_arr, a) for a in alpha_levels}

                # Directionality among p<=0.05 (unadjusted)
                sig_mask = p_arr <= 0.05
                n_sig = int(np.sum(sig_mask))
                n_pos = int(np.sum((r_arr > 0) & sig_mask))
                n_neg = int(np.sum((r_arr < 0) & sig_mask))

                # Top-k by smallest p
                k = min(10, int(p_arr.size))
                # k = 36  # all with p<0.05
                top_order = np.argsort(p_arr)[:k]
                top_lines = [
                    f"    idx={used_indices[i]}  r={r_arr[i]:+.3f}  p={p_arr[i]:.3g}  slope={slope_arr[i]:+.3g} ms/layer"
                    for i in top_order
                ]

                print(
                    "Latency-vs-layer regression: significance across indices\n"
                    f"  n_total={len(indices)}, n_used={len(rs)}, n_skipped={len(skipped)}\n"
                    f"  Unadjusted p-value counts: "
                    + ", ".join([f"p<= {a:g}: {p_counts[a]}" for a in alpha_levels])
                    + "\n"
                    f"  Direction among p<=0.05: pos={n_pos}, neg={n_neg}, total={n_sig}\n"
                    f"  Effect size (|r|) percentiles: "
                    f"50%={float(np.median(np.abs(r_arr))):.3f}, 75%={float(np.percentile(np.abs(r_arr), 75)):.3f}, "
                    f"90%={float(np.percentile(np.abs(r_arr), 90)):.3f}\n"
                    f"  Slope (ms/layer) percentiles: "
                    f"50%={float(np.median(slope_arr)):+.3g}, 75%={float(np.percentile(slope_arr, 75)):+.3g}, "
                    f"90%={float(np.percentile(slope_arr, 90)):+.3g}\n"
                    "  Top indices by smallest p (with r, p):\n"
                    + "\n".join(top_lines)
                )
                if skipped:
                    print(f"  skipped (insufficient/degenerate regression): {skipped}")

                # --- New plot: per-layer points within each index ---
                # For each qualifying index (p<=0.05 and positive slope):
                #   for each significant layer in that index, plot one point:
                #     x = latency(layer) - latency(ref_layer)
                #     y = cosine_distance(activation(layer), activation(ref_layer))
                # where ref_layer is the FIRST significant layer for that index.
                # sig_ix = np.flatnonzero((p_arr <= 0.05) & (slope_arr > 0))
                sig_ix = np.flatnonzero(p_arr <= 0.05)
                if sig_ix.size == 0:
                    print("No indices with p<=0.05; skipping per-layer latencyΔ vs cos-dist plot.")
                else:
                    pt_template = (
                        "/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/"
                        "predicted_function_contours/asr_models/qwen/decoder_text/segment_8_layer{i}.pt"
                    )

                    x_lat_delta: list[float] = []
                    y_cos_dist: list[float] = []
                    c_val: list[float] = []
                    missing_idx: list[int] = []
                    n_points_total = 0

                    for j in sig_ix.tolist():
                        idx = int(used_indices[int(j)])
                        layers_sig = layers_used_by_idx.get(idx)
                        if layers_sig is None or layers_sig.size == 0:
                            missing_idx.append(idx)
                            continue

                        # Use the significant layers for this index (sorted)
                        layers_sig_sorted = np.sort(np.unique(layers_sig.astype(int)))
                        ref_layer = int(layers_sig_sorted[0])

                        # Reference latency: from the regression-selected latencies in _per_index_series_and_regression
                        # We can recover it from sig by taking the best latency at ref_layer.
                        sig_idx = sig[np.asarray(sig[:, 4], dtype=int) == int(idx)]
                        if sig_idx.size == 0:
                            missing_idx.append(idx)
                            continue
                        layer_int = np.asarray(sig_idx[:, 3], dtype=int)
                        lat = np.asarray(sig_idx[:, 0], dtype=float)
                        logp = np.asarray(sig_idx[:, 2], dtype=float)

                        m_ref = layer_int == ref_layer
                        if not np.any(m_ref):
                            missing_idx.append(idx)
                            continue
                        ref_lat = float(lat[m_ref][int(np.argmax(logp[m_ref]))])

                        # Reference activation
                        ref_path = pt_template.format(i=ref_layer)
                        y_ref = _load_activation_trace(ref_path, idx)
                        if y_ref is None:
                            missing_idx.append(idx)
                            continue

                        # For each significant layer: latency delta and cosine distance
                        for li in layers_sig_sorted.tolist():
                            li_int = int(li)
                            m_li = layer_int == li_int
                            if not np.any(m_li):
                                continue
                            li_lat = float(lat[m_li][int(np.argmax(logp[m_li]))])
                            li_logp = float(np.max(logp[m_li]))

                            pt_path = pt_template.format(i=li_int)
                            y = _load_activation_trace(pt_path, idx)
                            if y is None:
                                continue

                            dx = float(li_lat - ref_lat)
                            dy = _cosine_distance(y, y_ref)
                            if not (np.isfinite(dx) and np.isfinite(dy)):
                                continue

                            x_lat_delta.append(dx)
                            y_cos_dist.append(float(dy))
                            # Choose the color value for this point.
                            # - 'layer': integer layer index
                            # - 'logp':  -log(pval) for the best hit in that layer for this index
                            # - 'r':     per-index regression Pearson r (same color across layers of an index)
                            # - 'p':     per-index regression p-value (same color across layers of an index)
                            if per_layer_scatter_color == "layer":
                                c_val.append(float(li_int))
                            elif per_layer_scatter_color == "logp":
                                c_val.append(float(li_logp))
                            elif per_layer_scatter_color == "r":
                                c_val.append(float(r_by_idx.get(idx, float("nan"))))
                            elif per_layer_scatter_color == "p":
                                c_val.append(float(p_by_idx.get(idx, float("nan"))))
                            else:
                                raise ValueError(
                                    "per_layer_scatter_color must be one of: 'layer', 'logp', 'r', 'p'"
                                )
                            n_points_total += 1

                    if n_points_total == 0:
                        print(
                            "Per-layer latencyΔ vs cos-dist plot: could not generate any points "
                            f"(n_sig_idx={sig_ix.size})."
                        )
                    else:
                        fig_sc, ax_sc = plt.subplots(figsize=(6.8, 5.5))
                        x_sc = np.asarray(x_lat_delta, dtype=float)
                        y_sc = np.asarray(y_cos_dist, dtype=float)
                        c_sc = np.asarray(c_val, dtype=float)

                        # Correlation for the scatter (Pearson r and p-value)
                        r_xy, p_xy = _pearsonr_safe(x_sc, y_sc)

                        sc = ax_sc.scatter(
                            x_sc,
                            y_sc,
                            c=c_sc,
                            cmap=(
                                "viridis"
                                if per_layer_scatter_color in {"layer", "logp", "p"}
                                else "coolwarm"
                            ),
                            s=14,
                            alpha=0.65,
                            linewidths=0,
                        )

                        # Fit + draw a simple linear trend line on the scatter (cos-dist ~ latencyΔ)
                        # (This is purely descriptive; correlation stats are shown separately.)
                        if x_sc.size >= 2 and float(np.nanstd(x_sc)) > 0 and float(np.nanstd(y_sc)) > 0:
                            try:
                                lr_sc = linregress(x_sc, y_sc)
                                xline = np.linspace(float(np.min(x_sc)), float(np.max(x_sc)), 100)
                                yline = lr_sc.slope * xline + lr_sc.intercept
                                ax_sc.plot(xline, yline, linestyle=":", linewidth=2, color="black", alpha=0.85)
                            except Exception:
                                pass
                        cbar = fig_sc.colorbar(sc, ax=ax_sc)
                        if per_layer_scatter_color == "layer":
                            cbar.set_label("Layer")
                        elif per_layer_scatter_color == "logp":
                            cbar.set_label("-log(pval) for best hit in that layer")
                        elif per_layer_scatter_color == "r":
                            cbar.set_label("Per-index latency-vs-layer regression Pearson r")
                        elif per_layer_scatter_color == "p":
                            cbar.set_label("Per-index latency-vs-layer regression p-value")

                        ax_sc.set_xlabel("Latency difference (ms): layer - first significant layer")
                        ax_sc.set_ylabel("Cosine distance to first significant layer activation")
                        ax_sc.set_title(
                            f"LatencyΔ vs activation change (per-layer points)\n"
                            f"indices: p<=0.05 (n_idx={sig_ix.size}, n_pts={n_points_total})\n"
                            f"color: {per_layer_scatter_color}"
                        )
                        if np.isfinite(r_xy) and np.isfinite(p_xy):
                            ax_sc.text(
                                0.02,
                                0.98,
                                f"Pearson r = {r_xy:.3g}\np = {p_xy:.3g}",
                                transform=ax_sc.transAxes,
                                va="top",
                                ha="left",
                                fontsize=9,
                                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
                            )
                        ax_sc.grid(True, alpha=0.25)
                        fig_sc.tight_layout()

                        # Keep the legacy filename for the default coloring to avoid breaking existing scripts.
                        if per_layer_scatter_color == "layer":
                            save_loc_sc = output_dir / f"{dataset}_latency_delta_vs_activation_cosdist.png"
                        else:
                            save_loc_sc = (
                                output_dir
                                / f"{dataset}_latency_delta_vs_activation_cosdist_color-{per_layer_scatter_color}.png"
                            )
                        fig_sc.savefig(save_loc_sc, dpi=300)
                        plt.close(fig_sc)

                        if missing_idx:
                            print(
                                f"Per-layer latencyΔ vs cos-dist plot: skipped {len(missing_idx)} indices due to missing/invalid data."
                            )
        else:
            idx_int = int(idx_to_plot)

            out = _per_index_series_and_regression(idx_int)
            if out is None:
                print(f"No significant entries found for neuron index {idx_int}.")
            else:
                layers_used, lat_used, logp_used, slope, intercept, r, p = out

                fig3, ax3 = plt.subplots()
                sc3 = ax3.scatter(
                lat_used,
                layers_used,
                c=logp_used,
                cmap="turbo",
                norm=logp_norm,
                s=18,
                alpha=0.9,
                linewidths=0,
                label=f"Neuron index {idx_int}",
                )

                # Colorbar legend for significance (-log(pval))
                cbar3 = fig3.colorbar(sc3, ax=ax3)
                cbar3.set_label("-log(pval)")

                ax3.set_xlim(-250, 850)
                ax3.set_xlabel("Latency (ms)")
                ax3.set_ylabel("Layer number")
                ax3.set_ylim(-1, layer)
                ax3.set_title(f"Neuron index {idx_int}: latency vs layer")

                # Linear regression (same style as `_plot_line_of_best_fit`):
                # fit latency = slope*layer + intercept, then plot predicted latency across all layers.
                layers_all = np.arange(layer)
                if layers_used.size >= 2:
                    x_pred = slope * layers_all + intercept
                    ax3.plot(x_pred, layers_all, linestyle=":", linewidth=2, color="black")
                    ax3.text(
                        0.02,
                        0.98,
                        f"Pearson r = {r:.3g}\np = {p:.3g}",
                        transform=ax3.transAxes,
                        va="top",
                        ha="left",
                        fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
                    )
                else:
                    ax3.text(
                        0.02,
                        0.98,
                        "Not enough layers with data for regression",
                        transform=ax3.transAxes,
                        va="top",
                        ha="left",
                        fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
                    )

                plt.tight_layout()
                save_dir3 = output_dir / f"{idx_int}_layer"
                save_dir3.mkdir(parents=True, exist_ok=True)
                save_loc3 = save_dir3 / f"{dataset}_neuron_{idx_int}_latency_vs_layer.png"
                fig3.savefig(save_loc3, dpi=600)
                plt.close(fig3)
        
            # Optional: plot activation contours over time for this fixed neuron index across layers
            layers_to_plot = list(range(layer))  # or e.g. [0, 5, 10, 15, 20, 25, 29]
            pt_template = (
                "/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/"
                "predicted_function_contours/asr_models/qwen/decoder_text/segment_8_layer{i}.pt"
            )

            # Quantify how the activation contour changes across layers.
            # For each layer we extract y_act(t) = activation[0, :, idx_int] as a vector over time.
            # Then we compute:
            #   - cosine distance to layer-0 (shape change, scale-invariant)
            #   - Pearson correlation to layer-0 (shape similarity)
            #   - L2 norm of y_act per layer (overall magnitude)
            #   - L2 norm of delta to previous layer (how much it changes step-to-step)
            # And we also plot a layers x time heatmap for a compact overview.
            y_by_layer: list[np.ndarray] = []
            valid_layers: list[int] = []
            missing_layers: list[int] = []

            for li in layers_to_plot:
                pt_path = pt_template.format(i=li)
                y_act = _load_activation_trace(pt_path, idx_int)
                if y_act is None:
                    missing_layers.append(li)
                    continue
                y_by_layer.append(y_act)
                valid_layers.append(li)

            if len(y_by_layer) >= 2:
                Y = np.stack(y_by_layer, axis=0)  # (n_layers_valid, T)
                sig_idx = sig[np.asarray(sig[:, 4], dtype=int) == int(idx_to_plot)]
                Y = Y[max(1, int(np.min(sig_idx[:, -2]))) : min(int(np.max(sig_idx[:, -2])), Y.shape[0])]  # focus on layers with significant hits for this index
                valid_layers = valid_layers[max(1, int(np.min(sig_idx[:, -2]))) : min(int(np.max(sig_idx[:, -2])), len(valid_layers))]

                # Reference: first valid layer (usually layer 0)
                y0 = Y[0]
                y0c = y0 - float(np.mean(y0))
                y0_norm = float(np.linalg.norm(y0))
                y0c_norm = float(np.linalg.norm(y0c))

                def _corr_to_ref(a: np.ndarray, ref_c: np.ndarray, ref_c_norm: float, eps: float = 1e-12) -> float:
                    ac = a - float(np.mean(a))
                    an = float(np.linalg.norm(ac))
                    if an <= eps or ref_c_norm <= eps:
                        return float("nan")
                    return float(np.dot(ac, ref_c) / (an * ref_c_norm))

                cos_dist0 = np.asarray([_cosine_distance(y, y0) for y in Y], dtype=float)
                corr0 = np.asarray([_corr_to_ref(y, y0c, y0c_norm) for y in Y], dtype=float)
                l2_norm = np.linalg.norm(Y, axis=1).astype(float)
                delta_l2 = np.full_like(l2_norm, np.nan, dtype=float)
                if Y.shape[0] >= 2:
                    delta_l2[1:] = np.linalg.norm(Y[1:] - Y[:-1], axis=1).astype(float)

                figm, axm = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharex=True)
                axm = np.asarray(axm).ravel()

                axm[0].plot(valid_layers, cos_dist0, marker="o", linewidth=1.5)
                axm[0].set_ylabel("Cosine distance to ref")
                axm[0].set_title("Shape change vs reference layer")
                axm[0].grid(True, alpha=0.25)

                axm[1].plot(valid_layers, corr0, marker="o", linewidth=1.5)
                axm[1].set_ylabel("Pearson corr to ref")
                axm[1].set_title("Shape similarity vs reference layer")
                axm[1].grid(True, alpha=0.25)

                axm[2].plot(valid_layers, l2_norm, marker="o", linewidth=1.5)
                axm[2].set_ylabel("L2 norm")
                axm[2].set_title("Magnitude")
                axm[2].grid(True, alpha=0.25)

                axm[3].plot(valid_layers, delta_l2, marker="o", linewidth=1.5)
                axm[3].set_ylabel(r"$||y_{\ell}-y_{\ell-1}||_2$")
                axm[3].set_title("Step-to-step change")
                axm[3].grid(True, alpha=0.25)

                for a in axm:
                    a.set_xlabel("Layer")

                figm.suptitle(f"Activation change metrics for index {idx_int}", y=1.02)
                figm.tight_layout()

                save_locm = (
                    output_dir
                    / f"{idx_int}_layer"
                    / f"segment_8_activation_change_metrics_index_{idx_int}.png"
                )
                figm.savefig(save_locm, dpi=300)
                plt.close(figm)

                # Heatmap: layer x time (use only valid layers)
                figh, axh = plt.subplots(figsize=(10, 4))
                im = axh.imshow(
                    Y,
                    aspect="auto",
                    interpolation="nearest",
                    cmap="viridis",
                )
                axh.set_xlabel("Time (t)")
                axh.set_ylabel("Layer (valid subset)")
                axh.set_title(f"Activation heatmap for index {idx_int} (layers={len(valid_layers)})")

                # Put actual layer numbers on y-axis (subsample if too many)
                if len(valid_layers) <= 16:
                    axh.set_yticks(np.arange(len(valid_layers)))
                    axh.set_yticklabels([str(v) for v in valid_layers])
                else:
                    step = max(1, len(valid_layers) // 12)
                    yt = np.arange(0, len(valid_layers), step)
                    axh.set_yticks(yt)
                    axh.set_yticklabels([str(valid_layers[i]) for i in yt])

                cbar = figh.colorbar(im, ax=axh)
                cbar.set_label("Activation")
                figh.tight_layout()

                save_loch = (
                    output_dir
                    / f"{idx_int}_layer"
                    / f"segment_8_activation_heatmap_index_{idx_int}.png"
                )
                figh.savefig(save_loch, dpi=300)
                plt.close(figh)

                if missing_layers:
                    print(
                        f"Activation contour metrics: skipped {len(missing_layers)} layers due to missing/invalid .pt files: "
                        f"{missing_layers[:10]}" + ("..." if len(missing_layers) > 10 else "")
                    )
            else:
                print(
                    f"Activation contour metrics: not enough valid layers to compute changes "
                    f"(valid={len(y_by_layer)}, missing={len(missing_layers)})."
                )

            ncols = 6
            nrows = int(np.ceil(len(layers_to_plot) / ncols))
            fig2, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(3.2 * ncols, 2.4 * nrows),
                sharex=True,
            )
            axes = np.atleast_1d(axes).ravel()

            for ax_i, li in enumerate(layers_to_plot):
                ax = axes[ax_i]
                pt_path = pt_template.format(i=li)
                try:
                    t = torch.load(pt_path, map_location="cpu")
                except FileNotFoundError:
                    ax.set_title(f"Layer {li} (missing)", fontsize=10)
                    ax.axis("off")
                    continue

                if hasattr(t, "detach"):
                    t = t.detach()
                t_np = np.asarray(t)

                if t_np.ndim != 3 or t_np.shape[0] != 1 or t_np.shape[1] != 115:
                    raise ValueError(f"Unexpected tensor shape in {pt_path}: {t_np.shape}")

                if idx_int < 0 or idx_int >= t_np.shape[2]:
                    raise IndexError(
                        f"Index {idx_int} out of bounds for last dim {t_np.shape[2]} in {pt_path}"
                    )

                y_act = t_np[0, :, idx_int].astype(float)

                x_time = np.arange(y_act.shape[0])
                ax.plot(x_time, y_act, linewidth=1.2, color="black")
                ax.set_title(f"Layer {li}", fontsize=10)
                ax.grid(True, alpha=0.25)

            for j in range(len(layers_to_plot), len(axes)):
                axes[j].axis("off")

            fig2.supxlabel("Time (t)")
            fig2.supylabel(f"Activation [0, :, {idx_int}]")
            fig2.tight_layout()

            save_loc2 = (
                output_dir
                / f"{idx_int}_layer"
                / f"segment_8_activation_index_{idx_int}_by_layer.png"
            )
            fig2.savefig(save_loc2, dpi=300)
            plt.close(fig2)

    # NOTE: keep non-interactive for batch runs

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
    # Save significant (layer, neuron) pairs to .npy
    # sig columns: [peak_lat, sensor_ind, logp, layer_no, neuron_no]
    layer_neuron_pairs = sig[:, [3, 4]].astype(int)
    import ipdb; ipdb.set_trace()

    save_path = Path(
        "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/"
        "kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/sig_neurons.npy"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, layer_neuron_pairs)

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
                               min_count_for_average=6)


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

        # # --- Residual plot: (observed mean latency - fitted mean latency) per layer ---
        # # Use only layers that contributed to the regression fit (mask).
        # fitted_mean_lat_by_layer = lr.slope * x_fit + lr.intercept
        # residual_by_layer = y_fit - fitted_mean_lat_by_layer

        # fig_res, ax_res = plt.subplots()
        # ax_res.axhline(0.0, linestyle=":", linewidth=1.5, color="black", alpha=0.7)
        # ax_res.scatter(
        #     residual_by_layer,
        #     x_fit,
        #     c=n_sig_by_layer[mask],
        #     cmap="turbo",
        #     norm=count_norm,
        #     marker="o",
        #     s=25,
        #     edgecolors="black",
        #     linewidths=0.3,
        # )
        # ax_res.plot(residual_by_layer, x_fit, linestyle="-", linewidth=1.0, color="black", alpha=0.6)

        # # Reuse the same r/p reporting to contextualize the residuals.
        # r = float(lr.rvalue)
        # p = float(lr.pvalue)
        # ax_res.text(
        #     0.02,
        #     0.98,
        #     f"Residuals vs layer\nPearson r = {r:.3g}\np = {p:.3g}",
        #     transform=ax_res.transAxes,
        #     va="top",
        #     ha="left",
        #     fontsize=9,
        #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
        # )

        # ax_res.set_xlabel("Residual mean latency (ms): observed - fitted")
        # ax_res.set_ylabel("Layer number")
        # ax_res.set_ylim(-1, layer)
        # plt.tight_layout()

        # save_loc_res = output_dir / f"{dataset}_best_fit_residuals.png"
        # plt.savefig(save_loc_res, dpi=600)
        # plt.close(fig_res)

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
    parser.add_argument(
        '--per-layer-scatter-color',
        choices=['layer', 'logp', 'r', 'p'],
        default='layer',
        help=(
            "Coloring for the per-layer latencyΔ vs activation-change scatter: "
            "'layer' (default), 'logp' (-log(pval) of the best hit in that layer), "
            "'r' (per-index latency-vs-layer regression Pearson r), or 'p' (per-index regression p-value)."
        ),
    )
    args = parser.parse_args()

    neuron_scatter(
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        x_axis=args.x_axis,
        dataset=args.dataset,
        per_layer_scatter_color=args.per_layer_scatter_color,
    )
