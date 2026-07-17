from logging import basicConfig, INFO
from pathlib import Path
import os
from os import path
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap, to_hex

from kymata.datasets.data_root import data_root_path
from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set
from kymata.plot.expression import expression_plot, legend_display_dict
from kymata.plot.color import constant_color_dict, gradient_color_dict, anchored_gradient_colormap

import time
from tqdm import tqdm


_LAYER_RE = re.compile(r"(?:^|[^0-9])layer\s*([0-9]{1,3})(?:[^0-9]|$)", re.IGNORECASE)


def _extract_layer_index(transform_name: str) -> int | None:
    """Extract a layer index from a transform string.

    Expected formats include e.g. "layer0", "layer_12", "...layer 28...".
    Returns None if no layer index is found.
    """
    m = _LAYER_RE.search(transform_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


meg_cmap = anchored_gradient_colormap("MEG Reds",   anchor_values=[0, 10, 28, 32], colors=["#F38B85", "#E62C46", "#83082C", "#57051D"])
eeg_cmap = anchored_gradient_colormap("MEG Greens", anchor_values=[0, 10, 28, 32], colors=["#7BC682", "#3EA341", "#277C16", "#1D650F"])


def _get_color_dict(cmap, transform_names: list[str], layer_minmax_for_cmap: tuple[int, int]):
    lmin, lmax = layer_minmax_for_cmap
    if lmax <= lmin:
        raise ValueError("Layer range must be strictly increasing")
    layer_idxs: list[tuple[str, int]] = []
    for t in transform_names:
        layer_idx = _extract_layer_index(t)
        if layer_idx is None:
            raise ValueError("Could not determine layer number")
        layer_idxs.append((t, layer_idx))

    d = dict()
    for layer_name, layer_idx in layer_idxs:
        if layer_idx < lmin:
            cmap_value = 0.0
        elif layer_idx > lmax:
            cmap_value = 1.0
        else:
            cmap_value = (layer_idx - lmin) / (lmax - lmin)
        d[layer_name] = cmap(cmap_value)
    return d


def red_gradient_color_dict(
    transform_names,
    *,
    # n_layers: int = 29,
    n_layers: int = 32,
    cmap_name: str = "Reds",
    min_intensity: float = 0.1, # 0.3
    # max_intensity: float = 0.86,
    max_intensity: float = 0.92,
):
    """Generate a constant-color mapping for expression_plot using a red gradient per layer.

    Assumptions:
    - `transform_names` contains strings with embedded layer info (layer0..layer28).
    - If a layer can't be parsed, it falls back to mid-red.

    Returns a dict-like mapping from transform name -> hex color string.
    """
    # Matplotlib colormap gives perceptually ordered reds.
    cmap = plt.get_cmap(cmap_name)
    # Precompute colors for each layer index.
    # Clamp intensity bounds to [0, 1] and keep enough contrast.
    lo = float(np.clip(min_intensity, 0.0, 1.0))
    hi = float(np.clip(max_intensity, 0.0, 1.0))
    if hi < lo:
        lo, hi = hi, lo
    layer_ts = np.linspace(lo, hi, n_layers)
    layer_colors = [to_hex(cmap(t)) for t in layer_ts]

    # Build mapping for each transform.
    fallback = to_hex(cmap((lo + hi) / 2))
    out = {}
    for name in transform_names:
        layer = _extract_layer_index(str(name))
        if layer is None:
            out[name] = fallback
        else:
            # Robust to out-of-range layers.
            layer = int(np.clip(layer, 0, n_layers - 1))
            out[name] = layer_colors[layer]
    return out

def add_layer_colorbar(
    fig,
    *,
    n_layers: int = 29,
    cmap_name: str = "Reds",
    min_intensity: float = 0.1, # 0.3
    max_intensity: float = 0.86,
    label: str = "Layer",
):
    """Attach a layer-index colorbar to an existing figure.

    Uses the same colormap/intensity range as `red_gradient_color_dict`.
    """
    lo = float(np.clip(min_intensity, 0.0, 1.0))
    hi = float(np.clip(max_intensity, 0.0, 1.0))
    if hi < lo:
        lo, hi = hi, lo

    # Use a truncated version of the colormap so the colorbar matches the
    # intensities used by `red_gradient_color_dict`.
    base_cmap = plt.get_cmap(cmap_name)
    cmap = LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc_{lo:.2f}_{hi:.2f}",
        base_cmap(np.linspace(lo, hi, 256)),
    )
    norm = Normalize(vmin=0, vmax=n_layers - 1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(np.linspace(0, n_layers - 1, n_layers))

    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.035, pad=0.02)
    cbar.set_label(label)
    # Keep ticks readable; show 0, 4, 8, ..., 28 by default.
    step = 4
    ticks = list(range(0, n_layers, step))
    if (n_layers - 1) not in ticks:
        ticks.append(n_layers - 1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])
    return cbar

def load_all_expression_data(base_folder):
    expression_data = None
    # Loop through each subdirectory inside the base folder
    for subdir in os.listdir(base_folder):
        subdir_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subdir_path):  # Ensure we are processing directories
            # List all .nkg files inside the subdirectory
            nkg_files = tqdm([f for f in os.listdir(subdir_path) if f.endswith('.nkg')])
            for nkg_file in nkg_files:
                file_path = os.path.join(subdir_path, nkg_file)
                nkg_files.set_description(f"Loading {file_path}")
                if expression_data is None:
                    # Load the first .nkg file
                    expression_data = load_expression_set(file_path)
                else:
                    # Add data from subsequent .nkg files
                    expression_data += load_expression_set(file_path)
    return expression_data

def load_part_of_expression_data(base_folder, pick):
    expression_data = None
    # Loop through each subdirectory inside the base folder
    for subdir in os.listdir(base_folder):
        subdir_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subdir_path) and [int(x) for x in subdir.split('_')] in pick.tolist():  # Ensure we are processing directories
            # List all .nkg files inside the subdirectory
            nkg_files = tqdm([f for f in os.listdir(subdir_path) if f.endswith('.nkg')])
            for nkg_file in tqdm(nkg_files):
                file_path = os.path.join(subdir_path, nkg_file)
                nkg_files.set_description(f"Loading {file_path} (pick)")
                if expression_data is None:
                    # Load the first .nkg file
                    expression_data = load_expression_set(file_path)
                else:
                    # Add data from subsequent .nkg files
                    expression_data += load_expression_set(file_path)
    return expression_data

def main():

    transform_family_type = 'by_layer'
    path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-core", "kymata-core-data", "output")
    # path_to_nkg_files = '/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output'

    # template invoker for printing out expression set .nkgs

    save_loc = data_root_path()

    if transform_family_type == 'simple':

        expression_data_qwen_decoder = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_eeg/decoder/expression')
        decoder_name = expression_data_qwen_decoder.transforms
        fig = expression_plot(expression_data_qwen_decoder, paired_axes=True, minimap='large', show_legend=True,
                                color=constant_color_dict(decoder_name, color='red'),
                                legend_display=legend_display_dict(decoder_name, 'QWEN decoder features'))
        fig.savefig(save_loc / "output/qwen_english_eeg/decoder/qwen_eeg_decoder_source.png")

    elif transform_family_type == 'by_layer':

 
        # expression_data_qwen_encoder = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english/source/encoder/expression')
        # encoder_name = expression_data_qwen_encoder.transforms
        expression_data_qwen_encoder = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_meg/encoder/expression')
        encoder_name = expression_data_qwen_encoder.transforms

        # expression_data_salmonn_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/phone_source')
        # expression_data_salmonn_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/single_neuron_phone')
        # phone_name = expression_data_salmonn_phone.transforms
        # expression_data_tvl = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/english_TVL_family_source_baseline_derangments_6.nkg')
        # tvl_name = expression_data_tvl.transforms
        # IL_name = [i for i in tvl_name if i != 'STL']
        # STL_name = ['STL']
        fig = expression_plot(expression_data_qwen_encoder, paired_axes=True, minimap='large', show_legend=True,
                                color=_get_color_dict(meg_cmap, encoder_name, (0, 32)),
                                    # | constant_color_dict(tvl_name, color= 'yellow')
                                    # | constant_color_dict(IL_name, color= 'purple')
                                    # | constant_color_dict(STL_name, color= 'pink'),
                                    # | constant_color_dict(phone_name, color='green'),
                                legend_display=legend_display_dict(encoder_name, 'QWEN encoder features'))
                                #     | legend_display_dict(decoder_name, 'QWEN decoder features'))
                                #     # | legend_display_dict(tvl_name, 'TVL transforms')
                                    # | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
                                    # | legend_display_dict(STL_name, 'Short Term Loudness transform'))
                                    # | legend_display_dict(phone_name, 'SALMONN phone features'))
        # fig = expression_plot(expression_data_tvl[40:55], paired_axes=True, minimap=False, show_legend=True)
        # fig = expression_plot(expression_data_tvl, paired_axes=True, minimap=False, show_legend=True,
        #                         color=constant_color_dict(IL_name, color= 'purple')
        #                             | constant_color_dict(STL_name, color= 'pink'),
        #                         legend_display=legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
        #                             | legend_display_dict(STL_name, 'Short Term Loudness transform'))

    add_layer_colorbar(fig, n_layers=32, label="QWEN layer")
    fig.savefig(save_loc / "output/qwen_english_meg/encoder/qwen_meg_encoder_source_by_layer_new_scale.png")

if __name__ == '__main__':
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    main()
