from matplotlib.colors import to_rgb, to_hex, LinearSegmentedColormap
from numpy import linspace

transparent = (0, 0, 0, 0)


def gradient_color_dict(transforms: list[str], start_color, stop_color) -> dict[str, str]:
    """
    Generates a dictionary mapping transform names to colors forming a linear gradient.

    Parameters:
    -----------
    transforms : list[str]
        A list of transform names to be assigned colors.
    start_color : str or tuple
        The starting color of the gradient. Can be any format supported by matplotlib (e.g., color name, hex string, RGB tuple).
    stop_color : str or tuple
        The ending color of the gradient. Can be any format supported by matplotlib.

    Returns:
    --------
    dict[str, str]
        A dictionary where each key is a transform name from the input list and each value is a color in hex format.
        The colors form a linear gradient between the start_color and stop_color.

    Notes:
    ------
    The returned dictionary is compatible with color specifications for expression plots.
    """
    start_rgb = to_rgb(start_color)
    stop_rgb = to_rgb(stop_color)

    return {
        transform: to_hex(
            (
                linspace(start_rgb[0], stop_rgb[0], len(transforms))[i],  # R
                linspace(start_rgb[1], stop_rgb[1], len(transforms))[i],  # G
                linspace(start_rgb[2], stop_rgb[2], len(transforms))[i],  # B
            )
        )
        for i, transform in enumerate(transforms)
    }


def constant_color_dict(transforms: list[str], color) -> dict[str, str]:
    """
    Generates a dictionary mapping transform names to a single specified color.

    Parameters:
    -----------
    transforms : list[str]
        A list of transform names to be assigned the same color.
    color : str or tuple
        The color to assign to all transform names. Can be any format supported by matplotlib (e.g., color name, hex string, RGB tuple).

    Returns:
    --------
    dict[str, str]
        A dictionary where each key is a transform name from the input list and each value is the specified color in hex format.

    Notes:
    ------
    The returned dictionary is compatible with color specifications for expression plots.
    """
    return gradient_color_dict(transforms, color, color)


def anchored_gradient_colormap(name: str, anchor_values: list[float | int], colors: list):
    """
    Creates a colormap which is a stack of unequal-height gradients with segment endpoints at specific values.

    Args:
        name: The name of the colormap
        anchor_values: List of anchor values for gradient segment boundaries (including first and last endpoints)
        colors: Colours for each boundary point

    Raises:
        ValueError if `layer_anchors` and `colors` aren't of the same length.

    Returns:
        A LinearSegmentedColormap
    """
    if len(anchor_values) != len(colors):
        raise ValueError("Specify one color exactly for each layer anchor")

    # Convert layer anchors to value ranges
    vmin, vmax = anchor_values[0], anchor_values[-1]
    positions = [(v - vmin) / (vmax - vmin) for v in anchor_values]

    return LinearSegmentedColormap.from_list(name, list(zip(positions, colors)))
