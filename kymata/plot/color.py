from matplotlib.colors import to_rgb, to_hex
from numpy import linspace


def gradient_color_dict(functions: list[str], start_color, stop_color) -> dict[str, str]:
    """
    Given a set of function names, and two colours (any matplotlib format), returns a dict mapping the function names
    to a linear gradient between the two colours.
    The dict is compatible with expression_plot colour specifications.
    """
    start_rgb = to_rgb(start_color)
    stop_rgb = to_rgb(stop_color)

    return {
        function: to_hex((
            linspace(start_rgb[0], stop_rgb[0], len(functions))[i],  # R
            linspace(start_rgb[1], stop_rgb[1], len(functions))[i],  # G
            linspace(start_rgb[2], stop_rgb[2], len(functions))[i],  # B
        ))
        for i, function in enumerate(functions)
    }


def constant_color_dict(functions: list[str], color) -> dict[str, str]:
    """
    Given a set of function names, and a colour (any matplotlib format), returns a dict mapping the function names
    all to the same colour.
    The dict is compatible with expression_plot colour specifications.
    """
    return gradient_color_dict(functions, color, color)
