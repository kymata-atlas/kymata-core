from collections import defaultdict, Counter
from typing import Optional, NamedTuple

import matplotlib.patheffects as pe
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from networkx.classes import DiGraph
from networkx.relabel import relabel_nodes
from numpy.typing import NDArray
from scipy.interpolate import splev

from kymata.entities.expression import ExpressionSet
from kymata.ippm.graph import IPPMGraph, IPPMConnectionStyle, IPPMNode
from kymata.ippm.ippm import IPPM


class _YOrdinateStyle:
    """
    Enumeration for Y-ordinate plotting styles.

    Attributes:
        progressive: Points are plotted with increasing y ordinates.
        centered: Points are vertically centered.
    """
    progressive = "progressive"
    centered = "centered"


_XY = tuple[float, float]


class _PlottableNode(NamedTuple):
    x: float
    y: float
    label: str
    color: str
    size: float
    is_input: bool
    is_terminal: bool


class _PlottableIPPMGraph:
    """
    An extension of an `IPPMGraph` that associates coordinates, colors, annotations, etc. attached to each node in the
    graph. This enables visual representation of the IPPMGraph with customizable layout and appearance.

    Attributes:
        graph (DiGraph): A directed graph with additional attributes for visualization, such as node positions and
            colors.
    """

    def __init__(self,
                 ippm_graph: IPPMGraph,
                 colors: dict[str, str],
                 y_ordinate_style: str,
                 avoid_collinearity: bool,
                 scale_nodes: bool,
                 connection_style: IPPMConnectionStyle,
                 test_hemisphere_colors: bool,
                 ):
        """
        Initializes the _PlottableIPPMGraph by assigning coordinates, colors, and other visual attributes to each node.

        Args:
            ippm_graph (IPPMGraph): The source IPPMGraph to be visualized.
            colors (dict[str, str]): A dictionary mapping transform names to colors for node visualization.
            y_ordinate_style (str): Defines the style for the vertical positioning of nodes.
                Options include "progressive" or "centered" (members of `_YOrdinateStyle`).
            avoid_collinearity (bool): Whether to apply a small offset to avoid collinearity between nodes in the same
                serial step.
            scale_nodes (bool): Whether to scale node sizes based on the significance of the corresponding expression
                points.
            test_hemisphere_colors (bool): If True, overrides the `colors` to use predefined hemisphere colors.
        """

        # Vertical spacing between nodes
        node_spacing = 1

        # Hemispheres present in the graph
        hemispheres: set[str] = {
            node.hemisphere
            for node in ippm_graph.graph_full.nodes
        }

        # hemisphere → {transforms represented in this hemisphere}
        transforms_per_hemisphere: dict[str, list[str]] = {
            hemisphere: sorted({
                node.transform
                for node in ippm_graph.graph_full.nodes
                if node.hemisphere == hemisphere
            }, reverse=True)
            for hemisphere in hemispheres
        }

        # hemisphere → transform → y-ordinate
        y_ordinates: dict[str, dict[str, float]]
        if y_ordinate_style == _YOrdinateStyle.progressive:
            y_ordinates = _all_y_ordinates_progressive(ippm_graph, transforms_per_hemisphere, node_spacing)

        elif y_ordinate_style == _YOrdinateStyle.centered:
            y_ordinates = _all_y_ordinates_centered(ippm_graph, transforms_per_hemisphere, node_spacing, avoid_collinearity)
        else:
            raise NotImplementedError()

        # Apply an offset for each hemisphere.
        # Shift up by the maximum plotted height of all hemispheres.
        # (It's `node_spacing +` so that there is clear separation between the hemispheres; i.e. the last node of one does
        # not overlap with the first node of the next)
        hemisphere_separation_offset = node_spacing + max(
            y_ordinates[hemisphere][transform]
            for hemisphere, transforms in transforms_per_hemisphere.items()
            for transform in transforms
        )
        for hemisphere_i, hemisphere in enumerate(hemispheres):
            for transform in transforms_per_hemisphere[hemisphere]:
                y_ordinates[hemisphere][transform] += hemisphere_separation_offset * hemisphere_i

        # Apply y-ordinates to desired graph

        if connection_style == IPPMConnectionStyle.last_to_first:
            preferred_graph = ippm_graph.graph_last_to_first
        elif connection_style == IPPMConnectionStyle.first_to_first:
            preferred_graph = ippm_graph.graph_first_to_first
        else:
            # IPPMConnectionStyle.full typically has too many nodes and edges to be performant and should be avoided
            raise NotImplementedError()

        # Update this section to use IPPMNode attributes
        node: IPPMNode
        self.graph: DiGraph = relabel_nodes(
            preferred_graph,
            {
                node: _PlottableNode(
                    label=node.transform,
                    x=node.latency,
                    y=y_ordinates[node.hemisphere][node.transform],
                    # Conditionally set color based on test_hemisphere_colors flag
                    color=_get_test_hemisphere_color(node.hemisphere) if test_hemisphere_colors else colors[node.transform],
                    is_input=node.is_input,
                    is_terminal=node.transform in ippm_graph.terminals,
                    size=-1 * node.logp_value if scale_nodes else 150,
                )
                for node in preferred_graph.nodes
            }
        )

        pass


def _all_y_ordinates_progressive(ippm_graph: IPPMGraph,
                                 transforms_per_hemisphere: dict[str, list[str]],
                                 node_spacing: float,
                                 ) -> dict[str, dict[str, float]]:
    hemispheres = set(transforms_per_hemisphere.keys())
    y_ordinates: dict[str, dict[str, float]] = defaultdict(dict)

    inputs_per_hemisphere = {
        hemisphere: sorted({
            node.transform
            for node in ippm_graph.graph_full.nodes
            if node.hemisphere == hemisphere and node.is_input
        }, reverse=True)
        for hemisphere in hemispheres
    }
    for hemisphere in hemispheres:
        # Stack inputs
        partition_ptr = 0
        for input_transform in inputs_per_hemisphere[hemisphere]:
            y_ordinates[hemisphere][input_transform] = _get_y_coordinate_progressive(
                partition_number=partition_ptr,
                spacing=node_spacing)
            partition_ptr += 1

        # Stack others
        partition_ptr = 0
        # Non-input transforms
        other_transforms = [t for t in transforms_per_hemisphere[hemisphere] if t not in inputs_per_hemisphere[hemisphere]]

        # Build dictionary mapping function names to sequence steps
        serial_sequence = ippm_graph.candidate_transform_list.serial_sequence
        # transform → sequence_step_idx
        step_idxs: dict[str, int] = dict()
        for step_idx, step in enumerate(serial_sequence):
            for transform in step:
                step_idxs[transform] = step_idx

        # Sort by serial sequence, then alphabetically
        # Do both sorts in reverse order
        other_transforms.sort(reverse=True)
        other_transforms.sort(key=lambda trans: step_idxs[trans], reverse=True)

        for transform in other_transforms:
            y_ordinates[hemisphere][transform] = _get_y_coordinate_progressive(
                partition_number=partition_ptr,
                spacing=node_spacing)
            partition_ptr += 1

    return y_ordinates


def _all_y_ordinates_centered(ippm_graph: IPPMGraph,
                              transforms_per_hemisphere: dict[str, list[str]],
                              node_spacing: float,
                              avoid_collinearity: bool,
                              ) -> dict[str, dict[str, float]]:
    hemispheres = set(transforms_per_hemisphere.keys())
    y_ordinates: dict[str, dict[str, float]] = defaultdict(dict)
    # Build dictionary mapping function names to sequence steps
    serial_sequence = ippm_graph.candidate_transform_list.serial_sequence
    # transform → sequence_step_idx
    step_idxs: dict[str, int] = dict()
    for step_idx, step in enumerate(serial_sequence):
        for transform in step:
            step_idxs[transform] = step_idx
    # Count how "wide" each step in the serial sequence is
    # hemisphere → serial step idx → number of "parallel" transforms this step which are present in this hemisphere
    totals_within_serial_step = {
        hemisphere: Counter(step_idx
                       for trans, step_idx in step_idxs.items()
                       if trans in transforms_per_hemisphere[hemisphere])
        for hemisphere in hemispheres
    }
    # hemisphere → max "width" over the whole set of nodes this hemisphere
    max_transform_counts = {
        hemisphere: max(totals_within_serial_step[hemisphere].values())
        for hemisphere in hemispheres
    }
    for hemisphere in hemispheres:
        # Calculate totals and max totals for each hemisphere independently
        # This ensures y-coordinates are calculated relative to within their hemisphere
        temp_step_counts = defaultdict(set)
        # Within each serial step, we have a collection of transforms. This dictionary comprises a counter for each
        # of these steps, which increments for each successive transform in that level
        # step_idx → counter
        idxs_within_level: dict[int, int] = defaultdict(int)
        for transform in transforms_per_hemisphere[hemisphere]:
            step_idx = step_idxs[transform]
            y_ordinates[hemisphere][transform] = _get_y_coordinate_centered(
                function_idx_within_level=idxs_within_level[step_idx],
                function_total_within_level=totals_within_serial_step[hemisphere][step_idx],
                max_function_total_within_level=max_transform_counts[hemisphere],
                positive_nudge_frac=(step_idx / len(serial_sequence)
                                     if avoid_collinearity
                                     else 0),
                spacing=node_spacing
            )
            idxs_within_level[step_idx] += 1
    return y_ordinates


def _get_test_hemisphere_color(hemisphere: str) -> str:
    """Helper function to return a specific color based on hemisphere for testing."""
    if hemisphere == "LH":
        return "red"
    elif hemisphere == "RH":
        return "blue"
    else: # For SCALP or other
        return "green"


def plot_ippm(
        ippm: IPPM,
        colors: dict[str, str],
        title: Optional[str] = None,
        xlims_s: tuple[Optional[float], Optional[float]] = (None, None),
        y_ordinate_style: str = _YOrdinateStyle.centered,
        connection_style: str = IPPMConnectionStyle.last_to_first,
        scale_nodes: bool = False,
        figheight: int = 5,
        figwidth: int = 10,
        arrowhead_dims: tuple[float, float] = None,
        linewidth: float = 3,
        show_labels: bool = True,
        relabel: dict[str, str] | None = None,
        avoid_collinearity: bool = False,
        _test_hemisphere_colors: bool = False,
) -> Figure:
    """
    Plots an IPPM graph, always including all available hemispheres.

    Args:
        ippm (IPPM): IPPM object to plot.
        colors (dict[str, str]): Dictionary with keys as node names and values as colors in hexadecimal.
            Contains the color for each transform. The nodes and edges are colored accordingly.
        title (str): Title of the plot.
        scale_nodes (bool, optional): scales the node by the significance. Default is False
        figheight (int, optional): Height of the plot. Defaults to 5.
        figwidth (int, optional): Width of the plot. Defaults to 10.
        show_labels (bool, optional): Show transform names as labels on the graph. Defaults to True.
        relabel (dict[str, str], optional): Dictionary to specify optional labels for each node. Dictionary should map
            original transform labels to desired labels. Missing keys will be ignored. Defaults to None (no change).
        avoid_collinearity (bool, optional): Whether to apply a small offset to avoid collinearity between nodes in the same
            serial step. Defaults to False.
        _test_hemisphere_colors (bool, optional): If True, overrides the `colors` dict to color nodes according to their
            hemisphere, for testing vertical separation. Defaults to False.

    Returns:
        (pyplot.Figure): A figure of the IPPM graph.
    """

    if relabel is None:
        relabel = dict()

    # Always plot the entire graph
    ippm_graph_to_plot = ippm.graph

    if title is None:
        title = "IPPM Graph"

    # Pass the test_hemisphere_colors flag to _PlottableIPPMGraph
    plottable_graph = _PlottableIPPMGraph(ippm_graph_to_plot,
                                          avoid_collinearity=avoid_collinearity,
                                          colors=colors,
                                          y_ordinate_style=y_ordinate_style,
                                          connection_style=IPPMConnectionStyle(connection_style),
                                          scale_nodes=scale_nodes,
                                          test_hemisphere_colors=_test_hemisphere_colors)

    if arrowhead_dims is None:
        if plottable_graph.graph.nodes:
            max_y = max(node.y for node in plottable_graph.graph.nodes) if plottable_graph.graph.nodes else 1
            max_x = max(node.x for node in plottable_graph.graph.nodes) if plottable_graph.graph.nodes else 1
            arrowhead_dims = (
                max_y / 30,
                max_x / 30,
            )
        else:
            arrowhead_dims = (0.1, 0.1)

    # first lets aggregate all the information.
    node_x      = [] # x coordinates for nodes eg. (x, y) = (node_x[i], node_y[i])
    node_y      = [] # y coordinates for nodes
    node_colors = [] # color for nodes
    node_sizes  = []  # size of nodes
    edge_colors = []
    bsplines = []
    edge_labels = []

    node: _PlottableNode
    for i, plottable_node in enumerate(plottable_graph.graph.nodes):
        node_colors.append(plottable_node.color)
        node_sizes.append(plottable_node.size)
        node_x.append(plottable_node.x)
        node_y.append(plottable_node.y)

        incoming_edge_endpoints = []
        pred: _PlottableNode
        for pred in plottable_graph.graph.predecessors(plottable_node):
            incoming_edge_endpoints.append(
                (
                    (pred.x, pred.y),
                    (plottable_node.x, plottable_node.y)
                )
            )
            edge_colors.append(plottable_node.color)
            edge_labels.append(plottable_node.label)

        bsplines += _make_bspline_paths(incoming_edge_endpoints)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    text_offset_x = -0.01
    for path, color, label in zip(bsplines, edge_colors, edge_labels):
        ax.plot(path[0], path[1], color=color, linewidth=linewidth, zorder=-1)
        if show_labels:
            display_label = relabel.get(label, f"{label}()")
            ax.text(
                x=path[0][-1] + text_offset_x,
                y=path[1][-1],
                s=display_label,
                color=color,
                zorder=1,
                horizontalalignment="right", verticalalignment='center',
                path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            )
        ax.arrow(
            x=path[0][-1], dx=0.001,
            y=path[1][-1], dy=0,
            shape="full", width=0, lw=0, head_width=arrowhead_dims[0], head_length=arrowhead_dims[1], color=color,
            length_includes_head=True, head_starts_at_zero=False,
        )

    ax.scatter(x=node_x, y=node_y, c=node_colors, s=node_sizes, marker="H", zorder=2)

    # Show lines trailing off into the future from terminal nodes
    future_width = 0.02
    for node in plottable_graph.graph.nodes:
        if node.is_terminal:
            if node.is_input:
                solid_line_to = 0.02
            else:
                solid_line_to = max(node_x) if node_x else 0.02
            solid_extension = solid_line_to + future_width / 2
            dotted_extension = solid_extension + future_width / 2
            ax.plot([node.x, solid_extension], [node.y, node.y], color=node.color, linewidth=linewidth,
                    linestyle="solid")
            ax.plot([solid_extension, dotted_extension], [node.y, node.y], color=node.color, linewidth=linewidth,
                    linestyle="dotted")

    plt.title(title)

    # Y-axis
    y_padding = 0.5
    if node_y:
        ax.set_ylim(min(node_y) - y_padding, max(node_y) + y_padding)
    ax.set_yticklabels([])
    ax.yaxis.set_visible(False)

    # X-axis
    current_xmin, current_xmax = ax.get_xlim()
    desired_xmin, desired_xmax = xlims_s
    if desired_xmin is None:
        desired_xmin = min(current_xmin, min(node_x)) if node_x else current_xmin

    if desired_xmax is None:
        desired_xmax = max(current_xmax, max(node_x) + future_width) if node_x else current_xmax + future_width

    ax.set_xlim((desired_xmin, desired_xmax))

    xticks = ax.get_xticks()
    plt.xticks(xticks,
               [round(tick * 1000)  # Convert labels to ms, and round to avoid float-math issues
                for tick in xticks])
    ax.set_xlabel("Latency (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.set_figheight(figheight)
    fig.set_figwidth(figwidth)

    return fig


def xlims_from_expressionset(es: ExpressionSet, padding: float = 0.05) -> tuple[float, float]:
    """
    Get an appropriate set of xlims from an ExpressionSet.

    Args:
        es (ExpressionSet):
        padding (float): The amount of padding to add either side of the IPPM plot, in seconds. Default is 0.05 (50ms).

    Returns:
        tuple[float, float]: xmin, xmax
    """
    return (
        es.latencies.min() - padding,
        es.latencies.max() + padding,
    )


def _make_bspline_paths(spike_coordinate_pairs: list[tuple[_XY, _XY]]) -> list[list[NDArray]]:
    """
    Given a list of spike positions pairs, return a list of
    b-splines. First, find the control points, and second
    create the b-splines from these control points.

    Args:
        spike_coordinate_pairs (List[List[Tuple[float, float]]]): Each list contains the x-axis values and y-axis values
            for the start and end of a BSpline, e.g., [(0, 1), (1, 0)].

    Returns:
        List[List[np.array]]: A list of a list of np arrays. Each list contains two np.arrays. The first np.array contains
            the x-axis points and the second one contains the y-axis points. Together, they define a BSpline. Thus, it is
            a list of BSplines.
    """
    bspline_path_array = []
    for pair in spike_coordinate_pairs:
        start_X = pair[0][0]
        start_Y = pair[0][1]
        end_X = pair[1][0]
        end_Y = pair[1][1]

        if start_X + 35 > end_X and start_Y == end_Y:
            # the nodes are too close to use a bspline. Null edge.
            # add 2d np array where the first element is xs and second is ys
            xs = np.linspace(start_X, end_X, 100, endpoint=True)
            ys = np.array([start_Y] * 100)
            bspline_path_array.append([xs, ys])
        else:
            ctr_pts = _make_bspline_ctr_points(pair)
            bspline_path_array.append(_make_bspline_path(ctr_pts))

    return bspline_path_array


def _make_bspline_ctr_points(start_and_end_node_coordinates: tuple[_XY, _XY]) -> NDArray:
    """
    Given the position of a start spike and an end spike, create
    a set of 6 control points needed for a b-spline.

    The first one and last one is the position of a start spike
    and an end spikes themselves, and the intermediate four are
    worked out using some simple rules.

    Args:
        start_and_end_node_coordinates (List[Tuple[float, float]]): List containing the start and end coordinates for one edge.
            First tuple is start, second is end. First element in tuple is x coord, second is y coord.

    Returns:
        np.array: A list of tuples of coordinates. Each coordinate pair represents a control point.
    """

    start_X, start_Y = start_and_end_node_coordinates[0]
    end_X, end_Y = start_and_end_node_coordinates[1]

    if end_X < start_X:
        # reverse BSpline
        start_X, end_X = end_X, start_X
        start_Y, end_Y = end_Y, start_Y

    # Offset points: chosen for aesthetics, but with a squish down to evenly-spaced when nodes are too small
    x_diff = end_X - start_X
    offsets = [
        min(0.005, 1 * x_diff / 5),
        min(0.010, 2 * x_diff / 5),
        min(0.020, 3 * x_diff / 5),
        min(0.030, 4 * x_diff / 5),
    ]

    ctr_points = np.array(
        [
            # start
            (start_X, start_Y),
            # first 2
            (start_X + offsets[0], start_Y),
            (start_X + offsets[1], start_Y),
            # second 2
            (start_X + offsets[2], end_Y),
            (start_X + offsets[3], end_Y),
            # end
            (end_X, end_Y),
        ]
    )

    return ctr_points


def _make_bspline_path(ctr_points: NDArray) -> list[NDArray]:
    """
    With an input of six control points, return an interpolated
    b-spline path which corresponds to a curved edge from one node to another.

    Args:
        ctr_points (NDArray): 2d NDArray containing the coordinates of the center points.

    Returns:
        List[NDArray]: A list of NDArrays that represent one BSpline path. The first list is a list of x-axis coordinates
            the second is a list of y-axis coordinates.
    """
    x = ctr_points[:, 0]
    y = ctr_points[:, 1]

    length = len(x)
    t = np.linspace(0, 1, length - 2, endpoint=True)
    t = np.append([0, 0, 0], t)
    t = np.append(t, [1, 1, 1])

    tck = [t, [x, y], 3]
    u3 = np.linspace(0, 1, (max(length * 2, 70)), endpoint=True)
    # Don't know why this is raising a warning
    # noinspection PyTypeChecker
    bspline_path: list[NDArray] = splev(u3, tck)

    return bspline_path


def _get_y_coordinate_progressive(
        partition_number: int,
        spacing: float = 1) -> float:
    return partition_number * spacing


def _get_y_coordinate_centered(
        function_idx_within_level: int,
        function_total_within_level: int,
        max_function_total_within_level: int,
        positive_nudge_frac: float,
        spacing: float = 1) -> float:
    total_height = (max_function_total_within_level - 1) * spacing
    this_height = (function_total_within_level - 1) * spacing
    baseline = (total_height - this_height) / 2
    y_ord = baseline + function_idx_within_level * spacing
    # / 2 because sometimes there's a 1/2-spacing offset between consecutive steps depending on parity, which can
    # inadvertently cause collinearity again, which we're trying to avoid
    y_ord += positive_nudge_frac * spacing / 2
    return y_ord
