from collections import defaultdict, Counter
from copy import copy
from enum import StrEnum
from typing import Optional, NamedTuple
from warnings import warn

import matplotlib.patheffects as pe
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from networkx.classes import DiGraph
from networkx.relabel import relabel_nodes
from numpy.typing import NDArray
from scipy.interpolate import splev

from kymata.entities.expression import ExpressionSet
from kymata.ippm.graph import IPPMGraph, IPPMConnectionStyle, IPPMNode  # Import IPPMNode
from kymata.ippm.ippm import IPPM


class _YOrdinateStyle(StrEnum):
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
                 serial_sequence: Optional[list[list[str]]],
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
            serial_sequence (Optional[list[list[str]]]): A list representing the serial execution order of transforms.
                If None, the default serial sequence from ippm_graph is used.
            scale_nodes (bool): Whether to scale node sizes based on the significance of the corresponding expression
                points.
            test_hemisphere_colors (bool): If True, overrides the `colors` to use predefined hemisphere colors.
        """

        if serial_sequence is None:
            serial_sequence = ippm_graph.candidate_transform_list.serial_sequence

        # y_ordinates now maps (transform_name, hemisphere_name) to y_coordinate
        y_ordinates: dict[tuple[str, str], float] = dict()

        # Gather all unique (transform, hemisphere) pairs present in the graph
        unique_transform_hemispheres = set()
        for node in ippm_graph.graph_full.nodes:
            unique_transform_hemispheres.add((node.transform, node.hemisphere))

        if y_ordinate_style == _YOrdinateStyle.progressive:
            warn("Progressive Y-ordinate style might not separate hemispheres well without further modification.") #

            # Stack inputs
            # To separate hemispheres in progressive style, you would need to define separate y-ranges
            # or a larger offset for each hemisphere. This part is left as is for now, as the focus
            # is on 'centered' style.
            n_transforms = len(ippm_graph.candidate_transform_list.inputs)
            y_axis_partition_size = 1 / n_transforms if n_transforms > 0 else 1
            partition_ptr = 0
            for input_transform in ippm_graph.inputs:
                for hemisphere_val in sorted(list(set(h for t, h in unique_transform_hemispheres if t == input_transform))):
                    y_ordinates[(input_transform, hemisphere_val)] = _get_y_coordinate_progressive(
                        partition_number=partition_ptr, partition_size=y_axis_partition_size)
                partition_ptr += 1

            # Stack others
            n_transforms = len(ippm_graph.transforms - ippm_graph.inputs)
            y_axis_partition_size = 1 / n_transforms if n_transforms > 0 else 1
            partition_ptr = 0
            for transform in ippm_graph.transforms - ippm_graph.inputs:
                for hemisphere_val in sorted(list(set(h for t, h in unique_transform_hemispheres if t == transform))):
                    y_ordinates[(transform, hemisphere_val)] = _get_y_coordinate_progressive(
                        partition_number=partition_ptr, partition_size=y_axis_partition_size)
                partition_ptr += 1


        elif y_ordinate_style == _YOrdinateStyle.centered:

            # Build dictionary mapping function names to sequence steps
            step_idxs = dict()
            for step_i, step in enumerate(serial_sequence):
                for function in step:
                    step_idxs[function] = step_i

            # Separate unique (transform, hemisphere) pairs by hemisphere
            unique_transform_hemispheres_lh = sorted([(t, h) for t, h in unique_transform_hemispheres if h == "LH"], key=lambda x: x[0]) #
            unique_transform_hemispheres_rh = sorted([(t, h) for t, h in unique_transform_hemispheres if h == "RH"], key=lambda x: x[0]) #
            unique_transform_hemispheres_other = sorted([(t, h) for t, h in unique_transform_hemispheres if h not in ["LH", "RH"]], key=lambda x: x[0]) #


            # Calculate totals and max totals for each hemisphere group independently
            # This ensures y-coordinates are calculated relative to within their hemisphere
            totals_within_serial_step_lh = Counter()
            idxs_within_level_lh = defaultdict(int)
            max_function_total_within_level_lh = 0
            temp_step_counts_lh = defaultdict(set)
            for transform, hemisphere in unique_transform_hemispheres_lh:
                if transform in step_idxs:
                    step_idx = step_idxs[transform]
                    temp_step_counts_lh[step_idx].add((transform, hemisphere))
            for step_idx, items_in_step in temp_step_counts_lh.items():
                totals_within_serial_step_lh[step_idx] = len(items_in_step)
                if len(items_in_step) > max_function_total_within_level_lh:
                    max_function_total_within_level_lh = len(items_in_step)

            totals_within_serial_step_rh = Counter()
            idxs_within_level_rh = defaultdict(int)
            max_function_total_within_level_rh = 0
            temp_step_counts_rh = defaultdict(set)
            for transform, hemisphere in unique_transform_hemispheres_rh:
                if transform in step_idxs:
                    step_idx = step_idxs[transform]
                    temp_step_counts_rh[step_idx].add((transform, hemisphere))
            for step_idx, items_in_step in temp_step_counts_rh.items():
                totals_within_serial_step_rh[step_idx] = len(items_in_step)
                if len(items_in_step) > max_function_total_within_level_rh:
                    max_function_total_within_level_rh = len(items_in_step)

            totals_within_serial_step_other = Counter()
            idxs_within_level_other = defaultdict(int)
            max_function_total_within_level_other = 0
            temp_step_counts_other = defaultdict(set)
            for transform, hemisphere in unique_transform_hemispheres_other:
                if transform in step_idxs:
                    step_idx = step_idxs[transform]
                    temp_step_counts_other[step_idx].add((transform, hemisphere))
            for step_idx, items_in_step in temp_step_counts_other.items():
                totals_within_serial_step_other[step_idx] = len(items_in_step)
                if len(items_in_step) > max_function_total_within_level_other:
                    max_function_total_within_level_other = len(items_in_step)

            # Determine the overall maximum "density" level across all hemispheres for scaling separation.
            overall_max_nodes_per_level = max(max_function_total_within_level_lh, max_function_total_within_level_rh, max_function_total_within_level_other) #
            if overall_max_nodes_per_level == 0:
                overall_max_nodes_per_level = 1

            # Define a base factor for vertical separation. Adjust this to control the overall gap.
            base_y_separation_factor = 1 # This factor multiplies the 'height' of the densest level
            # 'spacing' is 1 by default in _get_y_coordinate_centered.
            hemisphere_separation_offset = overall_max_nodes_per_level * base_y_separation_factor #

            # Assign y-coordinates for Left Hemisphere (LH) - Top
            for transform, hemisphere in unique_transform_hemispheres_lh:
                if transform not in step_idxs:
                    continue
                step_idx = step_idxs[transform]
                y_ordinates[(transform, hemisphere)] = _get_y_coordinate_centered(
                    function_idx_within_level=idxs_within_level_lh[step_idx],
                    function_total_within_level=totals_within_serial_step_lh[step_idx],
                    max_function_total_within_level=max_function_total_within_level_lh,
                    positive_nudge_frac=(step_idx / len(serial_sequence) if avoid_collinearity else 0),
                    hemisphere_offset=hemisphere_separation_offset / 2 # Offset upwards
                )
                idxs_within_level_lh[step_idx] += 1

            # Assign y-coordinates for Right Hemisphere (RH) - Bottom
            for transform, hemisphere in unique_transform_hemispheres_rh:
                if transform not in step_idxs:
                    continue
                step_idx = step_idxs[transform]
                y_ordinates[(transform, hemisphere)] = _get_y_coordinate_centered(
                    function_idx_within_level=idxs_within_level_rh[step_idx],
                    function_total_within_level=totals_within_serial_step_rh[step_idx],
                    max_function_total_within_level=max_function_total_within_level_rh,
                    positive_nudge_frac=(step_idx / len(serial_sequence) if avoid_collinearity else 0),
                    hemisphere_offset=-hemisphere_separation_offset / 2 # Offset downwards
                )
                idxs_within_level_rh[step_idx] += 1

            # Assign y-coordinates for Other Hemispheres (e.g., SCALP) - In the middle, or offset as desired
            for transform, hemisphere in unique_transform_hemispheres_other:
                if transform not in step_idxs:
                    continue
                step_idx = step_idxs[transform]
                y_ordinates[(transform, hemisphere)] = _get_y_coordinate_centered(
                    function_idx_within_level=idxs_within_level_other[step_idx],
                    function_total_within_level=totals_within_serial_step_other[step_idx],
                    max_function_total_within_level=max_function_total_within_level_other,
                    positive_nudge_frac=(step_idx / len(serial_sequence) if avoid_collinearity else 0),
                    hemisphere_offset=0 # No specific hemisphere offset for others, or define one
                )
                idxs_within_level_other[step_idx] += 1

        else:
            raise NotImplementedError()

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
                    # Access y_ordinate using the (transform, hemisphere) tuple
                    y=y_ordinates[(node.transform, node.hemisphere)], #
                    # Conditionally set color based on test_hemisphere_colors flag
                    color=_get_test_hemisphere_color(node.hemisphere) if test_hemisphere_colors else colors[node.transform], #
                    is_input=node.is_input_node,
                    is_terminal=node.transform in ippm_graph.terminals,
                    size=-1 * node.logp_value if scale_nodes else 150,
                )
                for node in preferred_graph.nodes
            }
        )

        pass


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
        test_hemisphere_colors: bool = False,
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
        test_hemisphere_colors (bool, optional): If True, overrides the `colors` dict to color 'LH' nodes red,
            'RH' nodes blue, and 'SCALP' nodes green for testing vertical separation. Defaults to False.

    Returns:
        (pyplot.Figure): A figure of the IPPM graph.
    """

    if relabel is None:
        relabel = dict()

    # Always plot the entire graph
    ippm_graph_to_plot = ippm._graph

    if title is None:
        title = "IPPM Graph"

    # Retrieve serial_sequence from the IPPMGraph. This refers to the theoretical hierarchy.
    serial_sequence = ippm_graph_to_plot.candidate_transform_list.serial_sequence

    # Pass the test_hemisphere_colors flag to _PlottableIPPMGraph
    plottable_graph = _PlottableIPPMGraph(ippm_graph_to_plot,
                                          avoid_collinearity=avoid_collinearity,
                                          serial_sequence=serial_sequence,
                                          colors=colors,
                                          y_ordinate_style=y_ordinate_style,
                                          connection_style=IPPMConnectionStyle(connection_style),
                                          scale_nodes=scale_nodes,
                                          test_hemisphere_colors=test_hemisphere_colors)

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


def _make_bspline_paths(spike_coordinate_pairs: list[tuple[_XY, _XY]]) -> list[list[np.array]]:
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


def _make_bspline_ctr_points(start_and_end_node_coordinates: tuple[_XY, _XY]) -> np.array:
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
        partition_size: float) -> float:
    return 1 - partition_size * partition_number


def _get_y_coordinate_centered(
        function_idx_within_level: int,
        function_total_within_level: int,
        max_function_total_within_level: int,
        positive_nudge_frac: float,
        spacing: float = 1,
        hemisphere_offset: float = 0) -> float: #
    total_height = (max_function_total_within_level - 1) * spacing
    this_height = (function_total_within_level - 1) * spacing
    baseline = (total_height - this_height) / 2
    y_ord = baseline + function_idx_within_level * spacing
    # / 2 because sometimes there's a 1/2-spacing offset between consecutive steps depending on parity, which can
    # inadvertently cause collinearity again, which we're trying to avoid
    y_ord += positive_nudge_frac * spacing / 2
    y_ord += hemisphere_offset # Apply the hemisphere-specific offset
    return y_ord