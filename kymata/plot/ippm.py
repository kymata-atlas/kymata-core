from collections import defaultdict, Counter
from copy import copy
from enum import StrEnum
from typing import Optional, NamedTuple
from warnings import warn

import matplotlib.patheffects as pe
import numpy as np
from matplotlib import pyplot as plt
from networkx.classes import DiGraph
from networkx.relabel import relabel_nodes
from numpy.typing import NDArray
from scipy.interpolate import splev

from kymata.entities.expression import BLOCK_LEFT, BLOCK_SCALP, ExpressionPoint, ExpressionSet
from kymata.ippm.graph import IPPMGraph, IPPMConnectionStyle
from kymata.ippm.ippm import IPPM


class _YOrdinateStyle(StrEnum):
    """
    Enumeration for Y-ordinate plotting styles.

    Attributes:
        progressive: Points are plotted with increasing y ordinates.
        centered: Points are vertically centered.
    """
    progressive = "progressive"
    centered    = "centered"


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
        """

        ippm_graph = copy(ippm_graph)
        if serial_sequence is None:
            serial_sequence = ippm_graph.candidate_transform_list.serial_sequence

        y_ordinates: dict[str, float] = dict()

        if y_ordinate_style == _YOrdinateStyle.progressive:

            # Stack inputs
            n_transforms = len(ippm_graph.candidate_transform_list.inputs)
            y_axis_partition_size = 1 / n_transforms if n_transforms > 0 else 1
            partition_ptr = 0
            for input_transform in ippm_graph.inputs:
                y_ordinates[input_transform] = _get_y_coordinate_progressive(
                    partition_number=partition_ptr, partition_size=y_axis_partition_size)
                partition_ptr += 1

            # Stack others
            n_transforms = len(ippm_graph.transforms - ippm_graph.candidate_transform_list.inputs)
            y_axis_partition_size = 1 / n_transforms if n_transforms > 0 else 1
            partition_ptr = 0
            # Non-input transforms step upwards
            for transform in ippm_graph.transforms - ippm_graph.inputs:
                y_ordinates[transform] = _get_y_coordinate_progressive(
                    partition_number=partition_ptr, partition_size=y_axis_partition_size)
                partition_ptr += 1

        elif y_ordinate_style == _YOrdinateStyle.centered:

            # Build dictionary mapping function names to sequence steps
            step_idxs = dict()
            for step_i, step in enumerate(serial_sequence):
                for function in step:
                    step_idxs[function] = step_i
            totals_within_serial_step = Counter(step_idxs.values())
            # level -> latest idx
            idxs_within_level = defaultdict(int)
            for transform in sorted(ippm_graph.candidate_transform_list.transforms):
                step_idx = step_idxs[transform]
                y_ordinates[transform] = _get_y_coordinate_centered(
                    function_idx_within_level=idxs_within_level[step_idx],
                    function_total_within_level=totals_within_serial_step[step_idx],
                    max_function_total_within_level=max(totals_within_serial_step.values()),
                    # Nudge each step up progressively more to avoid collinearity
                    positive_nudge_frac=(step_idxs[transform] / len(serial_sequence)
                                         if avoid_collinearity
                                         else 0))
                idxs_within_level[step_idx] += 1

        else:
            raise NotImplementedError()

        if connection_style == IPPMConnectionStyle.last_to_first:
            preferred_graph = ippm_graph.graph_last_to_first
        elif connection_style == IPPMConnectionStyle.first_to_first:
            preferred_graph = ippm_graph.graph_first_to_first
        else:
            # IPPMConnectionStyle.full typically has too many nodes and edges to be performant and should be avoided
            raise NotImplementedError()

        node: ExpressionPoint
        self.graph: DiGraph = relabel_nodes(
            preferred_graph,
            {
                node: _PlottableNode(
                    label=node.transform,
                    x=node.latency,
                    y=y_ordinates[node.transform],
                    color=colors[node.transform],
                    is_input=node.transform in ippm_graph.inputs,
                    is_terminal=node.transform in ippm_graph.terminals,
                    size=-1*node.logp_value if scale_nodes else 150,
                )
                for node in preferred_graph.nodes
            }
        )

        pass


def plot_ippm(
    ippm: IPPM,
    colors: dict[str, str],
    hemisphere: Optional[str] = None,
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
    avoid_collinearity: bool = False,
    serial_sequence: Optional[list[list[str]]] = None,
) -> plt.Figure:
    """
    Plots an IPPM graph.

    Args:
        ippm (IPPM): IPPM object to plot.
        colors (dict[str, str]): Dictionary with keys as node names and values as colors in hexadecimal.
            Contains the color for each transform. The nodes and edges are colored accordingly.
        hemisphere (str): When generating from a HexelExpressionSet, specify whether the left or the right hemisphere
            should be used.
        title (str): Title of the plot.
        scale_nodes (bool, optional): scales the node by the significance. Default is False
        figheight (int, optional): Height of the plot. Defaults to 5.
        figwidth (int, optional): Width of the plot. Defaults to 10.
        show_labels (bool, optional): Show transform names as labels on the graph. Defaults to True.

    Returns:
        (pyplot.Figure): A figure of the IPPM graph.
    """

    if hemisphere is None:
        if BLOCK_LEFT in ippm:
            hemisphere = BLOCK_LEFT
            warn(f"No hemisphere specified, using {hemisphere}")
        elif BLOCK_SCALP in ippm:
            hemisphere = BLOCK_SCALP

    plottable_graph = _PlottableIPPMGraph(ippm[hemisphere],
                                          avoid_collinearity=avoid_collinearity,
                                          serial_sequence=serial_sequence,
                                          colors=colors,
                                          y_ordinate_style=y_ordinate_style,
                                          scale_nodes=scale_nodes)

    if arrowhead_dims is None:
        # Scale arrowheads by size of graph
        arrowhead_dims = (
            max(node.y for node in plottable_graph.graph.nodes) / 30,  # width
            max(node.x for node in plottable_graph.graph.nodes) / 30,  # length
        )
        
    # first lets aggregate all the information.
    node_x      = []  # x coordinates for nodes eg. (x, y) = (node_x[i], node_y[i])
    node_y      = []  # y coordinates for nodes
    node_colors = []  # color for nodes
    node_sizes  = []  # size of nodes
    edge_colors = []
    bsplines = []
    edge_labels = []
    node: _PlottableNode
    for i, node in enumerate(plottable_graph.graph.nodes):
        node_colors.append(node.color)
        node_sizes.append(node.size)
        node_x.append(node.x)
        node_y.append(node.y)

        incoming_edge_endpoints = []
        pred: _PlottableNode
        for pred in plottable_graph.graph.predecessors(node):
            # save edge coordinates and color the edge the same color as the finishing node.
            incoming_edge_endpoints.append(
                (
                    # Start
                    (pred.x, pred.y),
                    # End
                    (node.x, node.y)
                )
            )
            edge_colors.append(node_colors[i])
            edge_labels.append(node.label)
            
        bsplines += _make_bspline_paths(incoming_edge_endpoints)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    text_offset_x = -0.01  # s
    for path, color, label in zip(bsplines, edge_colors, edge_labels):
        ax.plot(path[0], path[1], color=color, linewidth=linewidth, zorder=-1)
        if show_labels:
            ax.text(
                x=path[0][-1] + text_offset_x,
                y=path[1][-1],
                s=f"{label}()",
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
    future_width = 0.02  # s
    for node in plottable_graph.graph.nodes:
        if node.is_terminal:
            # Plot a solid line, then a dotted section for "trailing off"
            if node.is_input:
                # Input nodes with nothing dowstream trail off quickly
                solid_line_to = 0.02  # s
            else:
                # Non-input nodes with nothing downstream trail off at the end of the graph
                solid_line_to = max(node_x)
            solid_extension = solid_line_to + future_width / 2
            dotted_extension = solid_extension + future_width / 2
            ax.plot([node.x, solid_extension], [node.y, node.y], color=node.color, linewidth=linewidth, linestyle="solid")
            ax.plot([solid_extension, dotted_extension], [node.y, node.y], color=node.color, linewidth=linewidth, linestyle="dotted")

    if title is not None:
        plt.title(title)

    # Y-axis
    y_padding = 0.5
    ax.set_ylim(min(node_y) - y_padding, max(node_y) + y_padding)
    ax.set_yticklabels([])
    ax.yaxis.set_visible(False)

    # X-axis
    # Set axis limits
    current_xmin, current_xmax = ax.get_xlim()
    desired_xmin, desired_xmax = xlims_s
    if desired_xmin is None:
        desired_xmin = min(current_xmin, min(node_x))
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
        spacing: float = 1) -> float:
    total_height = (max_function_total_within_level - 1) * spacing
    this_height = (function_total_within_level - 1) * spacing
    baseline = (total_height - this_height) / 2
    y_ord = baseline + function_idx_within_level * spacing
    # / 2 because sometimes there's a 1/2-spacing offset between consecutive steps depending on parity, which can
    # inadvertently cause collinearity again, which we're trying to avoid
    y_ord += positive_nudge_frac * spacing / 2
    return y_ord
