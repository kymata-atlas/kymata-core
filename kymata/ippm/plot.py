from itertools import cycle
from statistics import NormalDist
from typing import Optional

import matplotlib.colors
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from scipy.interpolate import splev
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import normalize

from kymata.entities.constants import HEMI_RIGHT, HEMI_LEFT
from kymata.ippm.build import IPPMGraph
from kymata.ippm.data_tools import SpikeDict, copy_hemisphere, ExpressionPairing


def plot_ippm(
    graph: IPPMGraph,
    colors: dict[str, str],
    title: Optional[str] = None,
    scale_spikes: bool = False,
    figheight: int = 5,
    figwidth: int = 10,
    arrowhead_dims: tuple[float, float] = (.02, 8),
    linewidth: float = 3,
):
    """
    Plots an acyclic, directed graph using the graph held in graph. Edges are generated using BSplines.

    Args:
        graph (NodeDict): Dictionary with keys as node names and values as IPPMNode objects.
            Contains nodes as keys and magnitude, position, and incoming edges in the IPPMNode object.
        colors (dict[str, str]): Dictionary with keys as node names and values as colors in hexadecimal.
            Contains the color for each function. The nodes and edges are colored accordingly.
        title (str): Title of the plot.
        scale_spikes (bool, optional): scales the node by the significance. Default is False
        figheight (int, optional): Height of the plot. Defaults to 5.
        figwidth (int, optional): Width of the plot. Defaults to 10.
    """
    def __get_label(inc_edge: str) -> str:
        try:
            # assumes inc edge is named as transform-x.
            label = inc_edge[:inc_edge.index("-")]
            return label
        except ValueError:
            # "-" not found in inc_edge. Therefore, it must be input transform.
            return inc_edge
        
    # first lets aggregate all the information.
    node_x      = list(range(len(graph.keys())))  # x coordinates for nodes eg. (x, y) = (node_x[i], node_y[i])
    node_y      = list(range(len(graph.keys())))  # y coordinates for nodes
    node_colors = list(range(len(graph.keys())))  # color for nodes
    node_sizes  = list(range(len(graph.keys())))  # size of nodes
    edge_colors = []
    bsplines = []
    edge_labels = []
    for i, node in enumerate(graph.keys()):
        for function, color in colors.items():
            # search for function color.
            if function in node:
                node_colors[i] = color
                break

        node_sizes[i] = graph[node].magnitude
        node_x[i], node_y[i] = graph[node].position

        pairs = []
        for inc_edge in graph[node].inc_edges:
            # save edge coordinates and color the edge the same color as the finishing node.
            start = graph[inc_edge].position
            end = graph[node].position
            pairs.append([(start[0], start[1]), (end[0], end[1])])
            edge_colors.append(node_colors[i])
            label = __get_label(inc_edge)
            edge_labels.append(label)
            
        bsplines += _make_bspline_paths(pairs)

    # override node size
    if not scale_spikes:
        node_sizes = [150] * len(graph.keys())

    fig, ax = plt.subplots()

    text_offset_x = -10
    for path, color, label in zip(bsplines, edge_colors, edge_labels):
        ax.plot(path[0], path[1], color=color, linewidth=linewidth, zorder=-1)
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
            x=path[0][-1], dx=1,
            y=path[1][-1], dy=0,
            shape="full", width=0, lw=0, head_width=arrowhead_dims[0], head_length=arrowhead_dims[1], color=color,
            length_includes_head=True, head_starts_at_zero=False,
        )

    ax.scatter(x=node_x, y=node_y, c=node_colors, s=node_sizes, marker="H", zorder=2)

    if title is not None:
        plt.title(title)

    ax.set_ylim(min(node_y) - 0.1, max(node_y) + 0.1)
    ax.set_yticklabels([])
    ax.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Latency (ms)")

    fig.set_figheight(figheight)
    fig.set_figwidth(figwidth)


def _make_bspline_paths(
    spike_coordinate_pairs: list[list[tuple[float, float]]],
) -> list[list[np.array]]:
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


def _make_bspline_ctr_points(
    start_and_end_node_coordinates: list[tuple[float, float]],
) -> np.array:
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
        min(5,  1 * x_diff / 5),
        min(10, 2 * x_diff / 5),
        min(20, 3 * x_diff / 5),
        min(30, 4 * x_diff / 5),
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
    bspline_path: list[NDArray] = splev(u3, tck)

    return bspline_path


def stem_plot(
    spikes: SpikeDict,
    title: Optional[str] = None,
    timepoints: int = 201,
    y_limit: float = pow(10, -100),
    number_of_spikes: int = 200000,
    figheight: int = 7,
    figwidth: int = 12,
):
    """
    Plots a stem plot using spikes.

    Params
    ------
        spikes : Contains function spikes in the form of a spike object. All timings are found there.
        title : Title of plot.
    """
    # estimate significance parameter
    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)  # 5-sigma
    bonferroni_corrected_alpha = 1 - (
        pow((1 - alpha), (1 / (2 * timepoints * number_of_spikes)))
    )

    # assign unique color to each function
    cycol = cycle(sns.color_palette("hls", len(spikes.keys())))
    for _, spike in spikes.items():
        spike.color = matplotlib.colors.to_hex(next(cycol))

    fig, (left_hem_expression_plot, right_hem_expression_plot) = plt.subplots(
        nrows=2, ncols=1, figsize=(figwidth, figheight)
    )
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(right=0.84, left=0.08)

    custom_handles = []
    custom_labels = []
    for key, my_function in spikes.items():
        color = my_function.color
        label = my_function.function

        custom_handles.extend(
            [Line2D([], [], marker=".", color=color, linestyle="None")]
        )
        custom_labels.append(label)

        # left
        left = list(zip(*(my_function.left_best_pairings)))
        if len(left) != 0:
            x_left, y_left = left[0], left[1]
            left_color = np.where(
                np.array(y_left) <= bonferroni_corrected_alpha, color, "black"
            )  # set all insignificant spikes to black
            left_hem_expression_plot.vlines(
                x=x_left, ymin=1, ymax=y_left, color=left_color
            )
            left_hem_expression_plot.scatter(x_left, y_left, color=left_color, s=20)

        # right
        right = list(zip(*(my_function.right_best_pairings)))
        if len(right) != 0:
            x_right, y_right = right[0], right[1]
            right_color = np.where(
                np.array(y_right) <= bonferroni_corrected_alpha, color, "black"
            )  # set all insignificant spikes to black
            right_hem_expression_plot.vlines(
                x=x_right, ymin=1, ymax=y_right, color=right_color
            )
            right_hem_expression_plot.scatter(x_right, y_right, color=right_color, s=20)

    for plot in [right_hem_expression_plot, left_hem_expression_plot]:
        plot.set_yscale("log")
        plot.set_xlim(-200, 800)
        plot.set_ylim(1, y_limit)
        plot.axvline(x=0, color="k", linestyle="dotted")
        plot.axhline(y=bonferroni_corrected_alpha, color="k", linestyle="dotted")
        plot.text(
            -100,
            bonferroni_corrected_alpha,
            "α*",
            bbox={"facecolor": "white", "edgecolor": "none"},
            verticalalignment="center",
        )
        plot.text(
            600,
            bonferroni_corrected_alpha,
            "α*",
            bbox={"facecolor": "white", "edgecolor": "none"},
            verticalalignment="center",
        )
        plot.set_yticks([1, pow(10, -50), pow(10, -100)])

    if title is not None:
        left_hem_expression_plot.set_title(title)
    left_hem_expression_plot.set_xticklabels([])
    right_hem_expression_plot.set_xlabel(
        "Latency (ms) relative to onset of the environment"
    )
    right_hem_expression_plot.xaxis.set_ticks(np.arange(-200, 800 + 1, 100))
    right_hem_expression_plot.invert_yaxis()
    left_hem_expression_plot.text(
        -180,
        y_limit * 10000000,
        "left hemisphere",
        style="italic",
        verticalalignment="center",
    )
    right_hem_expression_plot.text(
        -180,
        y_limit * 10000000,
        "right hemisphere",
        style="italic",
        verticalalignment="center",
    )
    y_axis_label = "p-value (with α at 5-sigma, Bonferroni corrected)"
    left_hem_expression_plot.text(
        -275, 1, y_axis_label, verticalalignment="center", rotation="vertical"
    )
    right_hem_expression_plot.text(
        0,
        1,
        "   onset of environment   ",
        color="white",
        fontsize="x-small",
        bbox={"facecolor": "grey", "edgecolor": "none"},
        verticalalignment="center",
        horizontalalignment="center",
        rotation="vertical",
    )
    left_hem_expression_plot.legend(
        handles=custom_handles,
        labels=custom_labels,
        fontsize="x-small",
        bbox_to_anchor=(1.2, 1),
    )

    plt.show()


def plot_k_dist_1D(
    timings: list[ExpressionPairing], k: int = 4, normalise: bool = False
):
    """
    This could be optimised further but since we aren't using it, we can leave it as it is.

    A utility function to plot the k-dist graph for a set of timings. Essentially, the k dist graph plots the distance
    to the kth neighbour for each point. By inspecting the gradient of the graph, we can gain some intuition behind the density of
    points within the dataset, which can feed into selecting the optimal DBSCAN hyperparameters.

    For more details refer to section 4.2 in https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf

    Parameters
    ----------
    timings: list of timings extracted from a spikes. It contains the timings for one function and one hemisphere
    k: the k we use to find the kth neighbour. Paper above advises to use k=4.
    normalise: whether to normalise before plotting the k-dist. It is important because the k-dist then equally weights both dimensions.

    Returns
    -------
    Nothing but plots a graph.
    """

    alpha = 3.55e-15
    X = pd.DataFrame(columns=["Latency"])
    for latency, spike in timings:
        if spike <= alpha:
            X.loc[len(X)] = [latency]

    if normalise:
        X = normalize(X)

    distance_M = euclidean_distances(
        X
    )  # rows are points, columns are other points same order with values as distances
    k_dists = []
    for r in range(len(distance_M)):
        sorted_dists = sorted(distance_M[r], reverse=True)  # descending order
        k_dists.append(sorted_dists[k])  # store k-dist
    sorted_k_dists = sorted(k_dists, reverse=True)
    plt.plot(list(range(0, len(sorted_k_dists))), sorted_k_dists)
    plt.show()


def plot_denoised_vs_noisy(spikes: SpikeDict, clusterer, title: str):
    """
    Utility function to plot the noisy and denoised versions. It runs the supplied clusterer and then copies the denoised spikes, which
    are fed into a stem plot.

    Parameters
    ----------
    spikes: spikes we want to denoise then plot
    clusterer: A child class of DenoisingStrategy that implements .cluster
    title: title of plot

    Returns
    -------
    Nothing but plots a graph.
    """
    denoised_spikes = clusterer.cluster(spikes, HEMI_RIGHT)
    copy_hemisphere(denoised_spikes, spikes, HEMI_LEFT, HEMI_RIGHT)
    stem_plot(denoised_spikes, title)
