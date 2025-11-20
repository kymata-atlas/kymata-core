from typing import Optional, NamedTuple

import matplotlib.patheffects as pe
from kymata.entities.expression import BLOCK_LEFT, BLOCK_RIGHT
from matplotlib import pyplot as plt
from networkx.classes import DiGraph
from networkx.relabel import relabel_nodes

from kymata.ippm.graph import IPPMGraph, IPPMConnectionStyle, IPPMNode
from kymata.ippm.ippm import IPPM
from kymata.plot.figure import resize_fig_keep_aspect
from kymata.plot.splines import make_bspline_paths


class _YOrdinateStyle:
    """
    Enumeration for Y-ordinate plotting styles.

    Attributes:
        progressive: Points are plotted with increasing y ordinates.
        centered: Points are vertically centered.
    """
    progressive = "progressive"
    centered = "centered"


def plot_ippm(
        ippm: IPPM,
        colors: dict[str, str],
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        xlims_s: tuple[Optional[float], Optional[float]] = (None, None),
        y_ordinate_style: str = _YOrdinateStyle.centered,
        connection_style: str = IPPMConnectionStyle.last_to_first,
        scale_nodes: bool = False,
        vertical_spacing: float = 11,
        figwidth: float = 10,
        arrowhead_dims: tuple[float, float] = None,
        linewidth: float = 3,
        show_labels: bool = True,
        relabel: dict[str, str] | None = None,
        avoid_collinearity: bool = False,
) -> plt.Axes:
    """
    Plots an IPPM graph, always including all available hemispheres.

    Args:
        ippm (IPPM): IPPM object to plot.
        colors (dict[str, str]): Dictionary with keys as node names and values as colors in hexadecimal.
            Contains the color for each transform. The nodes and edges are colored accordingly.
        title (str): Title of the plot.
        scale_nodes (bool, optional): scales the node by the significance. Default is False
        vertical_spacing (float, optional): Vertical space between adjacent nodes.
        figwidth (float, optional): Width of the plot. Height will be scaled to match. Defaults to 10.
        arrowhead_dims (tuple[float, float], optional): Tuple of arrowhead length and thickness.
        linewidth (float, optional): Thickness of the transform lines.
        show_labels (bool, optional): Show transform names as labels on the graph. Defaults to True.
        relabel (dict[str, str], optional): Dictionary to specify optional labels for each node. Dictionary should map
            original transform labels to desired labels. Missing keys will be ignored. Defaults to None (no change).
        avoid_collinearity (bool, optional): Whether to apply a small offset to avoid collinearity between nodes in the same
            serial step. Defaults to False.

    Returns:
        (pyplot.Axes): Axes on which the IPPM was plotted
    """

    if relabel is None:
        relabel = dict()

    # Always plot the entire graph
    ippm_graph_to_plot = ippm.graph

    if title is None:
        title = "IPPM Graph"

    plottable_graph = _PlottableIPPMGraph(
        ippm_graph_to_plot,
        avoid_collinearity=avoid_collinearity,
        colors=colors,
        y_ordinate_style=y_ordinate_style,
        spacing=vertical_spacing,
        connection_style=IPPMConnectionStyle(connection_style),
        scale_nodes=scale_nodes,
    )

    # Iterate over each node in the graph, and store its colour, incoming edge label, and incoming-edge path
    node_x      = []  # x coordinates for nodes eg. (x, y) = (node_x[i], node_y[i])
    node_y      = []  # y coordinates for nodes
    node_colors = []  # color for nodes
    node_sizes  = []  # size of nodes
    edge_colors = []
    bsplines    = []
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

        bsplines += make_bspline_paths(incoming_edge_endpoints)

    node_x_extent_s = (max(node_x) - min(node_x))  # Seconds
    node_y_extent = (max(node_y) - min(node_y))

    fig: plt.Figure
    _ax: plt.Axes
    if ax is None:
        fig, _ax = plt.subplots()
    else:
        _ax = ax
        fig = _ax.figure

    # Plot paths
    text_offset_x = -0.01
    for path, color, label in zip(bsplines, edge_colors, edge_labels):
        _ax.plot(path[0], path[1], color=color, linewidth=linewidth, zorder=-1)
        if show_labels:
            display_label = relabel.get(label, f"{label}()")
            _ax.text(
                x=path[0][-1] + text_offset_x,
                y=path[1][-1],
                s=display_label,
                color=color,
                zorder=1,
                horizontalalignment="right", verticalalignment='center',
                path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            )

    # Plot points as little hexes
    _ax.scatter(x=node_x, y=node_y, c=node_colors, s=node_sizes, marker="H", zorder=2)

    # Plot arrows
    if arrowhead_dims is not None:
        arrowhead_length, arrowhead_thickness = arrowhead_dims
    else:
        # Thickness is fixed because the vertical size of the figure increases with additional nodes
        arrowhead_thickness = 6
        # Length is rescaled as a fixed fraction of the width of the graph
        arrowhead_length = node_x_extent_s / 28  # Adjusted until it looked good
    for path, color in zip(bsplines, edge_colors):
        path_start_x = path[0][0]
        path_start_y = path[1][0]
        path_end_x   = path[0][-1]
        path_end_y   = path[1][-1]
        # If the arrow points backwards, the start and end are swapped
        if path_start_x < path_end_x:
            x = path_end_x
            y = path_end_y
        else:
            x = path_start_x
            y = path_start_y
        _ax.arrow(
            # Apply negative x offset so the arrow points to the edge of the hex
            x=x - 0.003,
            y=y,
            # Give direction to the right
            dx=0.001, dy=0,
            shape="full", width=0, lw=0,
            head_width=arrowhead_thickness, head_length=arrowhead_length,
            color=color,
            length_includes_head=True, head_starts_at_zero=False,
        )

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
            _ax.plot([node.x, solid_extension], [node.y, node.y], color=node.color, linewidth=linewidth,
                     linestyle="solid")
            _ax.plot([solid_extension, dotted_extension], [node.y, node.y], color=node.color, linewidth=linewidth,
                     linestyle="dotted")

    # Add separation line and left/right labels if there are two hemispheres
    if {BLOCK_LEFT, BLOCK_RIGHT} <= set(plottable_graph.hemispheres):
        # Plot midline
        midline = min(node_y) + node_y_extent / 2
        _ax.plot((0, 0.050), (midline, midline), color="black", linewidth=1, linestyle="solid")
        # Text labels
        bottom_label_height = min(node_y) + midline - (0.3 * vertical_spacing)
        top_label_height    = min(node_y) + midline + (0.3 * vertical_spacing)
        _ax.text(s="left", x=0, y=top_label_height,
                 horizontalalignment="left", verticalalignment="center",
                 fontsize="small")
        _ax.text(s="right", x=0, y=bottom_label_height,
                 horizontalalignment="left", verticalalignment="center",
                 fontsize="small")

    _ax.set_title(title)

    # Y-axis
    node_y_extent = min(node_y) - vertical_spacing, max(node_y) + vertical_spacing
    _ax.set_ylim(*node_y_extent)
    _ax.set_yticklabels([])
    _ax.yaxis.set_visible(False)

    # X-axis
    current_xmin, current_xmax = _ax.get_xlim()
    desired_xmin, desired_xmax = xlims_s
    if desired_xmin is None:
        desired_xmin = min(current_xmin, min(node_x)) if node_x else current_xmin

    if desired_xmax is None:
        desired_xmax = max(current_xmax, max(node_x) + future_width) if node_x else current_xmax + future_width

    _ax.set_xlim((desired_xmin, desired_xmax))

    # Set tick labels
    xticks = _ax.get_xticks()
    _ax.set_xticks(xticks,
                   [
                       # Convert labels to ms, and round to avoid float-math issues
                       round(tick * 1000)
                       for tick in xticks
                   ])

    # Set decoration
    _ax.set_xlabel("Latency (ms)")

    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["left"].set_visible(False)

    if ax is None:
        # Figure created locally
        # First set to standard size to make sure vertical spacing is preserved
        fig.set_figwidth(10)
        fig.set_figheight((node_y_extent[1] - node_y_extent[0]) / 25)  # This fraction adjusted until it looks good
        # Then resize to desired size
        resize_fig_keep_aspect(fig, figwidth)

    return _ax


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
                 spacing: float,
                 avoid_collinearity: bool,
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
            spacing (float): Vertical space between adjacent nodes.
            avoid_collinearity (bool): Whether to apply a small offset to avoid collinearity between nodes in the same
                serial step.
            scale_nodes (bool): Whether to scale node sizes based on the significance of the corresponding expression
                points.
        """

        # Vertical spacing between nodes
        self.node_spacing = spacing

        self.hemispheres: list[str] = sorted({
            node.hemisphere
            for node in ippm_graph.graph_full.nodes
        })

        # The hemispheres will be separated, so we'll proceed with each hemisphere in turn
        # (hemi, comp_i) → transform -> y-ordinate
        y_ordinates = dict()
        # hemi → [components]
        components_per_hemi: dict[str, list[frozenset[str]]] = dict()
        for hemisphere in self.hemispheres:
            this_is_second_hemisphere = (len(self.hemispheres) == 2 and hemisphere == self.hemispheres[1])
            if y_ordinate_style == _YOrdinateStyle.progressive:
                # Progressive wants stacked transforms without gaps, so we build set of components from the transforms
                # in the graph

                # { components } where component is { transforms }
                components_this_hemi: list[frozenset[str]] = [
                    frozenset({
                        node.transform
                        for node in ippm_graph.graph_full.nodes
                        if (node.hemisphere == hemisphere) and (node.transform in component)
                    })
                    for component in ippm_graph.candidate_transform_list.connected_components
                ]
                # Drop empty
                components_this_hemi = [c for c in components_this_hemi if len(c) > 0]
                components_per_hemi[hemisphere] = components_this_hemi
                for component_i, component_transforms in enumerate(components_this_hemi):
                    y_ordinates[(hemisphere, component_i)] = _y_ordinates_progressive(ippm_graph, set(component_transforms), self.node_spacing)

            elif y_ordinate_style == _YOrdinateStyle.centered:
                # Centered wants gaps where there are "missing" transforms in the graph, so we build the list of
                # transforms from the ctl
                components_this_hemi = sorted(ippm_graph.candidate_transform_list.connected_components)
                # Want order of components to be mirrored between the hemispheres
                if this_is_second_hemisphere:
                    components_this_hemi.reverse()
                components_per_hemi[hemisphere] = components_this_hemi
                for component_i, component_transforms in enumerate(components_this_hemi):
                    y_ordinates[(hemisphere, component_i)] = _all_y_ordinates_centered(ippm_graph, set(component_transforms), self.node_spacing, avoid_collinearity)
            else:
                raise NotImplementedError()

        # Apply an offset for each component.
        offset = 0
        # (hemi, comp_i) → offset
        component_offsets = dict()
        for hemi_i, hemi in enumerate(reversed(self.hemispheres)):  # reversed so hemispheres go top-to-bottom
            for comp_i, component in enumerate(components_per_hemi[hemi]):
                y_vals = list(y_ordinates[(hemi, comp_i)].values())
                if len(y_vals) > 0:
                    component_height = max(y_vals) - min(y_vals)
                else:
                    component_height = 0
                component_offsets[(hemi, comp_i)] = offset
                offset += component_height + self.node_spacing
            # Gap between hemispheres
            offset += self.node_spacing

        # Actually shift up the transforms for each component
        for (hemi, comp_i), y_vals in y_ordinates.items():
            for trans in y_vals:
                y_vals[trans] += component_offsets[(hemi, comp_i)]

        if connection_style == IPPMConnectionStyle.last_to_first:
            preferred_graph = ippm_graph.graph_last_to_first
        elif connection_style == IPPMConnectionStyle.first_to_first:
            preferred_graph = ippm_graph.graph_first_to_first
        else:
            # IPPMConnectionStyle.full typically has too many nodes and edges to be performant and should be avoided
            raise NotImplementedError()

        # Apply y-ordinates to desired graph

        def lookup_component_index(hemisphere: str, transform: str) -> int:
            for i, comp in enumerate(components_per_hemi[hemisphere]):
                if transform in comp:
                    return i
            raise RuntimeError(f"{transform=} not found in any component for {hemisphere=}")

        node: IPPMNode
        self.graph: DiGraph = relabel_nodes(
            preferred_graph,
            {
                node: _PlottableNode(
                    label=node.transform,
                    x=node.latency,
                    y=y_ordinates[(node.hemisphere, lookup_component_index(node.hemisphere, node.transform))][node.transform],
                    # Conditionally set color based on test_hemisphere_colors flag
                    color=colors[node.transform],
                    is_input=node.is_input,
                    is_terminal=node.transform in ippm_graph.terminals,
                    size=-1 * node.logp_value if scale_nodes else 150,
                )
                for node in preferred_graph.nodes
            }
        )


def _y_ordinates_progressive(ippm_graph: IPPMGraph,
                             transforms_this_component: set[str],
                             node_spacing: float,
                             ) -> dict[str, float]:
    """
    Compute progressive y-ordinates for one component, relative to the bottom of that component in the graph.

    Args:
        ippm_graph: The IPPMGraph.
        transforms_this_component (set[str]): Set of transform names for this component.
        node_spacing (float): Vertical spacing between vertically adjacent transforms.

    Returns:
        dict[str, float]: transform name → y-coordinate.
    """

    # Inputs sorted alphabetically (reversed)
    inputs = sorted({
        node.transform
        for node in ippm_graph.graph_full.nodes
        if node.is_input and node.transform in transforms_this_component
    }, reverse=True)

    serial_sequence = ippm_graph.candidate_transform_list.serial_sequence
    # transform → sequence_step_idx
    step_idxs: dict[str, int] = dict()
    for step_idx, step in enumerate(serial_sequence):
        for transform in step:
            step_idxs[transform] = step_idx
    # Sort by serial sequence, then alphabetically
    # Do both sorts in reverse order
    other_transforms = sorted((t for t in transforms_this_component if t not in inputs), reverse=True)
    other_transforms.sort(key=lambda trans: step_idxs[trans], reverse=True)

    # Stack inputs
    y_ordinates: dict[str, float] = {
        input_transform: i * node_spacing
        for i, input_transform in enumerate(inputs)
    }
    # Then reset position and stack non-input transforms
    for i, trans in enumerate(other_transforms):
        y_ordinates[trans] = i * node_spacing

    return y_ordinates


def _all_y_ordinates_centered(ippm_graph: IPPMGraph,
                              transforms_this_component: set[str],
                              node_spacing: float,
                              avoid_collinearity: bool,
                              ) -> dict[str, float]:
    """
    Compute centred y-ordinates for one component, relative to the bottom of that component in the graph.

    Args:
        ippm_graph: The IPPMGraph.
        transforms_this_component (set[str]): Set of transform names for this component.
        node_spacing (float): Vertical spacing between vertically adjacent transforms.
        avoid_collinearity (bool): Apply nudges to position to avoid collinearity.

    Returns:
        dict[str, float]: transform name → y-coordinate.
    """

    # Not 0.5 because sometimes there's a 1/2-spacing offset between consecutive steps depending on parity,
    # which can inadvertently cause collinearity again, which we're trying to avoid
    positive_nudge_frac = 0.25 if avoid_collinearity else 0

    # Build dictionary mapping function names to sequence steps
    serial_sequence = ippm_graph.candidate_transform_list.serial_sequence

    # Filter the serial sequence to include only transforms present in this component
    serial_seq_filtered = [
        [t for t in step if t in transforms_this_component]
        for step in serial_sequence
    ]
    serial_seq_filtered = [s for s in serial_seq_filtered if len(s) > 0]
    max_step_width = max(len(step) for step in serial_seq_filtered)
    total_height = (max_step_width - 1) * node_spacing

    y_ordinates: dict[str, float] = dict()

    # Go through each step and centre
    for step_idx, step in enumerate(serial_seq_filtered):
        width_this_step = len(step)
        this_step_height = (width_this_step - 1) * node_spacing
        this_step_baseline = (total_height - this_step_height) / 2
        for i, trans in enumerate(sorted(step)):
            y_ordinates[trans] = this_step_baseline + (i + positive_nudge_frac) * node_spacing

    return y_ordinates
