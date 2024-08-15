from typing import Dict, List, Tuple, Any

from scipy.interpolate import splev
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from kymata.ippm.data_tools import Node


class IPPMPlotter:
    def plot(self, graph: Dict[str, Node], func_colors: Dict[str, str]):
        x_coords, y_coords, colors, sizes, edge_colors, b_splines = (
            self._extract_data_info_from_dict(graph, func_colors)
        )
        self._draw(
            x_coords, y_coords, colors, sizes, edge_colors, b_splines, func_colors
        )

    def _extract_data_info_from_dict(
        self, graph: Dict[str, Node], func_colors: Dict[str, str]
    ):
        number_of_nodes = len(graph.keys())
        x_coords = [_ for _ in range(number_of_nodes)]
        y_coords = [_ for _ in range(number_of_nodes)]
        colors = [_ for _ in range(number_of_nodes)]
        sizes = [_ for _ in range(number_of_nodes)]
        b_splines = [_ for _ in range(number_of_nodes)]
        edge_colors = []

        for idx, node_infos in enumerate(graph.items()):
            node_name, node_info = node_infos

            function_name = (
                node_name[: node_name.index("-")] if "-" in node_name else node_name
            )
            colors[idx] = func_colors[function_name]

            sizes[idx] = node_info.magnitude
            x_coord, y_coord = node_info.position
            x_coords[idx], y_coords[idx] = x_coord, y_coord

            pairs = []
            for inc_edge in node_info.inc_edges:
                start_x, start_y = graph[inc_edge].position
                pairs.append([(start_x, start_y), (x_coord, y_coord)])
                edge_colors.append(colors[idx])

            b_splines.append(self._make_b_spline_paths(pairs))

        return x_coords, y_coords, colors, sizes, edge_colors, b_splines

    def _make_b_spline_paths(self, pairs: List[Tuple[Any, Any]]) -> np.array:
        """
        Given a list of hexel positions pairs, return a list of
        b-splines. First, find the control points, and second
        create the b-splines from these control points.

        Args:
            hexel_coordinate_pairs (List[List[Tuple[float, float]]]): Each list contains the x-axis values and y-axis values
                for the start and end of a BSpline, e.g., [(0, 1), (1, 0)].

        Returns:
            List[List[np.array]]: A list of a list of np arrays. Each list contains two np.arrays. The first np.array contains
                the x-axis points and the second one contains the y-axis points. Together, they define a BSpline. Thus, it is
                a list of BSplines.
        """
        bspline_path_array = []
        for pair in pairs:
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
                ctr_pts = self._make_b_spline_ctr_points(pair)
                bspline_path_array.append(self._make_b_spline_path(ctr_pts))

        return bspline_path_array

    def _make_b_spline_ctr_points(
        self, start_and_end_node_coordinates: List[Tuple[float, float]]
    ) -> np.array:
        """
        Given the position of a start hexel and an end hexel, create
        a set of 6 control points needed for a b-spline.

        The first one and last one is the position of a start hexel
        and an end hexel themselves, and the intermediate four are
        worked out using some simple rules.

        Args:
            start_and_end_node_coordinates (List[Tuple[float, float]]): List containing the start and end coordinates for one edge.
                First tuple is start, second is end. First element in tuple is x coord, second is y coord.

        Returns:
            np.array: A list of tuples of coordinates. Each coordinate pair represents a control point.
        """

        start_X = start_and_end_node_coordinates[0][0]
        start_Y = start_and_end_node_coordinates[0][1]
        end_X = start_and_end_node_coordinates[1][0]
        end_Y = start_and_end_node_coordinates[1][1]

        if end_X < start_X:
            # reverse BSpline
            start_X, end_X = end_X, start_X
            start_Y, end_Y = end_Y, start_Y

        bspline_ctr_points = []
        bspline_ctr_points.append((start_X, start_Y))

        # first 2
        bspline_ctr_points.append((start_X + 20, start_Y))
        bspline_ctr_points.append((start_X + 25, start_Y))

        # second 2
        bspline_ctr_points.append((start_X + 30, end_Y))
        bspline_ctr_points.append((start_X + 35, end_Y))

        bspline_ctr_points.append((end_X, end_Y))

        bspline_ctr_points = np.array(bspline_ctr_points)

        return bspline_ctr_points

    def _make_b_spline_path(self, ctr_points: np.array) -> List[np.array]:
        """
        With an input of six control points, return an interpolated
        b-spline path which corresponds to a curved edge from one node to another.

        Args:
            ctr_points (np.array): 2d np.array containing the coordinates of the center points.

        Returns:
            List[np.array]: A list of np.arrays that represent one BSpline path. The first list is a list of x-axis coordinates
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
        bspline_path = splev(u3, tck)

        return bspline_path

    def _draw(
        self,
        x_coords: List[float],
        y_coords: List[float],
        colors: List[str],
        sizes: List[float],
        edge_colors: List[str],
        b_splines: np.array,
        func_colors: Dict[str, str],
    ):
        fig, ax = plt.subplots()
        for path, color in zip(b_splines, edge_colors):
            ax.plot(path[0], path[1], color=color, linewidth="3", zorder=-1)

        ax.scatter(x=x_coords, y=y_coords, c=colors, s=sizes, zorder=1)
        legend = []
        for f in func_colors.keys():
            legend.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f,
                    markerfacecolor=func_colors[f],
                    markersize=15,
                )
            )

        plt.legend(handles=legend, loc="upper left")

        ax.set_ylim(min(y_coords) - 0.1, max(y_coords) + 0.1)
        ax.set_yticklabels([])
        ax.yaxis.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Latency (ms)")

        fig.set_figheight(5)
        fig.set_figwidth(10)

        plt.show()
