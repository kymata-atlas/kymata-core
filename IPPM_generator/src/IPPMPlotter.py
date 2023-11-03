from collections import namedtuple
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import splev


# convenient tuple/class to hold information about nodes.
Node = namedtuple('Node', 'magnitude position in_edges')


class IPPMPlotter(object):
    def draw(self, 
             graph: Dict[str, Node],
             colors: Dict[str, str],
             title: str,
             figheight: int=5,
             figwidth: int=10):
        """
            Generates an acyclic, directed graph using the graph held in graph. Edges are generated using BSplines.

            Parameters
            ----------
                graph dictionary with keys as node names and values as Hexel objects.
                    contains nodes as keys and magnitude, position, and incoming edges in the Hexel object.
                colors dictionary with keys as node names and values as colors in hexadecimal.
                    contains the color for each function. The nodes and edges are colored accordingly.
                title string
                    title of the plot
                figheight int
                    height
                figwidth int
                    width
        """
        # first lets aggregate all of the information.
        # TODO: refactor to generate BSplines in the first loop, so we dont have to loop again.
        hexel_x = [_ for _ in range(len(graph.keys()))]                  
        hexel_y = [_ for _ in range(len(graph.keys()))]                 
        node_colors = [_ for _ in range(len(graph.keys()))]              
        node_sizes = [_ for _ in range(len(graph.keys()))]
        hexel_coordinate_pairs = []  # [[start_coord, end_coord], ..]
        edge_colors = []
        for i, node in enumerate(graph.keys()):
            for function, color in colors.items():
                # search for function color. TODO: handle missing colors.
                if function in node:
                    node_colors[i] = color
                    break

            node_sizes[i] = graph[node].magnitude
            hexel_x[i] = graph[node].position[0]
            hexel_y[i] = graph[node].position[1]
            
            for inc_edge in graph[node].in_edges:
                # save edge coordinates and color the edge the same color as the finishing node.
                start = graph[inc_edge].position
                end = graph[node].position
                hexel_coordinate_pairs.append([(start[0], start[1]), (end[0], end[1])])
                edge_colors.append(node_colors[i])

        bspline_path_array = self._make_bspline_paths(hexel_coordinate_pairs)
        
        fig, ax = plt.subplots()
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        plt.axis('on')
        for path, color in zip(bspline_path_array, edge_colors):
            ax.plot(path[0], path[1], color=color, linewidth='3', zorder=-1)
        ax.scatter(x=hexel_x, y=hexel_y, c=node_colors, s=node_sizes, zorder=1)
        ax.tick_params(bottom=True, labelbottom=True, left=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Latency (ms)')

        legend = []
        for f in colors.keys():
            legend.append(Line2D([0], [0], marker='o', color='w', label=f, markerfacecolor=colors[f], markersize=15))

        plt.legend(handles=legend, loc='upper left')
        plt.title(title)

    def _make_bspline_paths(self, hexel_coordinate_pairs: List[List[Tuple[float, float]]]) -> List[List[np.array]]:
        """
            Given a list of hexel positions pairs, return a list of
            b-splines. First, find the control points, and second
            create the b-splines from these control points.

            Parameters
            ----------
                hexel_coordinate_pairs a list of a list of np arrays
                    each list contains the x-axis values and y-axis values for the start and end of a BSPpline.
                    E.g., [(0, 1), (1, 0)]
            
            Returns
            -------
                a list of a list of np arrays. Each list contains two np.arrays. The first np.array contains the x-axis points and the second one
                contains the y-axis points. Together, they define a BSpline. Thus, it is a list of BSplines.

        """
        bspline_path_array = []
        for pair in hexel_coordinate_pairs:
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
                ctr_pts = self._make_bspline_ctr_points(pair)
                bspline_path_array.append(self._make_bspline_path(ctr_pts))

        return bspline_path_array

    def _make_bspline_ctr_points(self, start_and_end_node_coordinates: List[Tuple[float, float]]) -> np.array:

        """
            Given the position of a start hexel and an end hexel, create
            a set of 6 control points needed for a b-spline.

            The first one and last one is the position of a start hexel
            and an end hexel themselves, and the intermediate four are
            worked out using some simple rules.

            Parameters
            ----------
                start_and_end_coordinates list containing the start and end coordinates for one edge.
                    first tuple is start, second is end. First element in tuple is x coord, second is y coord.
            
            Returns
            -------
                np.array that has a list of tuples of coordinates. Each coordinate pair represents a control point.
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
        bspline_ctr_points.append((start_X,start_Y))

        # first 2
        bspline_ctr_points.append((start_X + 20, start_Y))
        bspline_ctr_points.append((start_X + 25, start_Y))

        # second 2
        bspline_ctr_points.append((start_X + 30, end_Y))
        bspline_ctr_points.append((start_X + 35, end_Y))

        bspline_ctr_points.append((end_X, end_Y))

        bspline_ctr_points = np.array(bspline_ctr_points)

        return bspline_ctr_points

    def _make_bspline_path(self, ctr_points: np.array) -> List[np.array]:

        """
            With an input of six control points, return an interpolated
            b-spline path which corresponds to a curved edge from one node to another.

            Parameters
            ----------
                ctr_points np.array
                    2d np.array containing the coordinates of the center points. 

            Returns
            -------
                list of np.arrays that represent one BSpline path. The first list is a list of x-axis coordinates
                the second is alist of y-axis coordinates.
        """

        x = ctr_points[:, 0]
        y = ctr_points[:, 1]

        l = len(x)
        t = np.linspace(0, 1, l - 2, endpoint=True)
        t = np.append([0, 0, 0], t)
        t = np.append(t, [1, 1, 1])

        tck = [t, [x, y], 3]
        u3 = np.linspace(0, 1, (max(l * 2, 70)), endpoint=True)
        bspline_path = splev(u3, tck)

        return bspline_path