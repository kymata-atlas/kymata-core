from itertools import cycle
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.lines import Line2D

from kymata.ippm.builder import Node


class IPPMPlotter(object):
    """
        Use this class to plot a nx.DiGraph. Run the IPPMBuilder class prior to obtain the prerequisite
        dictionary of nodes, edges, sizes, colors, and positions. 
    """

    def plot(self, 
            graph : Dict[str, Node], 
            colors : Dict[str, str], 
            title : str, 
            x_axis: str='Latency (ms)',
            figheight : int=12, 
            figwidth : int=15) -> nx.DiGraph:
        """
            Plot a directed, acyclic graph representing the flow of information as specified by graph.

            Params
            ------
            - graph : a dictionary containing all nodes and node information. Node has attributes magnitude, position, color, incoming_edges
            - colors : a dictionary containing the color for each function
            - title : title of plot
            - figheight : figure height
            - figwidth : figure width

            Returns
            -------
            A nx.DiGraph object with all of the edges and nodes. Returned primarily for testing purposes.
        """
        nx_graph = nx.DiGraph()
        pos = {}
        for node, node_data in graph.items():
            nx_graph.add_node(node)
            pos[node] = node_data.position
            for edge in node_data.in_edges:
                nx_graph.add_edge(edge, node)

        color_map = [_ for _ in range(len(nx_graph.nodes))]
        size_map = [_ for _ in range(len(nx_graph.nodes))]
        rand_colors = cycle(sns.color_palette('bright', 100).as_hex())
        for i, node in enumerate(nx_graph.nodes):
            # need to do it afterwards, so the ordering of colors/sizes lines up with
            # the ordering of nodes.
            color_map[i] = None
            for function, color in colors.items():
                if function in node:
                    # function name is substring of node = found
                    color_map[i] = color
                    break
            if color_map[i] is None:
                color_map[i] = next(rand_colors)

            size_map[i] = graph[node].magnitude
        
        fig, ax = plt.subplots()
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        nx.draw(nx_graph, pos=pos, node_color=color_map, node_size=size_map, ax=ax)
        plt.axis('on')
        ax.tick_params(bottom=True, labelbottom=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Time (ms)')

        legend = []
        for f in colors.keys():
            legend.append(Line2D([0], [0], marker='o', color='w', label=f, markerfacecolor=colors[f], markersize=15))

        plt.legend(handles=legend, loc='upper left')
        plt.title(title)

        return nx_graph