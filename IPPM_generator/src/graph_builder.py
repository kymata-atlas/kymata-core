from typing import List, Dict
from collections import namedtuple
from itertools import cycle
from copy import deepcopy

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns

from data_tools import Hexel


# convenient tuple/class to hold information about nodes.
Node = namedtuple('Node', 'magnitude position in_edges')

class IPPMBuilder(object):
    """
        A graphing class used to construct a dictionary that contains the nodes and all relevant
        information to construct a nx.DiGraph.
    """

    def build_graph(self,
                    hexels: Dict[str, Hexel], 
                    function_hier : Dict[str, List[str]], 
                    inputs : List[str], 
                    hemi : str) -> Dict[str, Node]:
        """
            Builds a dictionary of nodes and information about the node. The information
            is built out of namedtuple class Node, which contains magnitude, position, color, and
            the incoming edges.

            Params
            ------
            - hexels : dictionary containing function names and Hexel objects with data in it.
            - function_hier : dictionary of the format (function_name : [children_functions])
            - inputs : list of input functions. function_hier contains the input functions, so we need this to distinguish between inputs and functoins.
            - hemi : leftHemisphere or rightHemisphere

            Returns
            -------
            A dictionary of nodes with unique names where the keys are node objects with all
            relevant information for plotting a nx.DiGraph.
        """
        functions = list(function_hier.keys())
        # filter out functions that are unneccessary
        filtered = {func : hexels[func] for func in functions if func in hexels.keys()}
        # sort it so that the null edges go in the right order (from left to right along time axis)
        sorted = self._sort_by_latency(filtered, hemi)

        hier = deepcopy(function_hier) # we will modify function_hier so copy
        n_partitions = 1 / len(function_hier.keys()) # partition x -axis
        part_idx = 0
        graph = {} # format: node_name : [magnitude, color, position, in_edges]
        while len(hier.keys()) > 0:
            top_level = self._get_top_level_functions(hier)
            for f in top_level:
                if f in inputs:
                    # input node default size is 100.
                    hier.pop(f)
                    graph[f] = Node(100, (0, 1 - n_partitions * part_idx), [])
                
                else:
                    children = hier[f]
                    hier.pop(f)

                    best_pairings = (
                            sorted[f].left_best_pairings if hemi == 'leftHemisphere' else
                                sorted[f].right_best_pairings
                        )
                    
                    if len(best_pairings) == 0:
                        # ignore functions with no spikes at all.
                        continue

                    for idx, pair in enumerate(best_pairings):
                        # add null edges to subsequent spikes in best_pairings.
                        latency, magnitude = pair 
                        graph[f + '-' + str(idx)] = Node(-10 * np.log10(magnitude),
                                                        (latency, 1 - n_partitions * part_idx),
                                                        [f + '-' + str(idx - 1)] if idx != 0 else [])
                    
                    part_idx += 1

                    for child in children:
                        # add edges coming from children to f.
                        if child in inputs:
                            graph[f + '-0'].in_edges.append(child)
                        else:
                            children_pairings = (
                                sorted[child].left_best_pairings if hemi == 'leftHemisphere' else 
                                    sorted[child].right_best_pairings
                                )
                            
                            if len(children_pairings) == 0:
                                # ignore this function
                                continue

                            # add an edge from the final spike of a function.
                            graph[f + '-0'].in_edges.append(child + '-' + str(len(children_pairings) - 1))
        return graph
    
    def _get_top_level_functions(self, edges: Dict[str, List[str]]) -> set:
        """
            Returns the top-level function. A top-level function is at the highest level
            of the function hierarchy. It can be found as the function that does not appear
            in any of the lists of children.

            Params
            ------
                edges : dictionary that contains the function hierarchy (including inputs)
            Returns
            -------
                a set containing the top-level functions.
        """
        funcs_leftover = list(edges.keys())
        children_funcs = [f for children in edges.values() for f in children]
        return set(funcs_leftover).difference(set(children_funcs))

    def _sort_by_latency(self, hexels: Dict[str, Hexel], hemi: str):
        """
            Sort pairings by latency in increasing order inplace.

            Params
            ------
                hexels contains all the functions and hexel objects to sort.
                hemi : leftHemisphere or rightHemisphere
            Returns
            -------
                sorted hexels.
        """
        for key in hexels.keys():
            if hemi == 'leftHemisphere':
                hexels[key].left_best_pairings.sort(key=lambda x: x[0])
            else:
                hexels[key].right_best_pairings.sort(key=lambda x: x[0])

        return hexels
    
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