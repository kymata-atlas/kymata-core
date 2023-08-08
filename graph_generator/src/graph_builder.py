
from data_tools import Hexel
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy
import numpy as np
from typing import List, Dict, Tuple

class GraphBuilder(object):
    """
        A graphing class used to construct a directed graph (via Networkx) using denoised hexel spikes.
        The Y-axis is based off latency while the X-axis partitions into classes (functions).
        Each node is scaled by the magnitude of the spike.
    """

    def draw(
                self, 
                hexels : Dict[str, Hexel], 
                function_hier : Dict[str, List[str]], 
                inputs : List[str], 
                hemi : str, 
                title : str, 
                figheight : int=12, 
                figwidth : int=15
            ):
        """
            Draws the directed graph.

            Params
            ------
                hexels : dictionary containing function names and Hexel objects with data in it.
                function_hier : dictionary of the format (function_name : [children_functions])
                inputs : list of input functions. function_hier contains the input functions, so we need this to distinguish between inputs and functoins.
                hemi : leftHemisphere or rightHemisphere
                title : title
                figheight : height
                figwidth : width
        """

        # filter only to functions in edges.
        functions = list(function_hier.keys())
        filtered = {func : hexels[func] for func in functions if func in hexels.keys()}
        sorted = self._sort_by_latency(filtered, hemi) # sort by latency, so you can construct null edges.

        edges_2 = deepcopy(function_hier)           # we pop a function from edges_2 once plotted. Hence, copy.
        
        n_partitions = 1 / len(function_hier.keys()) # partition x -axis
        part_idx = 0
        graph = nx.DiGraph()
        pos = {} # holds positions of nodes. (x-axis, y-axis)
        while len(edges_2.keys()) > 0:
            # top level node = no outgoing edges (not a child of any curr function).
            top_level = self._get_top_level_functions(edges_2)

            for f in top_level:
                if f in inputs:
                    # input function, so no null edges.
                    edges_2.pop(f)
                    graph.add_node(f)
                    #pos[f] = (1 - n_partitions * part_idx, 0)
                    pos[f] = (0, 1 - n_partitions * part_idx)
                else:
                    children = edges_2[f]
                    edges_2.pop(f)

                    # we need to add null edges to subsequent spikes.
                    best_pairings = (
                            sorted[f].left_best_pairings if hemi == 'leftHemisphere' else
                                sorted[f].right_best_pairings
                        )

                    if len(best_pairings) == 0:
                        continue

                    for idx, pair in enumerate(best_pairings):
                        # add an edge between best pairings, representing a null edge.
                        latency, _ = pair
                        name = f + '-' + str(idx)
                        graph.add_node(name)
                        if idx != 0:
                            graph.add_edge(f + '-' + str(idx - 1), name)

                        #pos[name] = (1 - n_partitions * part_idx, latency)
                        pos[name] = (latency, 1 - n_partitions * part_idx)
                    part_idx += 1
                    # add edges to children from the latest spike.
                    for child in children:
                        if child in inputs:
                            # input only have edge to input_0 (since there are no null edges for inputs).
                            graph.add_edge(child, f + '-0')
                        else:
                            children_pairings = (
                                    sorted[child].left_best_pairings if hemi == 'leftHemisphere' else 
                                        sorted[child].right_best_pairings
                                )
                                
                            if len(children_pairings) == 0:
                                continue 
                            # add an edge to the last spike of child class. I.e., the highest value in class column.
                            # spike goes from child to current function.
                            graph.add_edge(child + '-' + str(len(children_pairings) - 1), f + '-0')
        
        # Need to set up node sizes and colors.
        # set up color scheme for nodes.
        colors = sns.color_palette('bright', len(functions)).as_hex()
        f_colors = {f: color for f, color in zip(function_hier.keys(), colors)}
        node_sizes, color_map = self._set_node_attributes(graph, sorted, f_colors, hemi)
        
        fig, ax = plt.subplots()
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        nx.draw(graph, pos=pos, node_color=color_map, node_size=node_sizes, ax=ax)
        limits = plt.axis('on')
        ax.tick_params(bottom=True, labelbottom=True) # we want y-axis to be visible
        ax.spines['top'].set_visible(False)       # remove borders around figure
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Time (ms)')

        legend = []
        for function in function_hier.keys():
            # add legend to distinguish classes
            color = f_colors[function]
            legend.append(Line2D([0], [0], marker='o', color='w', label=function,markerfacecolor=color, markersize=15))

        plt.legend(handles=legend, loc='upper left')
        plt.title(title)
        return graph
        
    def _set_node_attributes(self, 
                            graph: nx.Graph, 
                            sorted: Dict[str, Hexel], 
                            f_colors: Dict[str, str], 
                            hemi: str
                        ) -> Tuple[List[float], List[str]]:
        """
            Use this to set the node size and color for each node in graph.

            Params
            ------
                graph : networkx graph object. It contains nodes and edges.
                sorted : dictionary of hexels but with sorted latencies.
                f_colors : dictionary of function names as keys and colors (as hex) as values
                hemi : leftHemisphere or rightHemisphere

            Returns
            -------
                the node size and color for each node.
        """
        node_sizes = []
        color_map = []
        for node in graph.nodes:
            if '-' in node:
                # not input node.
                func, idx = node[:node.index('-')], int(node[node.index('-') + 1:])
                best_pairings = (
                        sorted[func].left_best_pairings if hemi == 'leftHemisphere' else
                            sorted[func].right_best_pairings
                    )
                # scale with size with spike. Smaller = better. Take log10 to get power. It is 
                # negative so multiply by -1. To make size more noticeable, scale it by 10.
                node_sizes.append(-10 * np.log10(best_pairings[idx][1]))
            else:
                # default size for input.
                node_sizes.append(100)

            function = node[:node.index('-')] if '-' in node else node
            color_map.append(f_colors[function]) # save color from dict.

        return (node_sizes, color_map)
    

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