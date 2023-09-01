
from data_tools import Hexel
from copy import deepcopy
import numpy as np
from typing import List, Dict
from collections import namedtuple

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
                    # input node default size is 10.
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

