from copy import deepcopy
from typing import List, Dict

import numpy as np

from .data_tools import IPPMHexel, Node


class IPPMBuilder(object):
    """
    A graphing class used to construct a dictionary that contains the nodes and all relevant
    information to construct a dict containing node names as keys and Node objects (see namedtuple) as values.
    """

    def build_graph(self,
                    hexels: Dict[str, IPPMHexel],
                    function_hier : Dict[str, List[str]],
                    inputs : List[str],
                    hemi : str) -> Dict[str, Node]:
        """
        Builds a dictionary of nodes and information about the node. The information
        is built out of namedtuple class Node, which contains magnitude, position, color, and
        the incoming edges.

        Args:
            hexels (Dict[str, IPPMHexel]): Dictionary containing function names and Hexel objects with data in it.
            function_hier (Dict[str, List[str]]): Dictionary of the format {function_name: [parent_functions]}.
            inputs (List[str]): List of input functions. function_hier contains the input functions, so we need this
                                to distinguish between inputs and functions.
            hemi (str): 'leftHemisphere' or 'rightHemisphere'.

        Returns:
            Dict[str, Node]: A dictionary of nodes with unique names where the keys are node objects with all
                             relevant information for plotting a graph.

        Analysis:
            The overall complexity of the algorithm is O(f * nlogn) where:

             - f is the number of functions.
             - n is the number of hexels.

            The sorting step takes O(f * nlogn).

            Looping through functions and parents contributes O(f * n + f^2).

            Space Complexity:

             - Total space complexity is O(f * n), which includes copies of hexels and function hierarchy.

            The analysis section provides detailed reasoning for the complexities.

        Algorithm:
            The algorithm iteratively selects the top-level function, defined as a function that does not
            have any children (no outgoing arrows). It starts with the final function and proceeds
            in a top-down fashion towards the input node. Each selected top-level function creates a spike for every
            pairing in the best_pairings. Edges are added from the final function of each parent to the first current
            function, repeating for all functions. The input node is treated differently with a default size of 10 at
            latency 0.
        """

        hexels = deepcopy(hexels) # do it in-place to avoid modifying hexels.
        
        # sort it so that the null edges go in the right order (from left to right along time axis)
        sorted = self._sort_by_latency(hexels, hemi, list(function_hier.keys())) 

        hier = deepcopy(function_hier)               # we will modify function_hier so copy
        n_partitions = 1 / len(function_hier.keys()) # partition y-axis
        part_idx = 0                                 # pointer into the current partition out of [0, n_partitions). We go top-down cus of this.
                                                     # it ensures we order our nodes according to their level, with input at the bottom.
        graph = {}                                   # format: node_name : [magnitude, color, position, in_edges]
        while len(hier.keys()) > 0:
            top_level = self._get_top_level_functions(hier) # get function that is a child, i.e., it doesn't have any arrows going out of it
            for f in top_level:
                # We do the following:
                # if f == input_node:
                #     default_settings
                # else:
                #     create_spike_for_every_pairing
                #     for parent in parents:
                #          add spike from final parent node to the first f node. 
                # While doing this, if we encounter empty pairings for f or any parent, we skip them.
                
                if f in inputs:
                    # input node default size is 100.
                    hier.pop(f)
                    graph[f] = Node(100, (0, 1 - n_partitions * part_idx), [])
                
                else:
                    parents = hier[f]     
                    hier.pop(f)        # make sure to pop so we don't get the same top level fs every loop.

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
                    
                    part_idx += 1  # increment partition index

                    for parent in parents:
                        # add edges coming from parents to f.
                        if parent in inputs:
                            # if the parent is a input node, there is only one node.
                            graph[f + '-0'].in_edges.append(parent)
                        else:
                            parent_pairings = (
                                sorted[parent].left_best_pairings if hemi == 'leftHemisphere' else 
                                    sorted[parent].right_best_pairings
                                )
                            
                            if len(parent_pairings) == 0:
                                # ignore this function
                                continue

                            # add an edge from the final spike of parent to first spike of current function.
                            graph[f + '-0'].in_edges.append(parent + '-' + str(len(parent_pairings) - 1))
        return graph
    
    def _get_top_level_functions(self, hier: Dict[str, List[str]]) -> set:
        """
            Returns the top-level function. A top-level function is at the highest level
            of the function hierarchy. It can be found as the function that does not appear
            in any of the lists of parents. I.e., it is a function that that does not feed into
            any other functions; it only has functions feeding into it.

            Params
            ------
                hier : dictionary that contains the function hierarchy (including inputs)
            Returns
            -------
                a set containing the top-level functions.
        """
        funcs_leftover = list(hier.keys())
        parent_funcs = [f for parents in hier.values() for f in parents]
        return set(funcs_leftover).difference(set(parent_funcs))

    def _sort_by_latency(self, hexels: Dict[str, IPPMHexel], hemi: str, functions: List[str]) -> Dict[str, IPPMHexel]:
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
        for function in functions:
            if function not in hexels.keys():
                # function was not detected in the hexels
                continue
            
            if hemi == 'leftHemisphere':
                hexels[function].left_best_pairings.sort(key=lambda x: x[0])
            else:
                hexels[function].right_best_pairings.sort(key=lambda x: x[0])

        return hexels

