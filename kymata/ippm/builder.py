from copy import deepcopy
from typing import List, Dict
from collections import namedtuple

import numpy as np

from .data_tools import IPPMHexel

# convenient tuple/class to hold information about nodes.
Node = namedtuple('Node', 'magnitude position in_edges')


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
            the incoming edges

            Analysis
            --------
            
                sorting takes nlogn where n = # of hexels = 10000. assumption: quicksort == sort
                we do it for every f, so f * nlogn
                next we loop through every f and touch every pairing and parent. In the worst case,
                the number of pairings == # of hexels, and # of parent == # of fs - 1. Hence, this
                part has O(f * (n + f-1)) = O(f * n + f^2).

                Total complexity: O(f * nlogn + f * n + f^2) 
                What dominates: f * nlogn ~ f * n. if logn > 1, then f * nlogn > f * n.
                                We have n = 10,000. So logn > 1, hence f * nlogn > f * n.
                                As long as n >= 10, logn >= 1. Since this is big-O, we take worst
                                case but in reality, the # of pairings is typically from 0-10.
                                So, f * n >= f * nlogn but big-O is worst-case, so we stick with nlogn.

                                f * nlogn ~ f^2. if nlogn > f, f * nlogn > f^2.
                                assuming n = 10,000, nlogn = 40000 > f = 12.
                                         f = 12
                                Hence, f * nlogn is actually the dominant term.
                                Can we reduce it? It would involve updating the algorithm to avoid
                                sorting prior to building the graph. The primary we reason to sort
                                is so that we can exploit the naming of functions being ordered and
                                immediately add an edge from the last pairing of a parent to the first
                                pairing of a child. Without it being sorted, we would have to loop 
                                through the parent pairings to locate the last node. Hence, 
                                the complexity becomes: O(f * (n - (f - 1) * n)) = O(f * n - f^2 * n).
                                Now f^2 * n > nlogn, so it would actually make the algorithm slower.
                                Moreover, we took an unrealistic worst case assumption of n = 10,000.
                                In reality, it would be between 0-10, so nlogn would be approximately less
                                than or equal to f. So, in practice, it would not dominate. Especially 
                                as the dataset quality improves, the number of spikes would go down and
                                the number of functions would increase.

                                Therefore, we cannot reduce the complexity further. 
                                Final complexity: O(f * nlogn).


                Space Complexity: Let n be the maximum pairs of spikes out of all functions. 
                    We copy hexels, so O(n * f)
                    We copy func hier, so O(f * (f-1)) = O(f^2)
                    Our dict will contain a key for every pairing. Hence, it will be of size O(f * n).
                    Total space: O(n * f + f^2 + f * n) = O(f * n).

                We could feasibly trade-off time for space complexity but I think it is good as it is.

            Algorithm
            ---------

            It iteratively selects the top-level function, which is defined as a function that does not
            have any children, i.e., it does not have any arrows going out of it. Hence, it starts with the final function and proceeds
            in a top-down fashion towards the input node. Upon selecting a top-level function, it creates a spike for every pairing in
            the best_pairings. Since the spikes are already ordered, we get a nice labelling where func_name-0 corresponds to the earliest
            spike and func_name-{len(pairings)}-1 is the final spike. Next, we go through the parents (incoming edges) and add an edge
            from the final function (i.e., inc_edge_func_name-{len(pairings)}-1) to the first current function (func_name-0). We repeat this
            for all functions. Last thing to note is that the input node has to be defined because it is treated differently to the rest. The
            input function has only 1 spike and a default size of 10 at latency == 0.

            We do it top-down to make the ordering of nodes clean. Otherwise, we can get a messy jumble of nodes with the input in the middle and 
            the final output randomly assigned. 

            Params
            ------
            - hexels : dictionary containing function names and Hexel objects with data in it.
            - function_hier : dictionary of the format (function_name : [parent_functions])
            - inputs : list of input functions. function_hier contains the input functions, so we need this to distinguish between inputs and functions.
            - hemi : leftHemisphere or rightHemisphere

            Returns
            -------
            A dictionary of nodes with unique names where the keys are node objects with all
            relevant information for plotting a graph.
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

