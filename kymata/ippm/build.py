"""
A graphing functions used to construct a dictionary that contains the nodes and all relevant
information to construct a dict containing node names as keys and Node objects (see namedtuple) as values.
"""
from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np

from kymata.entities.constants import HEMI_RIGHT
from kymata.ippm.data_tools import IPPMSpike, IPPMNode


class IPPMBuilder:
    def __init__(
        self,
        spikes: Dict[str, IPPMSpike],
        inputs: List[str],
        hierarchy: Dict[str, List[str]],
        hemisphere: str,
    ):
        self._spikes = deepcopy(spikes)
        self._inputs = inputs
        self._hierarchy = deepcopy(hierarchy)
        self._hemisphere = hemisphere
        self._graph = {}  # Chosen structure is Dict[str, Node] since that enables quick look up for node position

    def build_graph_dict(self) -> Dict[str, IPPMNode]:
        self._spikes = self._sort_spikes_by_latency_asc()

        y_axis_partition_size = (
            1 / len(self._hierarchy.keys()) if len(self._hierarchy.keys()) > 0 else 1
        )
        partition_ptr = 0
        while childless_functions := self._get_childless_functions():
            for childless_func in childless_functions:
                self._graph = self._create_nodes_and_edges_for_function(
                    childless_func, partition_ptr, y_axis_partition_size
                )
                self._hierarchy.pop(childless_func)
                partition_ptr += 1

        return self._graph

    def _sort_spikes_by_latency_asc(self) -> Dict[str, IPPMSpike]:
        for function in self._spikes.keys():
            if self._hemisphere == HEMI_RIGHT:
                self._spikes[function].right_best_pairings.sort(key=lambda x: x[0])
            else:
                self._spikes[function].left_best_pairings.sort(key=lambda x: x[0])
        return self._spikes

    def _get_childless_functions(self):
        def __unpack_dict_values_into_list(dict_to_unpack):
            return [value for values in dict_to_unpack.values() for value in values]

        current_functions = set(self._hierarchy.keys())
        functions_with_children = set(__unpack_dict_values_into_list(self._hierarchy))
        # When no functions left, it returns empty set.
        return current_functions.difference(functions_with_children)

    def _create_nodes_and_edges_for_function(
        self, function_name: str, partition_ptr: int, partition_size: float
    ) -> Dict[str, IPPMNode]:
        def __get_y_coordinate(
            curr_partition_number: int, partition_size: float
        ) -> float:
            return 1 - partition_size * curr_partition_number

        func_parents = self._hierarchy[function_name]
        current_y_axis_coord = __get_y_coordinate(partition_ptr, partition_size)
        if function_name in self._inputs:
            self._graph[function_name] = IPPMNode(100, (0, current_y_axis_coord), [])
        else:
            childless_func_pairings = self._get_best_pairings_from_hemisphere(
                function_name
            )
            self._graph = self._create_nodes_for_childless_function(
                current_y_axis_coord, childless_func_pairings, function_name
            )
            self._graph = self._create_edges_between_parents_and_childless_function(
                func_parents, function_name
            )

        return self._graph

    def _get_best_pairings_from_hemisphere(
        self, func: str
    ) -> List[Tuple[float, float]]:
        if func in self._spikes.keys():
            return (
                self._spikes[func].right_best_pairings
                if self._hemisphere == HEMI_RIGHT
                else self._spikes[func].left_best_pairings
            )
        return []

    def _create_nodes_for_childless_function(
        self,
        current_y_axis_coord: float,
        childless_func_pairings: List[Tuple[float, float]],
        function_name: str,
    ):
        def __map_magnitude_to_node_size(magnitude: float) -> float:
            return -10 * np.log10(magnitude)

        for idx, pair in enumerate(childless_func_pairings):
            latency, magnitude = pair
            parent_function = [function_name + "-" + str(idx - 1)] if idx != 0 else []
            self._graph[function_name + "-" + str(idx)] = IPPMNode(
                __map_magnitude_to_node_size(magnitude),
                (latency, current_y_axis_coord),
                parent_function,
            )

        return self._graph

    def _create_edges_between_parents_and_childless_function(
        self,
        parents: List[str],
        function_name: str,
    ) -> Dict[str, IPPMNode]:
        if function_name in self._spikes.keys():
            for parent in parents:
                if parent in self._inputs:
                    self._graph[function_name + "-0"].inc_edges.append(parent)
                else:
                    parent_pairings = self._get_best_pairings_from_hemisphere(parent)
                    if len(parent_pairings) > 0:
                        self._graph[function_name + "-0"].inc_edges.append(
                            parent + "-" + str(len(parent_pairings) - 1)
                        )

        return self._graph
