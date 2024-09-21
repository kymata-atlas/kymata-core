"""
A graphing functions used to construct a dictionary that contains the nodes and all relevant
information to construct a dict containing node names as keys and Node objects (see namedtuple) as values.
"""
from collections import defaultdict, Counter
from copy import deepcopy
from enum import StrEnum
from typing import NamedTuple

import numpy as np

from kymata.entities.constants import HEMI_RIGHT
from kymata.ippm.data_tools import SpikeDict, TransformHierarchy, ExpressionPairing


class NodePosition(NamedTuple):
    x: float
    y: float


class IPPMNode(NamedTuple):
    """
    A node to be drawn in an IPPM graph.
    """
    magnitude: float
    position: NodePosition
    inc_edges: list


# Maps function names to nodes
IPPMGraph = dict[str, IPPMNode]


class _YOrdinateMethods(StrEnum):
    progressive = "progressive"
    centred     = "centred"


class IPPMBuilder:
    def __init__(
        self,
        spikes: SpikeDict,
        inputs: list[str],
        hierarchy: TransformHierarchy,
        hemisphere: str,
        y_ordinate_method: str = _YOrdinateMethods.progressive,
        levels: dict[str, int] = None
    ):
        self._spikes: SpikeDict = deepcopy(spikes)
        self._inputs: list[str] = inputs
        self._hierarchy: TransformHierarchy = hierarchy
        self._hemisphere: str = hemisphere

        self._sort_spikes_by_latency_asc()

        self.graph: IPPMGraph = dict()
        self.graph = self._build_graph_dict(deepcopy(self._hierarchy), y_ordinate_method, levels)

    def _build_graph_dict(self,
                          hierarchy: TransformHierarchy,
                          y_ordinate_method: str,
                          levels: dict[str, int],
                          ) -> IPPMGraph:
        """
        y_ordinate_method == "progressive" for y ordinates to be selected progressively from the input
        y_ordinate_method == "centred" for y ordinates to be centred vertically based on assigned levels in the
                             hierarchy
        levels: maps node names in the hierarchy to level-idxs of vertically centred nodes
        """
        if y_ordinate_method == _YOrdinateMethods.progressive:
            y_axis_partition_size = (
                1 / len(hierarchy.keys()) if len(hierarchy.keys()) > 0 else 1
            )
            partition_ptr = 0
            graph = dict()
            while childless_functions := self._get_childless_functions(hierarchy):
                for childless_func in childless_functions:
                    graph = self._create_nodes_and_edges_for_function_progressive(
                        childless_func, partition_ptr, y_axis_partition_size
                    )
                    hierarchy.pop(childless_func)
                    partition_ptr += 1

        elif y_ordinate_method == _YOrdinateMethods.centred:
            if levels is None:
                raise ValueError(f"Supply `levels` when using {_YOrdinateMethods.centred} option")
            totals_within_level = Counter(levels.values())
            idxs_within_level = defaultdict(int)
            graph = dict()
            while childless_functions := self._get_childless_functions(hierarchy):
                for childless_func in childless_functions:
                    graph = self._create_nodes_and_edges_for_function_centred(
                        childless_func,
                        idxs_within_level[childless_func],
                        totals_within_level[levels[childless_func]],
                        max(totals_within_level.values()),
                    )
                    hierarchy.pop(childless_func)
                    idxs_within_level[childless_func] += 1

        else:
            raise NotImplementedError()

        return graph

    def _sort_spikes_by_latency_asc(self) -> None:
        for function in self._spikes.keys():
            if self._hemisphere == HEMI_RIGHT:
                self._spikes[function].right_best_pairings.sort(key=lambda x: x[0])
            else:
                self._spikes[function].left_best_pairings.sort(key=lambda x: x[0])

    @classmethod
    def _get_childless_functions(cls, hierarchy: TransformHierarchy) -> set[str]:
        def __unpack_dict_values_into_list(dict_to_unpack):
            return [value for values in dict_to_unpack.values() for value in values]

        current_functions = set(hierarchy.keys())
        functions_with_children = set(__unpack_dict_values_into_list(hierarchy))
        # When no functions left, it returns empty set.
        return current_functions.difference(functions_with_children)

    def _create_nodes_and_edges_for_function_centred(self,
                                                     function_name: str,
                                                     function_idx_within_level: int,
                                                     function_total_within_level: int,
                                                     max_function_total_within_level: int,
                                                     ) -> IPPMGraph:
        """
        x_batch_size: how many nodes in a vertically-centred batch.
        x_batch_idx: which node this is in the batch
        """


    def _create_nodes_and_edges_for_function_progressive(self,
                                                         function_name: str,
                                                         partition_ptr: int,
                                                         partition_size: float,
                                                         ) -> IPPMGraph:
        def __get_y_coordinate(curr_partition_number: int, partition_size: float) -> float:
            return 1 - partition_size * curr_partition_number

        func_parents = self._hierarchy[function_name]
        current_y_axis_coord = __get_y_coordinate(partition_ptr, partition_size)
        if function_name in self._inputs:
            self.graph[function_name] = IPPMNode(100, NodePosition(0, current_y_axis_coord), [])
        else:
            childless_func_pairings = self._get_best_pairings_from_hemisphere(
                function_name
            )

            if len(childless_func_pairings) == 0:
                return self.graph

            self.graph = self._create_nodes_for_childless_function(
                current_y_axis_coord, childless_func_pairings, function_name
            )
            self.graph = self._create_edges_between_parents_and_childless_function(
                func_parents, function_name
            )

        return self.graph

    def _get_best_pairings_from_hemisphere(self, func: str) -> list[ExpressionPairing]:
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
        childless_func_pairings: list[ExpressionPairing],
        function_name: str,
    ):
        def __map_magnitude_to_node_size(magnitude: float) -> float:
            return -10 * np.log10(magnitude)

        for idx, pair in enumerate(childless_func_pairings):
            latency, magnitude = pair
            parent_function = [function_name + "-" + str(idx - 1)] if idx != 0 else []
            self.graph[function_name + "-" + str(idx)] = IPPMNode(
                __map_magnitude_to_node_size(magnitude),
                NodePosition(latency, current_y_axis_coord),
                parent_function,
            )

        return self.graph

    def _create_edges_between_parents_and_childless_function(
        self,
        parents: list[str],
        function_name: str,
    ) -> IPPMGraph:
        if function_name in self._spikes.keys():
            for parent in parents:
                if parent in self._inputs:
                    self.graph[function_name + "-0"].inc_edges.append(parent)
                else:
                    parent_pairings = self._get_best_pairings_from_hemisphere(parent)
                    if len(parent_pairings) > 0:
                        self.graph[function_name + "-0"].inc_edges.append(
                            parent + "-" + str(len(parent_pairings) - 1)
                        )

        return self.graph
