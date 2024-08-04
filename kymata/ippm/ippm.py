from typing import Dict, List

from kymata.ippm.ippm_builder import IPPMBuilder
from kymata.ippm.data_tools import IPPMHexel
from kymata.ippm.ippm_plotter import IPPMPlotter


class IPPM:
    def __init__(self,
                 hexels: Dict[str, IPPMHexel],
                 inputs: List[str],
                 hierarchy: Dict[str, List[str]],
                 hemisphere: str):
        self._builder = IPPMBuilder(hexels, inputs, hierarchy, hemisphere)
        self._plotter = IPPMPlotter()

    def create_ippm(self, colors: Dict[str, str]):
        graph = self._builder.build_graph_dict()
        self._plotter.plot(graph, colors)
