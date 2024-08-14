from typing import Dict, List

from kymata.ippm.build import IPPMBuilder
from kymata.ippm.data_tools import IPPMHexel
from kymata.ippm.plot import plot_ippm


class IPPM:
    def __init__(self,
                 hexels: Dict[str, IPPMHexel],
                 inputs: List[str],
                 hierarchy: Dict[str, List[str]],
                 hemisphere: str):
        self._builder = IPPMBuilder(hexels, inputs, hierarchy, hemisphere)

    def create_ippm(self, colors: Dict[str, str], title: str):
        graph = self._builder.build_graph_dict()
        plot_ippm(graph, colors, title=title)
