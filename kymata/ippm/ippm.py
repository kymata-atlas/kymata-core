from typing import Dict, List

from kymata.ippm.build import build_graph_dict
from kymata.ippm.data_tools import IPPMHexel
from kymata.ippm.plot import plot_ippm


class IPPM:

    def create_ippm(self, hexels, function_hierarchy, inputs, hemisphere, colors: Dict[str, str], title: str):
        graph = build_graph_dict(hexels, function_hierarchy, inputs, hemisphere, )
        plot_ippm(graph, colors, title=title)
