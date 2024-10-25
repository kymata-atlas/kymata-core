from typing import Dict, List

from kymata.ippm.build import IPPMBuilder
from kymata.ippm.data_tools import SpikeDict
from kymata.ippm.ctl import TransformHierarchy
from kymata.ippm.plot import plot_ippm


class IPPM:
    def __init__(
        self,
        spikes: SpikeDict,
        inputs: List[str],
        hierarchy: TransformHierarchy,
        hemisphere: str,
    ):
        self._builder = IPPMBuilder(spikes, inputs, hierarchy, hemisphere)

    def create_ippm(self, colors: Dict[str, str]):
        plot_ippm(self._builder.graph, colors)
