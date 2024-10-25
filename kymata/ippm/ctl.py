from dataclasses import dataclass
from typing import Optional

from networkx import DiGraph


# Maps function names to lists of parent functions
TransformHierarchy = dict[str, list[str]]


@dataclass
class Transform:
    name: str
    equation: Optional[str] = None
    color: str = "#000000"


class CandidateTransformList:
    def __init__(self, hierarchy: TransformHierarchy,
                 color_overrides: Optional[dict[str, str]] = None,
                 equation_overrides: Optional[dict[str, str]] = None):
        if color_overrides is None:
            color_overrides = dict()
        if equation_overrides is None:
            equation_overrides = dict()

        self.graph = DiGraph()

        # Add nodes
        for trans, parent_trans in hierarchy.items():
            node = Transform(trans)
            if trans in color_overrides:
                node.color = color_overrides[trans]
            if trans in equation_overrides:
                node.equation = equation_overrides[trans]
            self.graph.add_node(node)
        # Add edges
        for trans, parent_trans in hierarchy.items():
            for parent in parent_trans:
                if parent not in self.graph:
                    raise ValueError(f"{parent=} not in transform list")
                self.graph.add_edge(parent, trans)

    @property
    def transforms(self) -> set[Transform]:
        """All transforms."""
        return set(self.graph.nodes)

    @property
    def inputs(self) -> set[Transform]:
        """Input transforms (those with no predecessors)."""
        return set(t for t in self.graph.nodes if self.graph.in_degree(t) == 0)

    @property
    def terminals(self) -> set[Transform]:
        """Terminal transforms (those with no successors)."""
        return set(t for t in self.graph.nodes if self.graph.out_degree(t) == 0)
