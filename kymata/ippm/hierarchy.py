from dataclasses import dataclass
from typing import Optional

from networkx import DiGraph


@dataclass(frozen=True)
class TransformMetadata:
    name: str
    equation: Optional[str] = None
    color: str = "#000000"
    commit: Optional[str] = None

    def __str__(self):
        # Hide optional members if not used
        equation_str = f",\tequation='{self.equation}'" if self.equation is not None else ""
        commit_str = f",\tcommit='{self.commit}'" if self.commit is not None else ""
        return f"TransformNode(name='{self.name}'{equation_str},\tcolor='{self.color}'{commit_str})"


# Maps transforms to lists of names of parent transforms
TransformHierarchy = dict[TransformMetadata, list[str]]


class CandidateTransformList:
    def __init__(self, hierarchy: TransformHierarchy):
        nodes_by_name: dict[str, TransformMetadata] = {
            node.name: node
            for node in hierarchy.keys()
        }

        self.graph = DiGraph()
        # Add nodes
        self.graph.add_nodes_from(hierarchy.keys())
        # Add edges
        for trans, parents in hierarchy.items():
            for parent_name in parents:
                parent = nodes_by_name[parent_name]
                if parent not in self.transforms:
                    raise ValueError(f"{parent_name=} not in transform list")
                self.graph.add_edge(parent, trans)

    @property
    def transforms(self) -> set[TransformMetadata]:
        """All transforms."""
        return set(self.graph.nodes)

    @property
    def inputs(self) -> set[TransformMetadata]:
        """Input transforms (those with no predecessors)."""
        # noinspection PyUnresolvedReferences
        return set(t for t in self.graph.nodes if self.graph.in_degree[t] == 0)

    @property
    def terminals(self) -> set[TransformMetadata]:
        """Terminal transforms (those with no successors)."""
        # noinspection PyUnresolvedReferences
        return set(t for t in self.graph.nodes if self.graph.out_degree[t] == 0)
