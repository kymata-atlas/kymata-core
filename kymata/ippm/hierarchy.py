from typing import Optional

from networkx import DiGraph

from kymata.entities.expression import ExpressionPoint

# Maps transforms to lists of names of parent/predecessor/upstream transforms
TransformHierarchy = dict[str, list[str]]


# transform_name → points
PointCloud = dict[str, list[ExpressionPoint]]


class CandidateTransformList:
    """A theoretical IPPM graph, in absence of data."""
    def __init__(self, hierarchy: TransformHierarchy):

        graph = DiGraph()
        # Add nodes
        graph.add_nodes_from(hierarchy.keys())
        # Add edges
        for trans, parents in hierarchy.items():
            for parent_name in parents:
                parent = parent_name
                if parent not in self.transforms:
                    raise ValueError(f"{parent_name=} not in transform list")
                graph.add_edge(parent, trans)

        self.graph: DiGraph = graph

    @property
    def transforms(self) -> set[str]:
        """All transforms."""
        return set(self.graph.nodes)

    @property
    def inputs(self) -> set[str]:
        """Input transforms (those with no predecessors)."""
        # noinspection PyUnresolvedReferences
        return set(t for t in self.graph.nodes if self.graph.in_degree[t] == 0)

    @property
    def terminals(self) -> set[str]:
        """Terminal transforms (those with no successors)."""
        # noinspection PyUnresolvedReferences
        return set(t for t in self.graph.nodes if self.graph.out_degree[t] == 0)


def group_points_by_transform(points: list[ExpressionPoint], ctl: Optional[CandidateTransformList] = None) -> PointCloud:
    d: PointCloud = dict()
    if ctl is not None:
        # Initialise with empty lists for transforms with no points
        for t in ctl.transforms:
            d[t] = []
    for point in points:
        d[point.transform].append(point)
    return d
