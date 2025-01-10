from typing import Optional

from networkx import DiGraph

from kymata.entities.expression import ExpressionPoint

# Maps transforms to lists of names of parent/predecessor/upstream transforms
TransformHierarchy = dict[str, list[str]]


# transform_name → points
GroupedPoints = dict[str, list[ExpressionPoint]]


class CandidateTransformList:
    """A theoretical IPPM graph, in absence of data."""
    """
    Candidate Transform List ("CTL"). A theoretical IPPM graph, constructed in the absence of data.

    This class models the structure of an IPPM graph using a directed graph (DiGraph) where nodes represent
    transforms, and edges represent dependencies between them. It is used to define a hypothetical set of transforms
    and their relationships prior to applying them to actual data.

    Args:
        hierarchy (dict[str, list[str]]): A dictionary representing the relationships between transforms, where
            each transform is associated with a list of parent transforms.

    Attributes:
        graph (DiGraph): A directed graph representing the transforms and their dependencies.

    Methods:
        transforms: Returns all transforms in the CTL.
        inputs: Returns input transforms (those with no predecessors).
        terminals: Returns terminal transforms (those with no successors).
        serial_sequence: Returns a serial sequence of transforms grouped by parallelism.
    """
    def __init__(self, hierarchy: TransformHierarchy):
        """
        Initializes a CandidateTransformList from a given hierarchy.

        Args:
            hierarchy (TransformHierarchy): A dictionary of transforms and their parent–child relationships.
        """

        graph = DiGraph()
        # Add nodes
        graph.add_nodes_from(hierarchy.keys())
        # Add edges
        for trans, parents in hierarchy.items():
            for parent_name in parents:
                parent = parent_name
                if parent not in graph.nodes:
                    raise ValueError(f"{parent_name=} not in transform list")
                graph.add_edge(parent, trans)

        self.graph: DiGraph = graph

    @property
    def transforms(self) -> set[str]:
        """
        All transforms in the CTL.

        Returns:
            set[str]: A set of all transform names.
        """
        return set(self.graph.nodes)

    @property
    def inputs(self) -> set[str]:
        """
        Input transforms (those with no predecessors) in the CTL.

        Returns:
            set[str]: A set of input transform names (transforms with no predecessors).
        """
        # noinspection PyUnresolvedReferences
        return set(t for t in self.graph.nodes if self.graph.in_degree[t] == 0)

    @property
    def terminals(self) -> set[str]:
        """
        Terminal transforms (those with no successors) in the CTL.

        Returns:
            set[str]: A set of terminal transform names (transforms with no successors).
        """
        # noinspection PyUnresolvedReferences
        return set(t for t in self.graph.nodes if self.graph.out_degree[t] == 0)

    @property
    def serial_sequence(self) -> list[list[str]]:
        """
        The serial sequence of parallel transforms.

        Returns a sequence of transforms, where each batch in the sequence represents
        parallelizable transforms, ordered by serial dependency.

        Returns:
            list[list[str]]: A list of lists representing batches of parallel transforms in execution order.
        """
        # Add input nodes
        seq = [sorted(self.inputs)]
        # Recursively add children
        all_transforms = self.transforms - self.inputs
        while len(all_transforms) > 0:
            batch = set()
            # Previous step
            for transform in seq[-1]:
                for successor in  self.graph.successors(transform):
                    batch.add(successor)
                    try:
                        all_transforms.remove(successor)
                    except KeyError:
                        pass
            seq.append(sorted(batch))
        return seq


def group_points_by_transform(points: list[ExpressionPoint],
                              ctl: Optional[CandidateTransformList] = None,
                              ) -> GroupedPoints:
    """
    Groups a list of expression points by their associated transforms.

    This function organizes expression points into a dictionary, where each key is a transform, and each value is a list
    of expression points associated with that transform.

    If a CTL is provided, the function will initialize the dictionary with empty lists for all transforms in the CTL,
    ensuring all transforms are represented, even if no expression points are associated with them.

    Args:
        points (list[ExpressionPoint]): A list of expression points to be grouped.
        ctl (Optional[CandidateTransformList]): An optional transform list to initialize the dictionary with empty lists
            for all transforms.

    Returns:
        GroupedPoints: A dictionary mapping transform names to lists of expression points associated with each
            transform.
    """
    d: GroupedPoints = dict()
    if ctl is not None:
        # Initialise with empty lists for transforms with no points
        for t in ctl.transforms:
            d[t] = []
    for point in points:
        d[point.transform].append(point)
    return d
