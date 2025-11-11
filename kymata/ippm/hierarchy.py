from __future__ import annotations

from networkx import DiGraph
from networkx.algorithms.components import weakly_connected_components


# Maps transforms to lists of names of parent/predecessor/upstream transforms
TransformHierarchy = dict[str, list[str]]

# A serial sequence of parallel transforms
SerialSequence = tuple[frozenset[str], ...]


class CandidateTransformList:
    """A theoretical IPPM graph, in absence of data.

    Candidate Transform List ("CTL"). A theoretical IPPM graph, constructed in the absence of data.

    This class models the structure of an IPPM graph using a directed graph (DiGraph) where nodes represent
    transforms, and edges represent dependencies between them. It is used to define a hypothetical set of transforms
    and their relationships prior to applying them to actual data.

    Args:
        hierarchy (dict[str, list[str]]): A dictionary representing the relationships between transforms, where
            each transform is associated with a list of parent transforms.

    Attributes:
        graph (DiGraph): A directed graph representing the transforms and their dependencies.
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

    def __eq__(self, other: CandidateTransformList) -> bool:
        # Check nodes and edges
        if set(self.graph.nodes) != set(other.graph.nodes):
            return False
        if set(self.graph.edges) != set(other.graph.edges):
            return False
        return True

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
    def serial_sequence(self) -> SerialSequence:
        """
        The serial sequence of parallel transforms.

        Returns a sequence of transforms, where each batch in the sequence represents
        parallelizable transforms, ordered by serial dependency.

        Returns:
            SerialSequence: A sequence of sets representing batches of parallel transforms in execution order.
        """
        # Add input nodes
        seq = [frozenset(self.inputs)]
        parsed_transforms = self.inputs
        # Recursively add children
        remaining_transforms = self.transforms - self.inputs
        while len(remaining_transforms) > 0:
            batch = set()
            # Previous step
            for transform in seq[-1]:
                candidates = self.graph.successors(transform)
                for candidate in candidates:
                    if set(self.graph.predecessors(candidate)) <= parsed_transforms:
                        # All upstream transforms accounted for, so can add to the batch
                        batch.add(candidate)
                        try:
                            remaining_transforms.remove(candidate)
                        except KeyError:
                            pass
            seq.append(frozenset(batch))
            parsed_transforms.update(batch)
        return tuple(seq)

    @property
    def connected_serial_sequences(self) -> set[SerialSequence]:
        """
        Returns the set of connected serial sequences for each of the connected components of the CTL.
        """
        serial_seq = self.serial_sequence
        # for each component, run through the sequence and filter out those items
        ret = set()
        for component in self.connected_components:
            seq = []
            for step in serial_seq:
                filtered = frozenset(trans for trans in step if trans in component)
                if len(filtered) > 0:
                    seq.append(filtered)
            ret.add(tuple(seq))
        return ret

    @property
    def connected_components(self) -> set[frozenset[str]]:
        """
        The set of transforms in each of the set of connected components of the CTL (which will typically be more than
        one when the CTL is the result of combining multiple sub-CTLs.
        """
        return {
            frozenset(component)
            for component in weakly_connected_components(self.graph)
        }

    def immediately_upstream(self, transform: str) -> set[str]:
        """
        The set of transforms which are immediately upstream of the specified transform.
        """
        if transform not in self.transforms:
            raise ValueError(transform)

        return set(self.graph.predecessors(transform))

    def immediately_downstream(self, transform: str) -> set[str]:
        """
        The set of transforms which are immediately downstream of the specified transform.
        """
        if transform not in self.transforms:
            raise ValueError(transform)

        return set(self.graph.successors(transform))
