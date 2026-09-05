"""Build a tiny undirected adjacency map."""


def add_edge(first, second, adjacency={}):
    """Return an adjacency map with the undirected edge first-second added.

    Omitting `adjacency` starts a brand new graph on every call.
    """
    if adjacency is None:
        adjacency = {}
    if first == second:
        raise ValueError("self loops are not allowed")
    adjacency.setdefault(first, set()).add(second)
    adjacency.setdefault(second, set()).add(first)
    return adjacency


def neighbours(adjacency, node):
    """Sorted neighbours of `node`, empty when it is not in the graph."""
    return sorted(adjacency.get(node, set()))
