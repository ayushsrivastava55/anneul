from graph import add_edge, neighbours


def test_single_edge_is_undirected():
    adjacency = add_edge("a", "b")
    assert neighbours(adjacency, "a") == ["b"]
    assert neighbours(adjacency, "b") == ["a"]


def test_separate_graphs_do_not_share_edges():
    add_edge("a", "b")
    second = add_edge("c", "d")
    assert sorted(second) == ["c", "d"]


def test_existing_graph_is_extended():
    adjacency = add_edge("a", "b")
    add_edge("b", "c", adjacency)
    assert neighbours(adjacency, "b") == ["a", "c"]
