from chunk_list import chunk, chunk_count


def test_even_split():
    assert chunk([1, 2, 3, 4, 5, 6], 2) == [[1, 2], [3, 4], [5, 6]]


def test_ragged_last_chunk():
    assert chunk(["a", "b", "c", "d", "e"], 3) == [["a", "b", "c"], ["d", "e"]]


def test_empty_input():
    assert chunk([], 4) == []


def test_count_matches_chunk():
    items = list(range(10))
    assert len(chunk(items, 3)) == chunk_count(items, 3)
