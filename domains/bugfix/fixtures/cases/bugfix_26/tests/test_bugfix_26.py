from unique_ids import duplicate_ids, unique_ids


def test_order_is_preserved():
    assert unique_ids(["b", "a", "b", "c"]) == ["b", "a", "c"]


def test_result_is_a_list():
    assert isinstance(unique_ids(["a"]), list)


def test_no_duplicates():
    assert unique_ids([1, 2, 3]) == [1, 2, 3]


def test_empty():
    assert unique_ids([]) == []


def test_duplicate_ids():
    assert duplicate_ids(["b", "a", "b"]) == ["b"]
