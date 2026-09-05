from paginate import page_count, page_slice

ITEMS = ["a", "b", "c", "d", "e"]


def test_first_page_starts_at_the_beginning():
    assert page_slice(ITEMS, 1, 2) == ["a", "b"]


def test_middle_page():
    assert page_slice(ITEMS, 2, 2) == ["c", "d"]


def test_page_past_the_end_is_empty():
    assert page_slice(ITEMS, 9, 2) == []


def test_page_count():
    assert page_count(len(ITEMS), 2) == 3
