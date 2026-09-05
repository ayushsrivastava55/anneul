from binary_search import contains, search

VALUES = [1, 3, 5, 7, 9, 11]


def test_finds_the_first_value():
    assert search(VALUES, 1) == 0


def test_finds_a_middle_value():
    assert search(VALUES, 7) == 3


def test_finds_the_last_value():
    assert search(VALUES, 11) == 5


def test_missing_value():
    assert search(VALUES, 4) == -1


def test_contains_the_last_value():
    assert contains(VALUES, 11) is True
