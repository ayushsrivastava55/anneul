from overlap import any_overlap, intervals_overlap


def test_clear_overlap():
    assert intervals_overlap((0, 5), (3, 9)) is True


def test_touching_intervals_do_not_overlap():
    assert intervals_overlap((0, 5), (5, 9)) is False


def test_disjoint():
    assert intervals_overlap((0, 2), (7, 9)) is False


def test_any_overlap_on_touching_intervals():
    assert any_overlap([(0, 5), (5, 9), (9, 12)]) is False
