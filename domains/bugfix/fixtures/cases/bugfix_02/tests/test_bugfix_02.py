from moving_average import latest_average, moving_average


def test_all_windows_are_returned():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


def test_window_equal_to_length():
    assert moving_average([2, 4], 2) == [3.0]


def test_window_longer_than_input():
    assert moving_average([1], 3) == []


def test_latest_average_uses_the_final_window():
    assert latest_average([1, 2, 3, 10], 2) == 6.5
