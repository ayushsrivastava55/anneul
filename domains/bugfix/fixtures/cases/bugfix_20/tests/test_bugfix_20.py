from median import median, median_of_field


def test_odd_count():
    assert median([5, 1, 3]) == 3


def test_even_count_averages_the_middle_pair():
    assert median([1, 2, 3, 4]) == 2.5


def test_even_count_of_two():
    assert median([10, 20]) == 15


def test_unsorted_input():
    assert median([9, 1, 8, 2]) == 5


def test_median_of_field():
    assert median_of_field([{"n": 1}, {"n": 2}, {"n": 3}, {"n": 4}], "n") == 2.5
