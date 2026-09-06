from rounding import round_money, split_evenly, total


def test_rounds_to_two_places():
    assert round_money(1.239) == 1.24


def test_explicit_digits():
    assert round_money(1.2345, 3) == 1.234


def test_split_evenly():
    assert split_evenly(10, 3) == 3.33


def test_total():
    assert total([1.005, 2.001]) == 3.01
