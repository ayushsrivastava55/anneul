import pytest
from mean import average, average_of_field, rounded_average


def test_ordinary_average():
    assert average([2, 4, 6]) == 4.0


def test_empty_series_is_zero():
    assert average([]) == 0.0


def test_average_of_field_skips_missing():
    rows = [{"n": 2}, {"other": 9}, {"n": 4}]
    assert average_of_field(rows, "n") == 3.0


def test_average_of_a_field_nobody_has():
    assert average_of_field([{"other": 1}], "n") == 0.0


def test_rounded_average():
    assert rounded_average([1, 2]) == pytest.approx(1.5)
