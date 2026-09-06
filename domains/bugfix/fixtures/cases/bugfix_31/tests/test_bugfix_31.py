from datetime import date

from date_span import days_between, is_overdue, shift


def test_forward_span_is_positive():
    assert days_between(date(2024, 1, 1), date(2024, 1, 10)) == 9


def test_backward_span_is_negative():
    assert days_between(date(2024, 1, 10), date(2024, 1, 1)) == -9


def test_same_day():
    assert days_between(date(2024, 1, 1), date(2024, 1, 1)) == 0


def test_is_overdue():
    assert is_overdue(date(2024, 1, 1), date(2024, 1, 2)) is True
    assert is_overdue(date(2024, 1, 2), date(2024, 1, 1)) is False


def test_shift():
    assert shift(date(2024, 1, 1), 3) == date(2024, 1, 4)
