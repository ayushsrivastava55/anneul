"""Whole-day arithmetic on date objects."""

from datetime import date, timedelta


def days_between(start, end):
    """Return how many days `end` is after `start`.

    A later end date gives a positive number; an earlier one gives a negative number.
    """
    if not isinstance(start, date) or not isinstance(end, date):
        raise TypeError("start and end must be dates")
    return (start - end).days


def is_overdue(due, today):
    """True when `today` is strictly after the due date."""
    return days_between(due, today) > 0


def shift(day, days):
    """The date `days` after `day`."""
    return day + timedelta(days=days)
