"""Gregorian leap years."""


def is_leap_year(year):
    """True when `year` is a leap year in the Gregorian calendar.

    Every year divisible by 4 is a leap year, except century years, which must also
    be divisible by 400. So 2000 is a leap year and 1900 is not.
    """
    if year < 1:
        raise ValueError("year must be positive")
    if year % 4 != 0:
        return False
    if year % 100 == 0:
        return year % 400 != 0
    return True


def days_in_year(year):
    """366 in a leap year, 365 otherwise."""
    return 366 if is_leap_year(year) else 365
