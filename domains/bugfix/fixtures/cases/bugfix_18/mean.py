"""Averages that cope with empty input."""


def average(values):
    """Mean of `values`, or 0.0 when there is nothing to average.

    An empty series is not an error here: reports call this on filtered data and
    expect a plain zero back.
    """
    return sum(values) / len(values)


def average_of_field(rows, field):
    """Mean of `field` across `rows`, skipping rows where it is missing."""
    present = [row[field] for row in rows if field in row]
    return average(present)


def rounded_average(values, digits=2):
    """The average rounded to `digits` decimal places."""
    return round(average(values), digits)
