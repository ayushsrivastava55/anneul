"""Median of a list of numbers."""


def median(values):
    """Return the median of `values`.

    With an odd count that is the middle value. With an even count it is the mean of
    the two middle values. An empty list raises ValueError.
    """
    if not values:
        raise ValueError("median of an empty list")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


def median_of_field(rows, field):
    """Median of `field` across `rows` that have it."""
    return median([row[field] for row in rows if field in row])
