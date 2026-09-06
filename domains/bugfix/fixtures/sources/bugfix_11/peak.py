"""Find the largest reading in a series."""


def peak_value(readings):
    """Return the largest reading, or None when there are none."""
    best = None
    for reading in readings:
        if best is None:
            best = reading
            continue
        if reading > best:
            best = reading
    return best


def peak_index(readings):
    """Index of the first occurrence of the largest reading, or -1 when empty."""
    best = peak_value(readings)
    if best is None:
        return -1
    return readings.index(best)
