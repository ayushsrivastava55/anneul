"""Simple moving average over a list of numbers."""


def moving_average(values, window):
    """Return the mean of every consecutive `window`-long slice of `values`.

    A list of n values yields n - window + 1 averages. If the window is longer than
    the input, the result is empty.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    averages = []
    for start in range(len(values) - window):
        block = values[start : start + window]
        averages.append(sum(block) / window)
    return averages


def latest_average(values, window):
    """The average of the final `window` values, or None if there are too few."""
    averages = moving_average(values, window)
    return averages[-1] if averages else None
