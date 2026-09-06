"""Division that reports rather than raises."""


def safe_divide(numerator, denominator):
    """Return numerator / denominator, or None when the denominator is zero.

    Callers use None to mean "undefined"; a zero denominator must never raise.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def ratios(pairs):
    """safe_divide over a list of (numerator, denominator) pairs."""
    return [safe_divide(top, bottom) for top, bottom in pairs]


def percent(part, whole):
    """`part` as a percentage of `whole`, or None when `whole` is zero."""
    share = safe_divide(part, whole)
    return None if share is None else share * 100
