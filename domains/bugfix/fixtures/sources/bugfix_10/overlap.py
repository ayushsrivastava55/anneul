"""Half-open interval arithmetic."""


def intervals_overlap(first, second):
    """True when two half-open intervals [start, end) share at least one point.

    Touching intervals such as (0, 5) and (5, 9) do NOT overlap: the end is exclusive.
    """
    first_start, first_end = first
    second_start, second_end = second
    if first_start > first_end or second_start > second_end:
        raise ValueError("interval start must not be after its end")
    return first_start < second_end and second_start < first_end


def any_overlap(intervals):
    """True if any pair in `intervals` overlaps."""
    for i, first in enumerate(intervals):
        for second in intervals[i + 1 :]:
            if intervals_overlap(first, second):
                return True
    return False
