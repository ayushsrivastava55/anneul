"""De-duplicate ids while keeping the order they arrived in."""


def unique_ids(ids):
    """Return the ids with duplicates removed, in first-seen order.

    The result is a list: callers index into it and compare it to expected lists, so
    the ordering has to survive.
    """
    seen = set()
    ordered = []
    for value in ids:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def duplicate_ids(ids):
    """The ids that appear more than once, in first-seen order."""
    counts = {}
    for value in ids:
        counts[value] = counts.get(value, 0) + 1
    return [value for value in unique_ids(ids) if counts[value] > 1]
