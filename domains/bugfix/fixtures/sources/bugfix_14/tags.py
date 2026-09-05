"""Collect tags off a record and its children."""


def collect_tags(record, seen=None):
    """Return the set of tags on `record` and, recursively, its "children".

    Each top-level call starts from an empty set: results must not accumulate across
    separate calls.
    """
    if seen is None:
        seen = set()
    for tag in record.get("tags", []):
        seen.add(tag)
    for child in record.get("children", []):
        collect_tags(child, seen)
    return seen


def tag_list(record):
    """The tags on `record`, sorted."""
    return sorted(collect_tags(record))
