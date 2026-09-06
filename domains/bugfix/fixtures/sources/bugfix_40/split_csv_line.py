"""Split a comma-separated line, tolerating whitespace around the commas."""

import re

SPLIT_RE = re.compile(r"\s*,\s*")


def split_fields(line):
    """Return the fields of `line`, with the whitespace around each comma removed.

    Leading and trailing whitespace on the line itself is stripped first. An empty
    line yields an empty list.
    """
    trimmed = line.strip()
    if not trimmed:
        return []
    return SPLIT_RE.split(trimmed)


def field_count(line):
    """How many fields `line` has."""
    return len(split_fields(line))
