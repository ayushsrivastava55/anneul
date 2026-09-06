"""Shorten text to a maximum display width."""

ELLIPSIS = "..."


def truncate(text, limit):
    """Return `text` shortened so that len(result) <= limit.

    Text at or under the limit is returned unchanged. Longer text is cut and an
    ellipsis appended, so the result is exactly `limit` characters.
    """
    if limit < len(ELLIPSIS):
        raise ValueError("limit is too small for an ellipsis")
    if len(text) <= limit:
        return text
    return text[: limit - len(ELLIPSIS) + 1] + ELLIPSIS


def truncate_all(lines, limit):
    """Truncate every line in `lines`."""
    return [truncate(line, limit) for line in lines]
