"""Slice a list of results into pages."""


def page_slice(items, page, per_page):
    """Return the items on `page` (1-based) when the list is cut into `per_page` blocks.

    Page 1 is the first `per_page` items. A page past the end is empty.
    """
    if page < 1:
        raise ValueError("page is 1-based")
    if per_page < 1:
        raise ValueError("per_page must be at least 1")
    start = page * per_page
    return items[start : start + per_page]


def page_count(total, per_page):
    """How many pages `total` items fill."""
    if per_page < 1:
        raise ValueError("per_page must be at least 1")
    full, remainder = divmod(total, per_page)
    return full + (1 if remainder else 0)
