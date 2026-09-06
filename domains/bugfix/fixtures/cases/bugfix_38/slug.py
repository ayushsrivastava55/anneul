"""Validate URL slugs."""

import re

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*")


def is_valid_slug(text):
    """True when `text` is a lowercase, hyphen-separated slug and nothing else.

    Uppercase letters, spaces, leading or trailing hyphens and doubled hyphens are
    all invalid, and the check covers the whole string.
    """
    return SLUG_RE.match(text) is not None


def valid_slugs(candidates):
    """The valid slugs from `candidates`, in order."""
    return [candidate for candidate in candidates if is_valid_slug(candidate)]


def slugify(title):
    """A best-effort slug for `title`."""
    lowered = re.sub(r"[^a-z0-9]+", "-", title.lower())
    return lowered.strip("-")
