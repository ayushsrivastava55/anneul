"""Pull semantic version numbers out of release notes."""

import re

VERSION_RE = re.compile(r"v(\d+)\.(\d+)\.(\d)")


def find_versions(text):
    """Return every "vMAJOR.MINOR.PATCH" in `text` as a list of (major, minor, patch) ints.

    Each component may have any number of digits: v10.20.30 is as valid as v1.2.3.
    """
    found = []
    for match in VERSION_RE.finditer(text):
        found.append(tuple(int(part) for part in match.groups()))
    return found


def latest_version(text):
    """The highest version mentioned in `text`, or None when there is none."""
    versions = find_versions(text)
    return max(versions) if versions else None
