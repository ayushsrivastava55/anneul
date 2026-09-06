"""Remove a known prefix from a name."""


def strip_prefix(name, prefix):
    """Return `name` without the leading `prefix`.

    When `name` does not actually start with `prefix` it is returned unchanged, and
    an empty prefix is a no-op.
    """
    if not prefix:
        return name
    if not name.startswith(prefix):
        return name
    return name[len(prefix) :]


def strip_all(names, prefix):
    """strip_prefix over a list of names."""
    return [strip_prefix(name, prefix) for name in names]


def common_prefix_count(names, prefix):
    """How many names actually carry `prefix`."""
    return sum(1 for name in names if prefix and name.startswith(prefix))
