"""Build storage paths for uploaded files."""

import posixpath


def storage_path(base, name):
    """Join `base` and `name` into a storage path.

    `base` is the directory, `name` the file inside it. A trailing slash on the base
    must not double up.
    """
    if not name:
        raise ValueError("name must not be empty")
    return posixpath.join(base.rstrip("/"), name)


def storage_paths(base, names):
    """storage_path for each name."""
    return [storage_path(base, name) for name in names]


def basename(path):
    """The final component of `path`."""
    return posixpath.basename(path)
