"""Redact secrets out of log lines."""

MASK = "***"


def redact(line, secret):
    """Return `line` with every occurrence of `secret` replaced by MASK.

    An empty secret leaves the line alone.
    """
    if not secret:
        return line
    return line.replace(MASK, secret)


def redact_all(lines, secret):
    """redact over a list of lines."""
    return [redact(line, secret) for line in lines]


def contains_secret(line, secret):
    """True when `secret` still appears in `line`."""
    return bool(secret) and secret in line
