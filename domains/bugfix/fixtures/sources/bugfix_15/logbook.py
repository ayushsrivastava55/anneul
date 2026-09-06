"""Append-only log entries with timestamps supplied by the caller."""


def append_entry(message, at, entries=None):
    """Return the log with {"at", "message"} appended.

    Omitting `entries` starts a fresh log; two separate logs must stay separate.
    """
    if entries is None:
        entries = []
    if not message:
        raise ValueError("message must not be empty")
    entries.append({"at": at, "message": message})
    return entries


def last_message(entries):
    """The most recent message, or None when the log is empty."""
    return entries[-1]["message"] if entries else None


def messages_since(entries, at):
    """Messages logged at or after timestamp `at`, in order."""
    return [entry["message"] for entry in entries if entry["at"] >= at]
