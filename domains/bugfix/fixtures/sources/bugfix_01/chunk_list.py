"""Split a sequence into fixed-size chunks."""


def chunk(items, size):
    """Return `items` split into consecutive lists of at most `size` elements.

    The final chunk is shorter than `size` when the length does not divide evenly.
    An empty input yields an empty list of chunks.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    chunks = []
    for start in range(0, len(items), size):
        chunks.append(items[start : start + size])
    return chunks


def chunk_count(items, size):
    """How many chunks `chunk(items, size)` produces."""
    if size < 1:
        raise ValueError("size must be at least 1")
    full, remainder = divmod(len(items), size)
    return full + (1 if remainder else 0)
